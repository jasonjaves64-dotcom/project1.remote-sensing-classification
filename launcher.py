# =============================================================================
# launcher.py — 遥感影像作物分类系统 统一启动器
#
# 打包为EXE后双击即可运行，提供功能选择菜单。
# 支持模式：
#   1. 桌面GUI推理 (PyQt5)       — 图形界面，加载数据 → 推理 → 可视化
#   2. Web界面 (Streamlit)        — 浏览器中运行，完整功能
#   3. API服务 (FastAPI)          — 后台启动REST API
#   4. 命令行推理                 — 批量处理
#   5. 校准分析                   — EDL校准报告
#   6. 模型诊断                   — 检查模型健康状态
# =============================================================================
import sys
import os
import subprocess
import importlib


def print_banner():
    print(r"""
  ╔══════════════════════════════════════════════════════════╗
  ║       遥感影像作物分类系统 v2.0                           ║
  ║       FusionCropNetV5EDL + 校准验证 + 可解释性            ║
  ║       Remote Sensing Crop Classification System          ║
  ╚══════════════════════════════════════════════════════════╝
    """)


def check_device():
    import torch
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"


def check_model():
    model_paths = [
        "best_model.pth",
        "checkpoints/best_phase2_edl.pth",
        "checkpoints/best_phase1_edl.pth",
    ]
    for p in model_paths:
        if os.path.exists(p):
            return p
    return None


def run_desktop_gui():
    print("启动桌面GUI...")
    try:
        from desktop_app import main as desktop_main
        desktop_main()
    except ImportError as e:
        print(f"桌面GUI启动失败: {e}")
        print("请确保已安装 PyQt5: pip install PyQt5")
        input("按回车键返回...")


def run_web_ui():
    print("启动Web界面 (Streamlit)...")
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    if getattr(sys, 'frozen', False):
        # Running from PyInstaller: launch streamlit in-process
        import streamlit.web.cli as stcli
        old_argv = sys.argv
        sys.argv = ["streamlit", "run", app_path, "--server.port", "8501",
                     "--server.headless", "true", "--browser.serverAddress", "localhost",
                     "--global.developmentMode", "false"]
        try:
            stcli.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
    else:
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", app_path,
                            "--server.port", "8501", "--browser.serverAddress", "localhost",
                            "--global.developmentMode", "false"],
                           check=False)
        except Exception as e:
            print(f"Web界面启动失败: {e}")
            input("按回车键返回...")


def run_api_server():
    print("启动API服务 (FastAPI)...")
    api_path = os.path.join(os.path.dirname(__file__), "api", "main.py")
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app",
                        "--host", "0.0.0.0", "--port", "8000"], check=False)
    except Exception as e:
        print(f"API服务启动失败: {e}")
        input("按回车键返回...")


def _create_model(model_type="v5edl", backbone="resnet50"):
    """Create model instance by type. Supports: v5, v5edl, v5pro."""
    if model_type == "v5pro":
        from models.fusion_net_v5pro import FusionCropNetV5Pro
        return FusionCropNetV5Pro(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone=backbone, pretrained=False,
            n_heads=16, win_size=4, n_layers=4)
    elif model_type == "v5edl":
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        return FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4)
    else:
        from models.fusion_net_v5 import FusionCropNetV5
        return FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4)


def run_inference(model_type="v5edl"):
    print("\n=== 命令行推理 ===")
    from models.fusion_net_v5_edl import dirichlet_to_predictions
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    model_path = check_model()
    if model_path:
        print(f"模型: {model_path}")
    else:
        print("未找到训练好的模型，使用随机初始化权重")

    model = _create_model(model_type).to(device)

    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)

    model.eval()

    print("\n请选择输入数据来源:")
    print("  1. 使用随机测试数据")
    print("  2. 加载NPY文件 (data/processed/)")
    choice = input("选择 (1/2): ").strip()

    H, W = 128, 128
    T = 12

    if choice == "2":
        try:
            opt_seq = np.load("data/processed/opt_sequence.npy")
            sar_seq = np.load("data/processed/sar_sequence.npy")
            doy = np.load("data/processed/doy_norm.npy")
            dem = np.load("data/processed/dem.npy") if os.path.exists("data/processed/dem.npy") else np.zeros((5, opt_seq.shape[2], opt_seq.shape[3]), dtype=np.float32)
            print(f"数据加载成功: opt={opt_seq.shape}, sar={sar_seq.shape}")
        except Exception as e:
            print(f"数据加载失败: {e}, 使用随机数据")
            choice = "1"

    if choice != "2":
        opt_seq = np.random.randn(T, 10, H, W).astype(np.float32) * 0.1
        sar_seq = np.random.randn(T, 5, H, W).astype(np.float32) * 0.1
        dem = np.random.randn(5, H, W).astype(np.float32) * 0.1
        doy = np.linspace(0, 1, T).astype(np.float32)

    opt_t = torch.from_numpy(opt_seq).unsqueeze(0).float().to(device)
    sar_t = torch.from_numpy(sar_seq).unsqueeze(0).float().to(device)
    dem_t = torch.from_numpy(dem).unsqueeze(0).float().to(device)
    doy_t = torch.from_numpy(doy).unsqueeze(0).float().to(device)

    print("\n推理中...")
    use_edl = input("使用EDL不确定性估计? (y/n, 默认y): ").strip().lower() != "n"

    with torch.no_grad():
        if use_edl:
            result = model.predict_uncertainty(opt_t, sar_t, dem_t, doy_t,
                                                n_passes=5, use_tta=True)
            pred = result["pred_class"][0].cpu().numpy()
            vacuity = result["vacuity"][0].cpu().numpy()
            dissonance = result["dissonance"][0].cpu().numpy()
        else:
            alpha = model(opt_t, sar_t, dem_t, doy_t)
            pred = alpha.argmax(dim=1)[0].cpu().numpy()

    os.makedirs("output", exist_ok=True)
    np.save("output/prediction.npy", pred)
    print(f"预测结果已保存到 output/prediction.npy")
    if use_edl:
        np.save("output/vacuity.npy", vacuity)
        np.save("output/dissonance.npy", dissonance)
        print(f"不确定性图已保存到 output/vacuity.npy, output/dissonance.npy")
        print(f"  平均Vacuity: {vacuity.mean():.4f}")
        print(f"  平均Dissonance: {dissonance.mean():.4f}")

    print("\n推理完成!")
    input("按回车键返回...")


def run_calibration_analysis():
    print("\n=== EDL校准验证分析 ===")
    from models.fusion_net_v5_edl import FusionCropNetV5EDL
    from utils.calibration import calibration_report, print_calibration_report
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = check_model()
    model = _create_model("v5edl").to(device)

    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"已加载模型: {model_path}")

    model.eval()

    print("\n请输入数据:")
    label_path = input("标签NPY路径 (留空使用随机标签): ").strip()
    data_dir = input("数据目录 (留空使用随机数据): ").strip()

    B, T, H, W = 2, 10, 64, 64

    if data_dir and os.path.exists(os.path.join(data_dir, "opt_sequence.npy")):
        opt_seq = np.load(os.path.join(data_dir, "opt_sequence.npy"))
        sar_seq = np.load(os.path.join(data_dir, "sar_sequence.npy"))
        doy = np.load(os.path.join(data_dir, "doy_norm.npy"))
        dem = np.load(os.path.join(data_dir, "dem.npy")) if os.path.exists(os.path.join(data_dir, "dem.npy")) else np.zeros((5, H, W), dtype=np.float32)
        if label_path and os.path.exists(label_path):
            labels = np.load(label_path)
        else:
            labels = np.random.randint(0, 7, (H, W))
    else:
        opt_seq = np.random.randn(B, T, 10, H, W).astype(np.float32) * 0.1
        sar_seq = np.random.randn(B, T, 5, H, W).astype(np.float32) * 0.1
        dem = np.random.randn(B, 5, H, W).astype(np.float32) * 0.1
        doy = np.random.rand(B, T).astype(np.float32)
        labels = np.random.randint(0, 7, (B, H, W))

    opt_t = torch.from_numpy(opt_seq).float().to(device)
    sar_t = torch.from_numpy(sar_seq).float().to(device)
    dem_t = torch.from_numpy(dem).float().to(device)
    doy_t = torch.from_numpy(doy).float().to(device)

    print("推理中...")
    with torch.no_grad():
        alpha = model(opt_t, sar_t, dem_t, doy_t)

    alpha_np = alpha.cpu().numpy()
    cal = calibration_report(alpha_np, labels, 7, n_bins=15)
    print_calibration_report(cal)

    os.makedirs("calibration_output", exist_ok=True)
    import json
    serializable = {k: v for k, v in cal.items() if k != "_raw"}
    with open("calibration_output/calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\n校准报告已保存到 calibration_output/calibration_report.json")
    input("按回车键返回...")


def run_model_diagnosis():
    print("\n=== 模型诊断 ===")
    from models.fusion_net_v5_edl import FusionCropNetV5EDL
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model("v5edl").to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params/1e6:.1f}M")

    model_path = check_model()
    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"已加载权重: {model_path}")

    print("运行推理测试...")
    model.eval()
    B, T, H, W = 1, 10, 32, 32
    opt_t = torch.randn(B, T, 10, H, W).to(device) * 0.1
    sar_t = torch.randn(B, T, 5, H, W).to(device) * 0.1
    dem_t = torch.randn(B, 5, H, W).to(device) * 0.1
    doy_t = torch.rand(B, T).to(device)

    with torch.no_grad():
        alpha = model(opt_t, sar_t, dem_t, doy_t)

    from models.fusion_net_v5_edl import dirichlet_to_predictions
    preds = dirichlet_to_predictions(alpha)
    print(f"  输出形状: alpha={alpha.shape}")
    print(f"  预测类别: {preds['pred_class'].unique().tolist()}")
    print(f"  Vacuity: {preds['vacuity'].mean().item():.4f}")
    print(f"  Dissonance: {preds['dissonance'].mean().item():.4f}")

    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU内存占用: {mem:.1f} MB")

    print("模型诊断通过")
    input("按回车键返回...")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    clear_screen()
    print_banner()

    try:
        device_info = check_device()
    except Exception:
        device_info = "未知"

    model_path = check_model()
    print(f"  设备: {device_info}")
    print(f"  模型: {model_path if model_path else '未找到训练权重 (将随机初始化)'}")
    print()

    menu_items = [
        ("1", "桌面GUI推理", "PyQt5图形界面，可视化推理与不确定性分析", run_desktop_gui),
        ("2", "Web界面", "浏览器中运行Streamlit应用 (完整功能)", run_web_ui),
        ("3", "API服务", "启动FastAPI后端服务 (端口8000)", run_api_server),
        ("4", "命令行推理", "批量推理，支持EDL不确定性估计", run_inference),
        ("5", "EDL校准分析", "生成校准报告 (ECE/NLL/Brier/拒绝曲线)", run_calibration_analysis),
        ("6", "模型诊断", "检查模型健康状态和参数", run_model_diagnosis),
        ("0", "退出", "关闭程序", None),
    ]

    while True:
        print("── 请选择功能 ──")
        for idx, name, desc, _ in menu_items:
            print(f"  [{idx}] {name} — {desc}")
        print()

        choice = input("输入数字选择 (0-6): ").strip()

        item = next((m for m in menu_items if m[0] == choice), None)
        if item is None:
            print("无效选择，请重试")
            continue

        if item[0] == "0":
            print("再见！")
            break

        clear_screen()
        print_banner()
        print(f"启动: {item[1]}\n")
        try:
            item[3]()
        except KeyboardInterrupt:
            print("\n操作取消")
        except Exception as e:
            print(f"\n错误: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            input("\n按回车键返回...")

        clear_screen()
        print_banner()
        print(f"  设备: {device_info}")
        print(f"  模型: {model_path if model_path else '未找到训练权重'}")
        print()


if __name__ == "__main__":
    main()
