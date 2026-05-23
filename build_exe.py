"""
build_exe.py — 遥感影像作物分类系统 打包脚本

使用方法:
  python build_exe.py                    # 交互式选择打包模式
  python build_exe.py --mode onedir      # 文件夹模式 (推荐，启动快)
  python build_exe.py --mode onefile     # 单文件模式 (便携，启动慢)
  python build_exe.py --mode all         # 两种模式都生成
  python build_exe.py --check-only       # 仅检查依赖和环境

输出:
  dist/遥感影像作物分类系统_portable/      # 文件夹版 (onedir)
  dist/遥感影像作物分类系统.exe            # 单文件版 (onefile)

注意事项:
  - 打包后体积较大 (~3-8GB)，主要来自PyTorch + CUDA
  - 推荐使用 onedir 模式，启动速度快
  - 如果不需要GPU推理，可安装CPU版PyTorch大幅减小体积
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def check_environment():
    """检查打包环境"""
    print("=" * 60)
    print("检查打包环境...")
    print("=" * 60)

    checks = []

    # Python 版本
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}")

    # PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller: {PyInstaller.__version__}")
        checks.append(True)
    except ImportError:
        print("PyInstaller: 未安装! 运行: pip install pyinstaller")
        checks.append(False)

    # PyTorch
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"PyTorch: {torch.__version__} (CUDA: {'是' if cuda else '否'})")
        if cuda:
            # 估算体积
            lib_size = sum(
                os.path.getsize(os.path.join(os.path.dirname(torch.__file__), 'lib', f))
                for f in os.listdir(os.path.join(os.path.dirname(torch.__file__), 'lib'))
                if os.path.isfile(os.path.join(os.path.dirname(torch.__file__), 'lib', f))
            ) / 1024**3
            print(f"  PyTorch CUDA库体积: ~{lib_size:.1f} GB")
            print(f"  提示: CPU版PyTorch可减小3-4GB")
        checks.append(True)
    except ImportError:
        print("PyTorch: 未安装!")
        checks.append(False)

    # PyQt5
    try:
        import PyQt5
        print("PyQt5: 已安装")
        checks.append(True)
    except ImportError:
        print("PyQt5: 未安装 (桌面GUI需要)")
        checks.append(False)

    # 模型文件
    model_files = ["best_model.pth", "checkpoints/best_phase2_edl.pth"]
    for mf in model_files:
        mp = PROJECT_ROOT / mf
        if mp.exists():
            size_mb = os.path.getsize(mp) / 1024**2
            print(f"模型文件 {mf}: {size_mb:.0f} MB (将被打包)")
        else:
            print(f"模型文件 {mf}: 不存在 (EXE将使用随机权重)")

    # 磁盘空间
    try:
        free_space = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
        print(f"可用磁盘空间: {free_space:.1f} GB")
        if free_space < 10:
            print("  警告: 磁盘空间不足! 建议至少20GB")
    except Exception:
        pass

    all_ok = all(checks)
    print()
    if all_ok:
        print("环境检查通过!")
    else:
        print("部分依赖缺失，请安装后再打包")
    print()

    return all_ok


def estimate_size():
    """估算打包后文件大小"""
    try:
        import torch
        total = 0
        # PyTorch
        torch_dir = os.path.dirname(torch.__file__)
        for root, dirs, files in os.walk(torch_dir):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        # numpy
        import numpy
        for root, dirs, files in os.walk(os.path.dirname(numpy.__file__)):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        # 模型
        for mf in ["best_model.pth", "checkpoints/best_phase2_edl.pth"]:
            mp = PROJECT_ROOT / mf
            if mp.exists():
                total += os.path.getsize(mp)

        gb = total / 1024**3
        print(f"  估算原始体积: ~{gb:.1f} GB")
        print(f"  压缩后估计: ~{gb * 0.4:.1f} GB (onefile)")
        print(f"  onedir文件夹: ~{gb:.1f} GB")
    except Exception:
        pass


def install_pyinstaller():
    """安装 PyInstaller"""
    print("安装 PyInstaller...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller"],
                       check=True)
        print("PyInstaller 安装完成!")
        return True
    except Exception as e:
        print(f"PyInstaller 安装失败: {e}")
        return False


def run_build(mode: str):
    """执行打包"""
    spec_file = PROJECT_ROOT / "作物分类系统_完整版.spec"

    if not spec_file.exists():
        print(f"错误: spec文件不存在: {spec_file}")
        return False

    # 清理旧的构建文件
    for d in ["build", "dist"]:
        dp = PROJECT_ROOT / d
        if dp.exists():
            print(f"清理旧构建: {d}/")
            shutil.rmtree(dp)

    print()
    print("=" * 60)
    print(f"开始打包 (模式: {mode})")
    print("=" * 60)
    estimate_size()
    print()

    if mode == "onefile":
        # 仅生成单文件EXE
        cmd = [
            sys.executable, "-m", "PyInstaller",
            str(spec_file),
            "--noconfirm",
            "--clean",
            "--log-level", "INFO",
        ]
    elif mode == "onedir":
        # 仅生成文件夹版（更快）
        cmd = [
            sys.executable, "-m", "PyInstaller",
            str(spec_file),
            "--noconfirm",
            "--clean",
            "--log-level", "INFO",
        ]
    else:
        print(f"未知模式: {mode}")
        return False

    print(f"执行: {' '.join(cmd)}")
    print("(这可能需要30分钟到2小时，取决于CPU和磁盘速度)")
    print()

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        print()
        print("=" * 60)
        print("打包完成!")
        print("=" * 60)

        dist_dir = PROJECT_ROOT / "dist"
        if dist_dir.exists():
            print(f"\n输出文件位于: {dist_dir}")
            for item in dist_dir.iterdir():
                if item.is_file():
                    size_mb = item.stat().st_size / 1024**2
                    print(f"  {item.name} ({size_mb:.0f} MB)")
                elif item.is_dir():
                    total = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    print(f"  {item.name}/ ({total/1024**2:.0f} MB)")

        print(f"\n使用方法:")
        if mode == "onefile":
            print(f"  双击运行: dist/遥感影像作物分类系统.exe")
        else:
            print(f"  进入目录: cd dist/遥感影像作物分类系统_portable")
            print(f"  双击运行: 遥感影像作物分类系统.exe")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n打包失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n打包取消")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="遥感影像作物分类系统 打包工具")
    parser.add_argument("--mode", type=str,
                        choices=["onedir", "onefile", "all"],
                        default=None,
                        help="打包模式: onedir(文件夹,推荐), onefile(单文件), all(全部)")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")
    parser.add_argument("--install-deps", action="store_true", help="安装缺少的依赖")
    args = parser.parse_args()

    print()
    print("╔" + "═" * 58 + "╗")
    print("║  遥感影像作物分类系统 — EXE打包工具                    ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 安装依赖
    if args.install_deps:
        install_pyinstaller()

    # 检查环境
    env_ok = check_environment()
    if not env_ok:
        print("环境不完整，请先安装缺失依赖:")
        print("  pip install pyinstaller torch PyQt5")
        if not args.install_deps:
            ans = input("是否自动安装PyInstaller? (y/n): ").strip().lower()
            if ans == "y":
                install_pyinstaller()
                env_ok = check_environment()

    if args.check_only:
        return 0 if env_ok else 1

    if not env_ok:
        print("无法继续打包。")
        return 1

    # 选择模式
    mode = args.mode
    if mode is None:
        print("请选择打包模式:")
        print("  1. onedir  — 文件夹模式 (推荐)")
        print("     启动快(5-15秒)，占用空间大(~4-8GB)")
        print("     包含所有依赖和模型文件在一个文件夹中")
        print()
        print("  2. onefile — 单文件模式")
        print("     便捷(一个.EXE文件)，启动慢(30秒-2分钟)")
        print("     首次启动需解压临时文件")
        print()
        print("  3. all     — 两种都生成")
        print()
        choice = input("选择 (1/2/3, 默认1): ").strip()
        mode_map = {"1": "onedir", "2": "onefile", "3": "all"}
        mode = mode_map.get(choice, "onedir")
        print(f"选择: {mode}")

    if mode == "all":
        print("\n提示: all模式会占用50GB+磁盘空间，生成时间可能超过1小时")
        confirm = input("确认? (y/n): ").strip().lower()
        if confirm != "y":
            print("取消。")
            return 0
        run_build("onedir")
        run_build("onefile")
    else:
        run_build(mode)

    return 0


if __name__ == "__main__":
    sys.exit(main())
