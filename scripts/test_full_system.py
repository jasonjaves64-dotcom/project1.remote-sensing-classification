"""
完整系统测试脚本 - 测试 FusionCropNetV5 的所有组件
"""

import torch
import numpy as np
import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

LOG_DIR = Path("test_logs")
LOG_DIR.mkdir(exist_ok=True)

def log(message, level="INFO"):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}\n"
    
    print(log_line.strip())
    
    with open(LOG_DIR / "test_log.txt", "a", encoding="utf-8") as f:
        f.write(log_line)

def test_model_initialization(use_edl=False):
    """测试模型初始化"""
    log("=== 测试 1: 模型初始化 ===")
    try:
        if use_edl:
            from models.fusion_net_v5_edl import FusionCropNetV5EDL
            model = FusionCropNetV5EDL(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet50", pretrained=False,
                n_heads=16, win_size=4, n_layers=4,
                edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50
            )
        else:
            from models.fusion_net_v5 import FusionCropNetV5
            model = FusionCropNetV5(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet50", pretrained=False,
                n_heads=16, win_size=4, n_layers=4
            )
        
        log(f"模型初始化成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        log(f"模型类型: {'EDL' if use_edl else 'Standard'}")
        return True, None
    except Exception as e:
        log(f"模型初始化失败: {str(e)}", "ERROR")
        return False, str(e)

def test_forward_pass(use_edl=False):
    """测试前向传播"""
    log("=== 测试 2: 前向传播 ===")
    try:
        if use_edl:
            from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
        else:
            from models.fusion_net_v5 import FusionCropNetV5
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"使用设备: {device}")
        
        if use_edl:
            model = FusionCropNetV5EDL(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet50", pretrained=False,
                n_heads=16, win_size=4, n_layers=4,
                edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50
            ).to(device)
        else:
            model = FusionCropNetV5(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet50", pretrained=False,
                n_heads=16, win_size=4, n_layers=4
            ).to(device)
        
        B, T, H, W = 2, 12, 64, 64
        
        opt_seq = torch.randn(B, T, 10, H, W).to(device) * 0.1
        sar_seq = torch.randn(B, T, 5, H, W).to(device) * 0.1
        dem = torch.randn(B, 5, H, W).to(device) * 0.1
        doy = torch.rand(B, T).to(device)
        
        model.train()
        start_time = time.time()
        if use_edl:
            alpha, ndvi_pred, consistency_loss = model(opt_seq, sar_seq, dem, doy, epoch=10)
        else:
            cloud_mask = torch.randint(0, 2, (B, T, H, W)).to(device)
            valid_count = torch.randint(1, T+1, (B, H, W)).to(device)
            alpha, ndvi_pred, consistency_loss = model(opt_seq, sar_seq, dem, doy, cloud_mask, valid_count)
        train_time = time.time() - start_time
        
        log(f"训练模式前向传播成功")
        log(f"  alpha/Logits 形状: {alpha.shape}")
        log(f"  ndvi_pred 形状: {ndvi_pred.shape}")
        log(f"  consistency_loss: {consistency_loss.item() if consistency_loss is not None else 'None'}")
        log(f"  耗时: {train_time:.4f}s")
        
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            if use_edl:
                alpha_eval = model(opt_seq, sar_seq, dem, doy)
                preds = dirichlet_to_predictions(alpha_eval)
            else:
                logits_eval = model(opt_seq, sar_seq, dem, doy)
        eval_time = time.time() - start_time
        
        log(f"推理模式前向传播成功")
        if use_edl:
            log(f"  alpha 形状: {alpha_eval.shape}")
            log(f"  平均 vacuity: {preds['vacuity'].mean().item():.4f}")
            log(f"  平均 dissonance: {preds['dissonance'].mean().item():.4f}")
        else:
            log(f"  logits 形状: {logits_eval.shape}")
        log(f"  耗时: {eval_time:.4f}s")
        
        if use_edl:
            pred = preds['pred_class']
        else:
            pred = logits_eval.argmax(dim=1)
        unique_classes = torch.unique(pred)
        log(f"预测类别覆盖: {sorted(unique_classes.tolist())}")
        
        if use_edl and hasattr(model, 'predict_uncertainty'):
            log(f"测试不确定性推理...")
            result = model.predict_uncertainty(opt_seq, sar_seq, dem, doy, n_passes=3, use_tta=False)
            log(f"  不确定性推理成功")
            log(f"  最终 vacuity: {result['vacuity'].mean().item():.4f}")
            log(f"  最终 dissonance: {result['dissonance'].mean().item():.4f}")
        
        return True, None
    except Exception as e:
        log(f"前向传播失败: {str(e)}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return False, str(e)

def test_components():
    """测试各组件"""
    log("=== 测试 3: 组件测试 ===")
    try:
        from models._base import (
            DEMEncoder, OpticalEncoder, SAREncoder,
            TemporalEncoderStream, CrossModalAttention, DEMSpatialConditioner,
            LateFusion, Decoder, PhenologyAuxHead
        )
        from models.fusion_net_v5 import FusionCropNetV5
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dem_enc = DEMEncoder(5, 128).to(device)
        dem = torch.randn(2, 5, 64, 64).to(device)
        dem_out = dem_enc(dem)
        log(f"DEMEncoder: 输入 {dem.shape} -> 输出 {dem_out.shape}")
        
        opt_enc = OpticalEncoder(10, 512, "resnet18", False).to(device)
        opt = torch.randn(4, 10, 64, 64).to(device)
        opt_out, p2, p3 = opt_enc(opt)
        log(f"OpticalEncoder: 输入 {opt.shape} -> 输出 {opt_out.shape}")
        
        sar_enc = SAREncoder(5, 32, 512, 128).to(device)
        sar = torch.randn(4, 5, 64, 64).to(device)
        dem_tiled = torch.randn(4, 128, 64, 64).to(device)
        s1, s2, s3 = sar_enc(sar, dem_tiled)
        log(f"SAREncoder: 输入 {sar.shape} -> 输出 s1:{s1.shape}, s2:{s2.shape}, s3:{s3.shape}")
        
        temporal = TemporalEncoderStream(512, 8, 2).to(device)
        seq = torch.randn(128, 12, 512).to(device)
        doy = torch.rand(128, 12).to(device)
        out, seq_out = temporal(seq, doy)
        log(f"TemporalEncoderStream: 输入 {seq.shape} -> 输出 {out.shape}")
        
        cross_modal = CrossModalAttention(512, 8, 4).to(device)
        opt_g = torch.randn(2, 512, 16, 16).to(device)
        sar_g = torch.randn(2, 512, 16, 16).to(device)
        xm_out = cross_modal(opt_g, sar_g)
        log(f"CrossModalAttention: 输入 {opt_g.shape} + {sar_g.shape} -> 输出 {xm_out.shape}")
        
        dem_cond = DEMSpatialConditioner(512, 128).to(device)
        fused = torch.randn(2, 512, 16, 16).to(device)
        dem_feat = torch.randn(2, 128, 16, 16).to(device)
        cond_out = dem_cond(fused, dem_feat)
        log(f"DEMSpatialConditioner: 输入 {fused.shape} + {dem_feat.shape} -> 输出 {cond_out.shape}")
        
        late_fuse = LateFusion(512).to(device)
        xm_flat = torch.randn(512, 512).to(device)
        opt_flat = torch.randn(512, 512).to(device)
        sar_flat = torch.randn(512, 512).to(device)
        late_out = late_fuse(xm_flat, opt_flat, sar_flat)
        log(f"LateFusion: 输入 {xm_flat.shape} -> 输出 {late_out.shape}")
        
        decoder = Decoder(512, [64, 128], n_heads=8, win=4).to(device)
        final = torch.randn(2, 512, 16, 16).to(device)
        logits = decoder(final, (p2[:2],), (s1[:2], s2[:2]), (64, 64))
        log(f"Decoder: 输入 {final.shape} -> 输出 {logits.shape}")
        
        pheno = PhenologyAuxHead(512).to(device)
        opt_feat = torch.randn(4, 512, 16, 16).to(device)
        ndvi = pheno(opt_feat)
        log(f"PhenologyAuxHead: 输入 {opt_feat.shape} -> 输出 {ndvi.shape}")
        
        log("所有组件测试通过")
        return True, None
    except Exception as e:
        log(f"组件测试失败: {str(e)}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return False, str(e)

def test_gradient_flow(use_edl=False):
    """测试梯度流动"""
    log("=== 测试 4: 梯度流动 ===")
    try:
        if use_edl:
            from models.fusion_net_v5_edl import FusionCropNetV5EDL
        else:
            from models.fusion_net_v5 import FusionCropNetV5
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_edl:
            model = FusionCropNetV5EDL(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet50", pretrained=False,
                n_heads=16, win_size=4, n_layers=4,
                edl_dropout_p=0.0
            ).to(device)
        else:
            model = FusionCropNetV5(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet50", pretrained=False,
                n_heads=16, win_size=4, n_layers=4,
                drop_timestep_p=0.0
            ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        opt_seq = torch.randn(1, 6, 10, 32, 32).to(device) * 0.1
        sar_seq = torch.randn(1, 6, 5, 32, 32).to(device) * 0.1
        dem = torch.randn(1, 5, 32, 32).to(device) * 0.1
        doy = torch.rand(1, 6).to(device)
        target = torch.randint(0, 7, (1, 32, 32)).to(device)
        
        optimizer.zero_grad()
        
        if use_edl:
            alpha, _, _ = model(opt_seq, sar_seq, dem, doy, epoch=1)
            from models.fusion_net_v5_edl import dirichlet_to_predictions
            preds = dirichlet_to_predictions(alpha)
            logits = preds['probs']
        else:
            logits, _, _ = model(opt_seq, sar_seq, dem, doy)
        
        loss = torch.nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
        
        grad_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        avg_grad_norm = np.mean(grad_norms)
        log(f"梯度流动测试通过")
        log(f"  损失值: {loss.item():.6f}")
        log(f"  平均梯度范数: {avg_grad_norm:.6f}")
        log(f"  有梯度的参数数量: {len(grad_norms)}")
        
        return True, None
    except Exception as e:
        log(f"梯度流动测试失败: {str(e)}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return False, str(e)

def test_calibration_validation(use_edl=True):
    """测试 EDL 校准验证"""
    log("=== 测试 5: EDL 校准验证 ===")
    if not use_edl:
        log("非 EDL 模式，跳过校准测试", "WARNING")
        return True, None
    try:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
        from utils.calibration import (calibration_report, expected_calibration_error,
                                        negative_log_likelihood_dirichlet,
                                        brier_score_dirichlet,
                                        uncertainty_error_correlation,
                                        uncertainty_auroc)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            edl_dropout_p=0.3
        ).to(device).eval()

        B, T, H, W = 1, 8, 32, 32
        opt_seq = torch.randn(B, T, 10, H, W).to(device) * 0.1
        sar_seq = torch.randn(B, T, 5, H, W).to(device) * 0.1
        dem = torch.randn(B, 5, H, W).to(device) * 0.1
        doy = torch.rand(B, T).to(device)
        targets = torch.randint(0, 7, (H, W))

        with torch.no_grad():
            alpha = model(opt_seq, sar_seq, dem, doy)

        alpha_np = alpha.cpu().numpy()
        cal = calibration_report(alpha_np, targets.numpy(), 7, n_bins=10)
        log(f"  ECE: {cal['ECE']:.4f}")
        log(f"  Adaptive ECE: {cal['AdaptiveECE']:.4f}")
        log(f"  NLL: {cal['NLL']:.4f}")
        log(f"  Brier: {cal['Brier']:.4f}")
        log(f"  Sharpness: {cal['Sharpness']:.4f}")
        log(f"  AUROC (err det): {cal['AUROC_error_detection']:.4f}")
        log("校准验证通过")
        return True, None
    except Exception as e:
        log(f"校准验证失败: {str(e)}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return False, str(e)


def test_interpretability(use_edl=True):
    """测试模型可解释性"""
    log("=== 测试 6: 模型可解释性 ===")
    if not use_edl:
        log("非 EDL 模式，跳过可解释性测试", "WARNING")
        return True, None
    try:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        from utils.interpretability import (
            modality_ablation, temporal_importance,
            spectral_band_importance, pixel_explanation_report,
            confusion_region_analysis, GradCAM_EDL
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device).eval()

        B, T, H, W = 1, 8, 32, 32
        opt_seq = torch.randn(B, T, 10, H, W).to(device) * 0.1
        sar_seq = torch.randn(B, T, 5, H, W).to(device) * 0.1
        dem = torch.randn(B, 5, H, W).to(device) * 0.1
        doy = torch.rand(B, T).to(device)
        targets = torch.randint(0, 7, (H, W))

        # Modality ablation
        abl = modality_ablation(model, opt_seq, sar_seq, dem, doy, device=device)
        log(f"  模态贡献: opt={abl['relative_importance']['optical']:.3f} "
            f"sar={abl['relative_importance']['sar']:.3f} "
            f"dem={abl['relative_importance']['dem']:.3f}")

        # Temporal importance
        t_imp, _ = temporal_importance(model, opt_seq, sar_seq, dem, doy, device=device)
        log(f"  时序重要性: top3={np.argsort(t_imp)[-3:][::-1]}")

        # Band importance
        band_imp = spectral_band_importance(model, opt_seq, sar_seq, dem, doy, device=device)
        log(f"  光学波段重要性: {np.argmax(band_imp['optical_bands'])}")
        log(f"  SAR波段重要性: {np.argmax(band_imp['sar_bands'])}")

        # Pixel explanation
        with torch.no_grad():
            alpha = model(opt_seq, sar_seq, dem, doy)
        px = pixel_explanation_report(alpha.cpu().numpy(), targets.numpy(), num_classes=7)
        log(f"  正确像素数: {px['correct']['n']}, 错误像素数: {px['incorrect']['n']}")
        log(f"  正确vacuity: {px['correct']['vacuity']['mean']:.4f}, "
            f"错误vacuity: {px['incorrect']['vacuity']['mean']:.4f}")

        # Confusion regions
        confusion = confusion_region_analysis(alpha.cpu().numpy(), targets.numpy(), num_classes=7)
        confusion_count = sum(1 for v in confusion.values() if v['n'] > 0)
        log(f"  混淆类别对数: {confusion_count}")

        # Grad-CAM
        gradcam = GradCAM_EDL(model, target_layer_name="decoder")
        hm = gradcam(opt_seq, sar_seq, dem, doy, class_idx=3)
        log(f"  Grad-CAM: shape={hm.shape}, range=[{hm.min():.3f}, {hm.max():.3f}]")

        log("可解释性分析通过")
        return True, None
    except Exception as e:
        log(f"可解释性分析失败: {str(e)}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return False, str(e)


def test_api_integration():
    """测试 API 集成"""
    log("=== 测试 7: API 集成 ===")
    try:
        from api.main import app, load_model, logger, monitor, tracker
        
        log("API 应用初始化成功")
        log(f"应用标题: {app.title}")
        log(f"应用版本: {app.version}")
        
        import io
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.get("/health")
        log(f"健康检查接口: {response.status_code}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        response = client.get("/model/info")
        log(f"模型信息接口: {response.status_code}")
        assert response.status_code == 200
        
        response = client.get("/classes")
        log(f"类别列表接口: {response.status_code}")
        assert response.status_code == 200
        
        log("API 集成测试通过")
        return True, None
    except ImportError:
        log("FastAPI 未安装，跳过 API 测试", "WARNING")
        return True, None
    except Exception as e:
        log(f"API 集成测试失败: {str(e)}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return False, str(e)

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--edl", action="store_true", help="测试 EDL 模型")
    args = parser.parse_args()
    
    use_edl = args.edl
    
    log("=" * 70)
    log("FusionCropNetV5 完整系统测试" + (" (EDL模式)" if use_edl else ""))
    log("=" * 70)
    log(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"PyTorch 版本: {torch.__version__}")
    log(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"CUDA 版本: {torch.version.cuda}")
        log(f"GPU 数量: {torch.cuda.device_count()}")
        log(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    log(f"EDL 模式: {'[ON] 启用' if use_edl else '[OFF] 禁用'}")
    
    tests = [
        ("模型初始化", lambda: test_model_initialization(use_edl)),
        ("前向传播", lambda: test_forward_pass(use_edl)),
        ("组件测试", test_components),
        ("梯度流动", lambda: test_gradient_flow(use_edl)),
        ("EDL校准验证", lambda: test_calibration_validation(use_edl)),
        ("模型可解释性", lambda: test_interpretability(use_edl)),
        ("API 集成", test_api_integration)
    ]
    
    results = []
    for name, test_func in tests:
        success, error = test_func()
        results.append((name, success, error))
        log("")
    
    log("=" * 70)
    log("测试结果汇总")
    log("=" * 70)
    
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    
    for name, success, error in results:
        status = "PASS" if success else "FAIL"
        log(f"{status}: {name}")
        if not success and error:
            log(f"  错误: {error}", "ERROR")
    
    log("")
    log(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        log("[PASS] 所有测试通过！", "INFO")
        return 0
    else:
        log("[FAIL] 部分测试失败，请检查日志", "ERROR")
        return 1

if __name__ == "__main__":
    sys.exit(main())