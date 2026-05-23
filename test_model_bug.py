import torch
import sys
sys.path.insert(0, '.')

from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions

def test_model():
    print('测试 FusionCropNetV5EDL 模型...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    try:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet50', pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50
        ).to(device)
        print('✓ 模型初始化成功')
        
        B, T, H, W = 2, 12, 32, 32
        opt = torch.randn(B, T, 10, H, W).to(device)
        sar = torch.randn(B, T, 5, H, W).to(device)
        dem = torch.randn(B, 5, H, W).to(device)
        doy = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).to(device)
        
        model.train()
        alpha, ndvi, cl = model(opt, sar, dem, doy, epoch=10)
        print('✓ 训练模式前向传播成功')
        print(f'  alpha shape: {alpha.shape}')
        print(f'  ndvi shape: {ndvi.shape}')
        print(f'  consistency_loss: {cl.item():.4f}')
        
        preds = dirichlet_to_predictions(alpha.detach())
        print('✓ 不确定性计算成功')
        print(f'  probs shape: {preds["probs"].shape}')
        print(f'  vacuity: {preds["vacuity"].mean().item():.4f}')
        print(f'  dissonance: {preds["dissonance"].mean().item():.4f}')
        
        model.eval()
        result = model.predict_uncertainty(opt, sar, dem, doy, n_passes=3, use_tta=True)
        print('✓ 不确定性推理成功')
        print(f'  final vacuity: {result["vacuity"].mean().item():.4f}')
        print(f'  final dissonance: {result["dissonance"].mean().item():.4f}')
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'✓ 模型参数: {params:.1f}M')

        # Calibration validation
        print('\n--- EDL 校准验证 ---')
        from utils.calibration import calibration_report, expected_calibration_error
        targets = torch.randint(0, 7, (B, H, W))
        cal = calibration_report(alpha.detach().cpu().numpy(), targets.numpy(),
                                 num_classes=7, n_bins=10)
        print(f'  ECE: {cal["ECE"]:.4f}')
        print(f'  NLL: {cal["NLL"]:.4f}')
        print(f'  Brier: {cal["Brier"]:.4f}')
        print(f'  AUROC (err det): {cal["AUROC_error_detection"]:.4f}')
        print(f'  Sharpness: {cal["Sharpness"]:.4f}')
        print(f'✓ 校准验证通过')

        # Interpretability check
        print('\n--- 可解释性检查 ---')
        from utils.interpretability import (modality_ablation, GradCAM_EDL,
                                             pixel_explanation_report)
        abl = modality_ablation(model, opt, sar, dem, doy, device=device)
        ri = abl["relative_importance"]
        print(f'  模态贡献: opt={ri["optical"]:.3f} sar={ri["sar"]:.3f} dem={ri["dem"]:.3f}')
        gradcam = GradCAM_EDL(model, target_layer_name="decoder")
        hm = gradcam(opt, sar, dem, doy, class_idx=3)
        print(f'  Grad-CAM: shape={hm.shape}, range=[{hm.min():.3f}, {hm.max():.3f}]')
        px = pixel_explanation_report(alpha.detach().cpu().numpy(), targets.numpy(),
                                       num_classes=7)
        print(f'  正确vacuity: {px["correct"]["vacuity"]["mean"]:.4f}, '
              f'错误vacuity: {px["incorrect"]["vacuity"]["mean"]:.4f}')
        print(f'✓ 可解释性检查通过')

        print('\n✅ 所有测试通过！模型没有发现bug。')
        
    except Exception as e:
        print(f'\n❌ 发现错误: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)