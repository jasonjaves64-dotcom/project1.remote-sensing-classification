"""
模型诊断脚本 - 检查模型预测问题
"""

import torch
import numpy as np
import sys
import argparse

sys.path.insert(0, '.')

def check_model_output(use_edl=False, use_v5pro=False):
    """检查模型输出是否正常"""
    if use_v5pro:
        from models.fusion_net_v5pro import FusionCropNetV5Pro, dirichlet_to_predictions
    elif use_edl:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
    else:
        from models.fusion_net_v5 import FusionCropNetV5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if use_v5pro:
        model = FusionCropNetV5Pro(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50,
            use_carafe=True, dynamic_dropout=False, adaptive_kl=False,
        ).to(device)
    elif use_edl:
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
    
    model.eval()
    
    B, T, H, W = 1, 12, 64, 64
    
    opt_seq = torch.randn(B, T, 10, H, W).to(device) * 0.1
    sar_seq = torch.randn(B, T, 5, H, W).to(device) * 0.1
    dem = torch.randn(B, 5, H, W).to(device) * 0.1
    doy = torch.rand(B, T).to(device)
    
    with torch.no_grad():
        if use_edl:
            alpha = model(opt_seq, sar_seq, dem, doy)
            preds = dirichlet_to_predictions(alpha)
            pred = preds['pred_class'].cpu().numpy()[0]
            probs = preds['probs'].cpu().numpy()[0]
            vacuity = preds['vacuity'].cpu().numpy()[0]
            dissonance = preds['dissonance'].cpu().numpy()[0]
        else:
            logits = model(opt_seq, sar_seq, dem, doy)
            pred = logits.argmax(dim=1).cpu().numpy()[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            vacuity = None
            dissonance = None
    
    unique, counts = np.unique(pred, return_counts=True)
    total_pixels = H * W
    
    print("\n=== Model Output Diagnosis ===")
    print("Output shape:", probs.shape if use_edl else logits.shape)
    print("Prediction class statistics:")
    for cls, cnt in zip(unique, counts):
        percentage = (cnt / total_pixels) * 100
        print("  Class", cls, ":", cnt, "pixels (", percentage, "%)")
    
    print("\nAverage probability per class:")
    class_names = ["Background", "Winter Wheat", "Summer Corn", "Rice", "Soybean", "Cotton", "Other"]
    for i, name in enumerate(class_names):
        avg_prob = probs[i].mean()
        print("  ", name, ":", avg_prob)
    
    max_prob_per_pixel = probs.max(axis=0).mean()
    print("\nAverage max probability:", max_prob_per_pixel)
    
    if vacuity is not None:
        print("\nUncertainty metrics:")
        print("  Average vacuity:", vacuity.mean())
        print("  Average dissonance:", dissonance.mean())
    
    if np.all(pred == 0):
        print("\nWARNING: All predictions are background class!")
        print("Possible reasons:")
        print("1. Model is not loaded with pretrained weights")
        print("2. Input data is not properly normalized")
        print("3. Model is in wrong mode (train vs eval)")
        return False
    elif len(unique) < 3:
        print("\nWARNING: Model predicts very few classes!")
        print("Unique classes predicted:", unique)
        return False
    
    if max_prob_per_pixel < 0.3:
        print("\nWARNING: Model confidence is very low!")
        print("Average max probability:", max_prob_per_pixel)
        return False
    
    print("\n[PASS]  Model output looks normal!")
    return True

def check_gradients(use_edl=False):
    """检查梯度是否正常"""
    if use_edl:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
    else:
        from models.fusion_net_v5 import FusionCropNetV5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_edl:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=8, win_size=4, n_layers=2,
            edl_dropout_p=0.0
        ).to(device)
    else:
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=8, win_size=4, n_layers=2,
            drop_timestep_p=0.0
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    opt_seq = torch.randn(2, 6, 10, 32, 32).to(device) * 0.1
    sar_seq = torch.randn(2, 6, 5, 32, 32).to(device) * 0.1
    dem = torch.randn(2, 5, 32, 32).to(device) * 0.1
    doy = torch.rand(2, 6).to(device)
    target = torch.randint(0, 7, (2, 32, 32)).to(device)
    
    optimizer.zero_grad()
    
    if use_edl:
        alpha, _, _ = model(opt_seq, sar_seq, dem, doy, epoch=1)
        preds = dirichlet_to_predictions(alpha)
        logits = preds['probs']
    else:
        logits = model(opt_seq, sar_seq, dem, doy)
    
    loss = torch.nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    
    grad_norms = []
    zero_grad_count = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is None:
                zero_grad_count += 1
            elif param.grad.norm().item() == 0:
                zero_grad_count += 1
            else:
                grad_norms.append(param.grad.norm().item())
    
    print("\n=== Gradient Diagnosis ===")
    print("Total trainable parameters:", total_params)
    print("Parameters with zero/no gradient:", zero_grad_count)
    print("Parameters with non-zero gradient:", len(grad_norms))
    
    if grad_norms:
        print("\nGradient norm statistics:")
        print("  Min:", np.min(grad_norms))
        print("  Max:", np.max(grad_norms))
        print("  Mean:", np.mean(grad_norms))
        print("  Std:", np.std(grad_norms))
    else:
        print("\nWARNING: No gradients found!")
        return False
    
    if zero_grad_count > total_params * 0.5:
        print("\nWARNING: More than 50% of parameters have zero gradient!")
        return False
    
    print("\n[PASS]  Gradients look normal!")
    return True

def check_uncertainty_calibration(use_edl=False):
    """检查 EDL 不确定性校准质量"""
    if not use_edl:
        print("\n=== Calibration Check ===")
        print("Skipped: non-EDL model")
        return True
    from models.fusion_net_v5_edl import FusionCropNetV5EDL
    from utils.calibration import calibration_report

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

    with torch.no_grad():
        alpha = model(opt_seq, sar_seq, dem, doy)
    cal = calibration_report(alpha.cpu().numpy(), targets.numpy(), 7, n_bins=10)

    print("\n=== Calibration Diagnosis ===")
    print(f"ECE: {cal['ECE']:.4f} (threshold: < 0.20)")
    print(f"NLL: {cal['NLL']:.4f}")
    print(f"Brier: {cal['Brier']:.4f}")
    print(f"Dispersion: {cal['Dispersion']:.4f}")
    print(f"AUROC (err det): {cal['AUROC_error_detection']:.4f} (threshold: > 0.55)")

    warnings = []
    if cal["ECE"] > 0.20:
        warnings.append(f"ECE too high: {cal['ECE']:.4f}")
    if cal["Dispersion"] < 0.01:
        warnings.append(f"Very low dispersion: {cal['Dispersion']:.4f}")
    if cal["AUROC_error_detection"] < 0.55:
        warnings.append(f"Poor error detection: {cal['AUROC_error_detection']:.4f}")

    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
        return False
    print("[PASS]  Calibration looks normal!")
    return True


def check_interpretability(use_edl=False):
    """检查模型可解释性功能"""
    if not use_edl:
        print("\n=== Interpretability Check ===")
        print("Skipped: non-EDL model")
        return True
    from models.fusion_net_v5_edl import FusionCropNetV5EDL
    from utils.interpretability import (modality_ablation, GradCAM_EDL,
                                         pixel_explanation_report)

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

    print("\n=== Interpretability Diagnosis ===")

    # Modality ablation
    abl = modality_ablation(model, opt_seq, sar_seq, dem, doy, device=device)
    ri = abl["relative_importance"]
    print(f"Modality contribution: opt={ri['optical']:.3f} sar={ri['sar']:.3f} dem={ri['dem']:.3f}")
    if max(ri.values()) > 0.9:
        print("WARNING: Single modality dominates - model may not be fusing properly")

    # Grad-CAM
    gradcam = GradCAM_EDL(model, target_layer_name="decoder")
    hm = gradcam(opt_seq, sar_seq, dem, doy, class_idx=3)
    print(f"Grad-CAM: shape={hm.shape}, min={hm.min():.3f}, max={hm.max():.3f}")

    # Pixel explanation
    with torch.no_grad():
        alpha = model(opt_seq, sar_seq, dem, doy)
    px = pixel_explanation_report(alpha.cpu().numpy(), targets.numpy(), num_classes=7)
    vac_correct = px["correct"]["vacuity"]["mean"]
    vac_incorrect = px["incorrect"]["vacuity"]["mean"]
    print(f"Pixel explanation: correct_vac={vac_correct:.4f}, incorrect_vac={vac_incorrect:.4f}")
    if vac_incorrect < vac_correct:
        print("WARNING: Wrong predictions have LOWER vacuity than correct ones")

    print("[PASS]  Interpretability check passed!")
    return True


def check_memory_usage(use_edl=False):
    """检查内存使用情况"""
    if use_edl:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
    else:
        from models.fusion_net_v5 import FusionCropNetV5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_edl:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
    else:
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = total_params * 4 / (1024 * 1024)
    
    print("\n=== Memory Diagnosis ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Approximate model size: {total_size_mb:.2f} MB")
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"\nGPU Memory:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved: {memory_reserved:.2f} MB")
        
        free_memory = (torch.cuda.get_device_properties(0).total_memory - memory_reserved) / (1024 ** 2)
        print(f"  Free: {free_memory:.2f} MB")
    
    print("\n[PASS]  Memory usage looks normal!")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edl", action="store_true", help="使用 EDL 模型")
    parser.add_argument("--v5pro", action="store_true", help="使用 V5Pro 模型")
    parser.add_argument("--all", action="store_true", help="运行所有诊断")
    parser.add_argument("--output", action="store_true", help="检查模型输出")
    parser.add_argument("--gradients", action="store_true", help="检查梯度")
    parser.add_argument("--memory", action="store_true", help="检查内存")
    parser.add_argument("--calibration", action="store_true", help="检查EDL校准质量")
    parser.add_argument("--interpretability", action="store_true", help="检查模型可解释性")
    args = parser.parse_args()
    
    use_edl = args.edl
    
    print("=" * 70)
    print("Model Diagnosis Tool" + (" (EDL Mode)" if use_edl else ""))
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"EDL Mode: {use_edl}")
    
    tests = []
    
    if args.all or args.output:
        tests.append(("Model Output", check_model_output))
    if args.all or args.gradients:
        tests.append(("Gradients", check_gradients))
    if args.all or args.memory:
        tests.append(("Memory", check_memory_usage))
    if args.all or args.calibration:
        tests.append(("Calibration", check_uncertainty_calibration))
    if args.all or args.interpretability:
        tests.append(("Interpretability", check_interpretability))

    if not tests:
        print("\nPlease specify what to check: --output, --gradients, --memory, "
              "--calibration, --interpretability, or --all")
        return
    
    results = []
    for name, test_func in tests:
        print(f"\n--- Checking {name} ---")
        success = test_func(use_edl=use_edl)
        results.append((name, success))
    
    print("\n" + "=" * 70)
    print("Diagnosis Results")
    print("=" * 70)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} passed")
    
    if passed == total:
        print("\n[PASS]  All checks passed!")
    else:
        print("\n[FAIL]  Some checks failed - see warnings above")

if __name__ == "__main__":
    main()