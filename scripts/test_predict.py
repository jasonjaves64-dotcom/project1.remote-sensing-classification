# -*- coding: utf-8 -*-
import sys
import os
import torch
import numpy as np
import argparse

print("Testing prediction...", file=sys.stderr)

def build_time_sequence(data_dir, year):
    import glob
    tif_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    
    if len(tif_files) == 0:
        print(f"Warning: No TIFF files found in {data_dir}", file=sys.stderr)
        seq_len = 12
        return np.random.randn(seq_len, 10, 64, 64).astype(np.float32), np.linspace(0, 1, seq_len)
    
    doy_values = []
    images = []
    
    for f in tif_files[:12]:
        try:
            import rasterio
            with rasterio.open(f) as src:
                img = src.read()
                images.append(img[:10])
                
                doy = int(os.path.basename(f).split("_")[1].split(".")[0])
                doy_values.append(doy / 365.0)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}", file=sys.stderr)
    
    if images:
        return np.array(images), np.array(doy_values)
    
    seq_len = 12
    return np.random.randn(seq_len, 10, 64, 64).astype(np.float32), np.linspace(0, 1, seq_len)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edl", action="store_true", help="使用EDL模型")
    parser.add_argument("--n_passes", type=int, default=3, help="EDL推理次数")
    parser.add_argument("--calibration", action="store_true", help="输出校准验证报告")
    args = parser.parse_args()
    
    use_edl = args.edl
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=sys.stderr)
    
    if use_edl:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50
        ).to(device)
        
        model_path = "checkpoints/best_phase2_edl.pth" if os.path.exists("checkpoints/best_phase2_edl.pth") else "best_phase2_edl.pth"
    else:
        from models.fusion_net_v5 import FusionCropNetV5
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        model_path = "checkpoints/best_phase2.pth" if os.path.exists("checkpoints/best_phase2.pth") else "best_phase2.pth"
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
            print(f"Model loaded successfully from {model_path}", file=sys.stderr)
        else:
            print(f"Warning: No trained model found at {model_path}, using random weights", file=sys.stderr)
        
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
    
    model.eval()
    
    sequence, doy_norm = build_time_sequence("./data/landsat_images/2023", 2023)
    print(f"Sequence shape: {sequence.shape}", file=sys.stderr)
    
    opt_seq = sequence[:, :, :32, :32]
    doy = doy_norm.copy()
    
    opt_t = torch.from_numpy(opt_seq).float().unsqueeze(0).to(device)
    sar_t = torch.randn(1, len(doy), 5, 32, 32).to(device)
    dem_t = torch.randn(1, 5, 32, 32).to(device)
    doy_t = torch.from_numpy(doy).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_edl:
            alpha = model(opt_t, sar_t, dem_t, doy_t)
            preds = dirichlet_to_predictions(alpha)
            pred = preds['pred_class'].squeeze(0).cpu().numpy()
            
            print(f"Prediction shape: {pred.shape}", file=sys.stderr)
            print(f"Unique predictions: {np.unique(pred)}", file=sys.stderr)
            print(f"Mean vacuity: {preds['vacuity'].mean().item():.4f}", file=sys.stderr)
            print(f"Mean dissonance: {preds['dissonance'].mean().item():.4f}", file=sys.stderr)
            
            result = model.predict_uncertainty(opt_t, sar_t, dem_t, doy_t, 
                                               n_passes=args.n_passes, use_tta=True)
            print(f"Uncertainty inference completed", file=sys.stderr)
            print(f"Final vacuity: {result['vacuity'].mean().item():.4f}", file=sys.stderr)
            print(f"Final dissonance: {result['dissonance'].mean().item():.4f}", file=sys.stderr)
        else:
            logits = model(opt_t, sar_t, dem_t, doy_t)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
            print(f"Prediction shape: {pred.shape}", file=sys.stderr)
            print(f"Unique predictions: {np.unique(pred)}", file=sys.stderr)
    
    print(f"Prediction test completed! (EDL: {'Yes' if use_edl else 'No'})", file=sys.stderr)

    # Calibration validation
    if args.calibration and use_edl:
        from utils.calibration import calibration_report, print_calibration_report
        targets = torch.randint(0, 7, (32, 32))
        cal = calibration_report(alpha.cpu().numpy(), targets.numpy(), num_classes=7, n_bins=10)
        print("\n" + "=" * 60, file=sys.stderr)
        print("EDL Calibration Report", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"ECE: {cal['ECE']:.4f}", file=sys.stderr)
        print(f"NLL: {cal['NLL']:.4f}", file=sys.stderr)
        print(f"Brier: {cal['Brier']:.4f}", file=sys.stderr)
        print(f"AUROC(err det): {cal['AUROC_error_detection']:.4f}", file=sys.stderr)

if __name__ == "__main__":
    main()