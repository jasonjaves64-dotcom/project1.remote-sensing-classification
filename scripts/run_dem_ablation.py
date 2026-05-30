#!/usr/bin/env python
# =============================================================================
# scripts/run_dem_ablation.py
# DEM Ablation Experiment Runner
#
# Quantifies the contribution of each DEM injection point by running
# systematic ablation experiments on a trained FusionCropNet model.
#
# Usage:
#   python scripts/run_dem_ablation.py --model checkpoints/best_model.pth
#   python scripts/run_dem_ablation.py --model checkpoints/best_model.pth --data data/processed
#   python scripts/run_dem_ablation.py --synthetic  # quick test with random data
# =============================================================================
import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dem_ablation import (
    DEMAblationRunner, INJECTION_POINTS, ABLATION_GROUPS, print_ablation_summary
)


def parse_args():
    p = argparse.ArgumentParser(
        description="DEM Ablation Experiment Runner — FusionCropNet")
    p.add_argument("--model", type=str, default="",
                   help="Path to trained model checkpoint (.pth)")
    p.add_argument("--data", type=str, default="",
                   help="Path to processed data directory")
    p.add_argument("--output", type=str, default="./ablation_output",
                   help="Output directory for reports")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: cuda / cpu")
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--model-version", type=str, default="v5edl",
                   choices=["v5edl", "v6", "v5pro"],
                   help="Model version to load")
    p.add_argument("--mode", type=str, default="full",
                   choices=["full", "individual", "grouped"],
                   help="Ablation mode")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic random data (for testing)")
    p.add_argument("--synthetic-size", type=int, default=64,
                   help="Spatial size for synthetic data")
    p.add_argument("--synthetic-temporal", type=int, default=12,
                   help="Temporal steps for synthetic data")
    return p.parse_args()


def load_model(args):
    """Load a trained FusionCropNet model from checkpoint."""
    from models.fusion_net_v5_edl import FusionCropNetV5EDL

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_v6 = args.model_version == "v6"

    if args.model_version == "v5pro":
        from models.fusion_net_v5pro import FusionCropNetV5Pro
        model = FusionCropNetV5Pro(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=args.num_classes,
            feat_dim=512, backbone="resnet50", pretrained=False,
            use_v6_enhancements=use_v6,
        ).to(device)
    else:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=args.num_classes,
            feat_dim=512, backbone="resnet50", pretrained=False,
            use_v6_enhancements=use_v6,
        ).to(device)

    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded model from: {args.model}")
    else:
        print("No checkpoint provided — using random weights (synthetic mode)")

    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model.__class__.__name__}  Params: {params:.1f}M  "
          f"V6: {use_v6}  Device: {device}")
    return model, device


def load_data(args, device):
    """Load real or synthetic data."""
    if args.synthetic or not args.data:
        print(f"Using synthetic data ({args.synthetic_size}×{args.synthetic_size}, "
              f"T={args.synthetic_temporal})")
        H = W = args.synthetic_size
        T = args.synthetic_temporal
        opt_seq = torch.randn(1, T, 10, H, W, device=device)
        sar_seq = torch.randn(1, T, 5, H, W, device=device)
        dem = torch.randn(1, 5, H, W, device=device)
        doy = torch.linspace(0, 1, T, device=device).unsqueeze(0)
        labels = torch.randint(1, args.num_classes, (1, H, W))
        return opt_seq, sar_seq, dem, doy, labels, None, None

    data_dir = args.data
    print(f"Loading data from: {data_dir}")

    opt_seq = torch.from_numpy(
        np.load(os.path.join(data_dir, "opt_sequence.npy"))).float().unsqueeze(0).to(device)
    sar_seq = torch.from_numpy(
        np.load(os.path.join(data_dir, "sar_sequence.npy"))).float().unsqueeze(0).to(device)
    doy = torch.from_numpy(
        np.load(os.path.join(data_dir, "doy_norm.npy"))).float().unsqueeze(0).to(device)

    dem_path = os.path.join(data_dir, "dem.npy")
    if os.path.exists(dem_path):
        dem = torch.from_numpy(np.load(dem_path)).float().unsqueeze(0).to(device)
    else:
        H, W = opt_seq.shape[3], opt_seq.shape[4]
        dem = torch.zeros(1, 5, H, W, device=device)

    label_path = os.path.join(data_dir, "label.npy")
    if os.path.exists(label_path):
        labels = np.load(label_path)
    else:
        labels = np.zeros((opt_seq.shape[3], opt_seq.shape[4]), dtype=np.int64)

    # Center crop to avoid edge effects
    H, W = opt_seq.shape[3], opt_seq.shape[4]
    ps = min(H, W, 128)
    r, c = (H - ps) // 2, (W - ps) // 2
    opt_seq = opt_seq[:, :, :, r:r + ps, c:c + ps]
    sar_seq = sar_seq[:, :, :, r:r + ps, c:c + ps]
    dem = dem[:, :, r:r + ps, c:c + ps]
    labels = labels[r:r + ps, c:c + ps]

    print(f"  Data shape: opt={list(opt_seq.shape)}, sar={list(sar_seq.shape)}, "
          f"dem={list(dem.shape)}, labels={labels.shape}")

    return opt_seq, sar_seq, dem, doy, labels, None, None


def main():
    args = parse_args()

    # Validate: V6 needed for early_fusion, opt_cond, temporal_bias
    if args.model_version == "v5edl":
        print("Note: V5EDL only has 3 DEM injection points (sar_film, spatial_cond, decoder_skip).")
        print("      V6-only points (early_fusion, opt_cond, temporal_bias) will have no effect.")

    model, device = load_model(args)
    opt_seq, sar_seq, dem, doy, labels, cloud_mask, valid_count = load_data(args, device)

    runner = DEMAblationRunner(model, device=device, num_classes=args.num_classes)

    if args.mode in ("full", "individual"):
        print("\n" + "=" * 60)
        print("  Individual Injection Point Ablation")
        print("=" * 60)
        indiv = runner.run_individual(opt_seq, sar_seq, dem, doy, labels,
                                       cloud_mask, valid_count)

    if args.mode in ("full", "grouped"):
        print("\n" + "=" * 60)
        print("  Grouped Ablation")
        print("=" * 60)
        grouped = runner.run_grouped(opt_seq, sar_seq, dem, doy, labels,
                                      cloud_mask, valid_count)

    if args.mode == "full":
        results = {"individual": indiv, "grouped": grouped,
                   "injection_points": INJECTION_POINTS}
    elif args.mode == "individual":
        results = {"individual": indiv, "injection_points": INJECTION_POINTS}
    else:
        results = {"grouped": grouped, "injection_points": INJECTION_POINTS}

    print_ablation_summary(results)
    runner.generate_report(results, output_dir=args.output)

    print(f"\nDone. Reports saved to: {args.output}/")


if __name__ == "__main__":
    main()
