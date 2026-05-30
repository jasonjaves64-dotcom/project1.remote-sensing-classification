#!/usr/bin/env python
# =============================================================================
# scripts/run_cross_year_analysis.py
# Cross-Year Per-Class Degradation Analysis Runner
#
# Runs multi-year inference and quantifies per-class IoU degradation
# to identify which crop types are most/least stable across growing seasons.
#
# Usage:
#   python scripts/run_cross_year_analysis.py --data-dir ./data/processed
#   python scripts/run_cross_year_analysis.py --model checkpoints/best.pth --years 2022 2023 2024
# =============================================================================
import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-Year Per-Class Degradation Analysis")
    p.add_argument("--model", type=str, default="",
                   help="Path to trained model checkpoint (.pth)")
    p.add_argument("--data-dir", type=str, default="./data/processed",
                   help="Base directory with per-year subdirectories")
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024],
                   help="Years to analyze (default: 2022 2023 2024)")
    p.add_argument("--reference-year", type=int, default=None,
                   help="Reference year for degradation calculation (default: first year)")
    p.add_argument("--output", type=str, default="./validation",
                   help="Output directory for reports and plots")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data for testing")
    p.add_argument("--synthetic-size", type=int, default=64)
    p.add_argument("--synthetic-temporal", type=int, default=12)
    p.add_argument("--model-version", type=str, default="v5edl",
                   choices=["v5edl", "v6", "v5pro"])
    return p.parse_args()


def load_model(args):
    """Load model from checkpoint or create with random weights."""
    import torch
    from models.fusion_net_v5_edl import FusionCropNetV5EDL

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_v6 = args.model_version == "v6"

    model = FusionCropNetV5EDL(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
        feat_dim=512, backbone="resnet50", pretrained=False,
        use_v6_enhancements=use_v6,
    ).to(device)

    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded model from: {args.model}")

    model.eval()
    return model, device


def load_synthetic_data(args):
    """Generate per-year synthetic data with controlled degradation patterns."""
    H = W = args.synthetic_size
    T = args.synthetic_temporal
    years_data = {}

    for yi, year in enumerate(args.years):
        degradation_factor = 1.0 - yi * 0.05  # 5% degradation per year
        opt_seq = np.random.randn(T, 10, H, W).astype(np.float32)
        sar_seq = np.random.randn(T, 5, H, W).astype(np.float32)
        doy = np.linspace(0, 1, T).astype(np.float32)
        dem = np.random.randn(5, H, W).astype(np.float32)

        # Simulate realistic label pattern with year-dependent noise
        label = np.zeros((H, W), dtype=np.int64)
        for ci in range(1, 7):
            mask_h = (slice(ci * 8, (ci + 1) * 8), slice(0, W))
            label[ci * 8:(ci + 1) * 8, :] = ci
        # Add noise proportional to degradation
        noise_mask = np.random.random((H, W)) < (0.05 * yi)
        label[noise_mask] = np.random.randint(1, 7, size=noise_mask.sum())

        years_data[str(year)] = {
            "opt_sequence": opt_seq,
            "sar_sequence": sar_seq,
            "doy_norm": doy,
            "dem": dem,
            "label": label,
            "year": year,
        }

    return years_data


def load_real_data(args):
    """Load per-year data from directory structure.

    Expected layout:
      data_dir/
        2022/
          opt_sequence.npy, sar_sequence.npy, doy_norm.npy, dem.npy, label.npy
        2023/
          ...
    """
    years_data = {}
    for year in args.years:
        year_dir = os.path.join(args.data_dir, str(year))
        if not os.path.isdir(year_dir):
            print(f"  Warning: year directory not found: {year_dir}, skipping")
            continue

        opt_seq = np.load(os.path.join(year_dir, "opt_sequence.npy"))
        sar_seq = np.load(os.path.join(year_dir, "sar_sequence.npy"))
        doy_norm = np.load(os.path.join(year_dir, "doy_norm.npy"))

        dem_path = os.path.join(year_dir, "dem.npy")
        dem = np.load(dem_path) if os.path.exists(dem_path) else None

        label_path = os.path.join(year_dir, "label.npy")
        if not os.path.exists(label_path):
            print(f"  Warning: no label.npy in {year_dir}, skipping")
            continue
        label = np.load(label_path)

        years_data[str(year)] = {
            "opt_sequence": opt_seq,
            "sar_sequence": sar_seq,
            "doy_norm": doy_norm,
            "dem": dem,
            "label": label,
            "year": year,
        }
        print(f"  Loaded {year}: opt={opt_seq.shape}, label={label.shape}")

    return years_data


def main():
    args = parse_args()
    model, device = load_model(args)

    if args.synthetic:
        test_years = load_synthetic_data(args)
    else:
        test_years = load_real_data(args)

    if len(test_years) < 2:
        print("Need at least 2 years of data for degradation analysis. Exiting.")
        return

    # Check if model supports DEM
    has_dem = any(data.get("dem") is not None for data in test_years.values())

    # Adapt model calls based on DEM availability
    from utils.evaluation import ValidationStrategy
    validator = ValidationStrategy()

    ref_year = str(args.reference_year) if args.reference_year else None
    summary = validator.cross_year_degradation_analysis(
        model, test_years,
        reference_year=ref_year,
        device=str(device),
        output_dir=args.output,
    )

    # ── Generate visualizations ──
    try:
        from scripts.visualize import (
            plot_cross_year_degradation,
            plot_cross_year_degradation_heatmap,
        )
        plot_cross_year_degradation(
            summary,
            save_path=os.path.join(args.output, "cross_year_degradation.png"))
        plot_cross_year_degradation_heatmap(
            summary,
            save_path=os.path.join(args.output, "cross_year_degradation_heatmap.png"))
    except Exception as e:
        print(f"  Visualization skipped: {e}")

    print(f"\nDone. Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
