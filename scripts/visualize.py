"""
Visualization utilities for FusionCropNetV5EDL.

Generates:
  - Model architecture diagram (requires torchviz)
  - Temporal attention weight heatmaps
  - Spatial feature maps from encoder layers
  - Prediction vs ground truth comparison
  - EDL uncertainty maps (vacuity, dissonance)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_model_architecture(model, save_path="model_architecture.png"):
    """Visualize model structure (requires torchviz + graphviz)."""
    try:
        from torchviz import make_dot

        device = next(model.parameters()).device
        dummy_opt = torch.randn(1, 10, 10, 32, 32).to(device)
        dummy_sar = torch.randn(1, 10, 5, 32, 32).to(device)
        dummy_dem = torch.randn(1, 5, 32, 32).to(device)
        dummy_doy = torch.randn(1, 10).to(device)

        output = model(dummy_opt, dummy_sar, dummy_dem, dummy_doy)
        if isinstance(output, tuple):
            output = output[0]
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.render(save_path.replace(".png", ""), format="png", cleanup=True)
        print(f"Model architecture saved to: {save_path}")
    except ImportError:
        print("Note: torchviz/graphviz not installed. Run: pip install torchviz graphviz")
        print("Skipping architecture diagram.")


def visualize_temporal_attention(model, opt_seq, sar_seq, dem, doy,
                                  save_dir="attention_plots"):
    """Visualize temporal attention weights from the optical temporal encoder.

    Hooks into the TransformerEncoder's self-attention to extract
    attention matrices for a center pixel.
    """
    Path(save_dir).mkdir(exist_ok=True)
    device = next(model.parameters()).device

    B = 1
    opt_t = torch.from_numpy(opt_seq).float().unsqueeze(0).to(device)
    sar_t = torch.from_numpy(sar_seq).float().unsqueeze(0).to(device)
    dem_t = torch.from_numpy(dem).float().unsqueeze(0).to(device)
    doy_t = torch.from_numpy(doy).float().unsqueeze(0).to(device)

    attention_maps = []

    def hook_fn(module, input, output):
        # output is (N, T, D) after self-attention
        # We can't easily get attention weights from nn.TransformerEncoder
        # Use a forward hook on MultiheadAttention instead
        pass

    # Hook into the temporal encoder's transformer layers
    hooks = []
    for layer in model.opt_temporal.transformer.layers:
        def make_hook(layer_idx):
            def attn_hook(module, input, output):
                attention_maps.append({
                    "layer": layer_idx,
                    "output_shape": output.shape,
                })
            return attn_hook
        hook = layer.self_attn.register_forward_hook(make_hook(len(attention_maps)))
        hooks.append(hook)

    model.eval()
    with torch.no_grad():
        alpha = model(opt_t, sar_t, dem_t, doy_t)
        if isinstance(alpha, tuple):
            alpha = alpha[0]

    for h in hooks:
        h.remove()

    # If hooks captured attention data, plot it
    if attention_maps:
        n_layers = len(attention_maps)
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
        if n_layers == 1:
            axes = [axes]
        for i, am in enumerate(attention_maps):
            axes[i].text(0.5, 0.5, f"Layer {am['layer']}\nShape: {am['output_shape']}",
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Temporal Layer {am['layer']}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/attention_info.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot DOY encoding as an alternative view
    T = doy_t.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(np.arange(T), doy_t[0].cpu().numpy(), 'o-', linewidth=2, color='#2196F3')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Normalized DOY')
    axes[0].set_title('Day-of-Year Encoding')
    axes[0].grid(True, alpha=0.3)

    # Pixel-level temporal feature norms as proxy for attention
    pre_head, *_ = model._encode(opt_t, sar_t, dem_t, doy_t, None, None)
    feat_norms = pre_head[0].norm(dim=0).cpu().numpy()

    axes[1].imshow(feat_norms, cmap='hot')
    axes[1].set_title('Feature Activation Map (|pre_head|)')
    axes[1].axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=axes[1])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/temporal_encoding.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Temporal attention plots saved to: {save_dir}/")


def visualize_encoder_features(model, opt_seq, doy,
                                patch_size=32, save_dir="feature_plots"):
    """Visualize optical encoder FPN feature maps."""
    Path(save_dir).mkdir(exist_ok=True)
    device = next(model.parameters()).device

    H, W = opt_seq.shape[2], opt_seq.shape[3]
    r = H // 2 - patch_size // 2
    c = W // 2 - patch_size // 2

    x = opt_seq[:, :, r:r + patch_size, c:c + patch_size]
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)

    B, T, C, Hp, Wp = x_t.shape
    x_flat = x_t.view(B * T, C, Hp, Wp)

    model.eval()
    with torch.no_grad():
        main_feat, p2_feat, p3_feat = model.opt_enc(x_flat)

    feature_sets = {
        "FPN_Output": main_feat[0],
        "FPN_P2_Skip": p2_feat[0],
        "FPN_P3_Skip": p3_feat[0],
    }

    for name, feat in feature_sets.items():
        n_ch = min(feat.shape[0], 16)
        n_cols = 8
        n_rows = (n_ch + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 1.8, n_rows * 1.8))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes = axes.flatten()

        for i in range(n_ch):
            axes[i].imshow(feat[i].cpu().numpy(), cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Ch{i + 1}', fontsize=7)
        for i in range(n_ch, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'{name} ({feat.shape[0]} channels)', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name}.png", dpi=120, bbox_inches='tight')
        plt.close()

    print(f"Encoder feature maps saved to: {save_dir}/")


def visualize_prediction_comparison(pred_map, label_map=None,
                                     save_path="prediction_comparison.png",
                                     num_classes=7):
    """Visualize prediction vs ground truth side by side."""
    n_plots = 2 if label_map is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap('tab10', num_classes)
    class_names = {
        0: "Background", 1: "Winter Wheat", 2: "Summer Corn",
        3: "Rice", 4: "Soybean", 5: "Cotton", 6: "Other"
    }

    im0 = axes[0].imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[0].set_title('Prediction', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if label_map is not None:
        im1 = axes[1].imshow(label_map, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction comparison saved to: {save_path}")


def visualize_uncertainty(model, opt_seq, sar_seq, dem, doy,
                           save_dir="uncertainty_plots"):
    """Visualize EDL uncertainty maps (vacuity, dissonance)."""
    Path(save_dir).mkdir(exist_ok=True)
    device = next(model.parameters()).device

    opt_t = torch.from_numpy(opt_seq).float().unsqueeze(0).to(device)
    sar_t = torch.from_numpy(sar_seq).float().unsqueeze(0).to(device)
    dem_t = torch.from_numpy(dem).float().unsqueeze(0).to(device)
    doy_t = torch.from_numpy(doy).float().unsqueeze(0).to(device)

    from models.fusion_net_v5_edl import dirichlet_to_predictions

    model.eval()
    with torch.no_grad():
        alpha = model(opt_t, sar_t, dem_t, doy_t)
        if isinstance(alpha, tuple):
            alpha = alpha[0]
        preds = dirichlet_to_predictions(alpha)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(preds["pred_class"][0].cpu(), cmap='tab10', vmin=0, vmax=6)
    axes[0].set_title('Prediction')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(preds["vacuity"][0].cpu(), cmap='Reds', vmin=0)
    axes[1].set_title('Vacuity (Data Uncertainty)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(preds["dissonance"][0].cpu(), cmap='Blues', vmin=0)
    axes[2].set_title('Dissonance (Epistemic Uncertainty)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    probs_max = preds["probs"][0].max(dim=0).values.cpu()
    im3 = axes[3].imshow(probs_max, cmap='Greens', vmin=0, vmax=1)
    axes[3].set_title('Max Probability')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/uncertainty_maps.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Uncertainty maps saved to: {save_dir}/")


def main(cfg):
    """Main visualization function."""
    from models.fusion_net_v5_edl import FusionCropNetV5EDL
    from models.fusion_net_v5pro import FusionCropNetV5Pro

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    use_v5pro = cfg.get("model_type", "").lower() == "v5pro"
    if use_v5pro:
        model = FusionCropNetV5Pro(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=cfg.get("num_classes", 7),
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4).to(device)
    else:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=cfg.get("num_classes", 7),
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4).to(device)

    if os.path.exists(cfg.get("model_path", "")):
        checkpoint = torch.load(cfg["model_path"], map_location=device)
        model.load_state_dict(
            checkpoint.get("model_state", checkpoint), strict=False)
        print(f"Loaded model: {cfg['model_path']}")
    else:
        print(f"Model not found: {cfg.get('model_path', '')}, using random init")

    model.eval()

    # Load data
    print("Loading data...")
    data_dir = cfg.get("data_dir", "data/processed")
    if os.path.exists(data_dir):
        opt_seq = np.load(os.path.join(data_dir, "opt_sequence.npy"))
        sar_seq = np.load(os.path.join(data_dir, "sar_sequence.npy"))
        doy_norm = np.load(os.path.join(data_dir, "doy_norm.npy"))
        dem_path = os.path.join(data_dir, "dem.npy")
        dem = np.load(dem_path) if os.path.exists(dem_path) else np.zeros(
            (5, opt_seq.shape[2], opt_seq.shape[3]), dtype=np.float32)
    else:
        print(f"Data dir not found: {data_dir}, using random data")
        opt_seq = np.random.randn(12, 10, 64, 64).astype(np.float32)
        sar_seq = np.random.randn(12, 5, 64, 64).astype(np.float32)
        dem = np.random.randn(5, 64, 64).astype(np.float32)
        doy_norm = np.linspace(0, 1, 12).astype(np.float32)

    ps = cfg.get("patch_size", 32)
    # Take center patch
    H, W = opt_seq.shape[2], opt_seq.shape[3]
    r, c = H // 2 - ps // 2, W // 2 - ps // 2
    opt_patch = opt_seq[:, :, r:r + ps, c:c + ps]
    sar_patch = sar_seq[:, :, r:r + ps, c:c + ps]
    dem_patch = dem[:, r:r + ps, c:c + ps]

    # 1. Architecture diagram
    if cfg.get("plot_architecture", True):
        plot_model_architecture(model, "model_architecture.png")

    # 2. Temporal attention
    if cfg.get("plot_attention", True):
        print("Visualizing temporal attention...")
        visualize_temporal_attention(model, opt_patch, sar_patch, dem_patch,
                                      doy_norm, "attention_plots")

    # 3. Encoder feature maps
    if cfg.get("plot_features", True):
        print("Visualizing encoder features...")
        visualize_encoder_features(model, opt_seq, doy_norm,
                                    cfg.get("patch_size", 32), "feature_plots")

    # 4. Uncertainty maps
    if cfg.get("plot_uncertainty", False):
        print("Visualizing uncertainty...")
        visualize_uncertainty(model, opt_patch, sar_patch, dem_patch,
                               doy_norm, "uncertainty_plots")

    # 5. Prediction comparison (requires label)
    if cfg.get("plot_prediction", True) and cfg.get("label_path"):
        print("Generating prediction comparison...")
        from models.fusion_net_v5_edl import dirichlet_to_predictions

        opt_t = torch.from_numpy(opt_patch).float().unsqueeze(0).to(device)
        sar_t = torch.from_numpy(sar_patch).float().unsqueeze(0).to(device)
        dem_t = torch.from_numpy(dem_patch).float().unsqueeze(0).to(device)
        doy_t = torch.from_numpy(doy_norm).float().unsqueeze(0).to(device)

        with torch.no_grad():
            alpha = model(opt_t, sar_t, dem_t, doy_t)
            if isinstance(alpha, tuple):
                alpha = alpha[0]
            preds_obj = dirichlet_to_predictions(alpha)
            pred_map = preds_obj["pred_class"][0].cpu().numpy()

        label_map = None
        if os.path.exists(cfg["label_path"]):
            label_map = np.load(cfg["label_path"])
            hr, hc = r + ps, c + ps
            label_map = label_map[r:hr, c:hc]

        visualize_prediction_comparison(
            pred_map, label_map, "prediction_comparison.png",
            cfg.get("num_classes", 7))

    print("\n" + "=" * 50)
    print("All visualizations complete!")
    print("=" * 50)


# =============================================================================
# Calibration plots
# =============================================================================
def plot_reliability_diagram(report, save_path="reliability_diagram.png",
                             title="Reliability Diagram"):
    """Plot reliability diagram (confidence vs accuracy per bin)."""
    import matplotlib.pyplot as plt

    bins = report.get("ECE_bins", [])
    if not bins:
        print("No bin data in report, skipping reliability diagram.")
        return

    confs = [b["conf"] for b in bins if b["n"] > 0]
    accs = [b["acc"] for b in bins if b["n"] > 0]
    weights = [b["n"] for b in bins if b["n"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')
    sizes = np.array(weights) / max(weights) * 150 + 20
    ax.scatter(confs, accs, s=sizes, c='#2196F3', edgecolors='navy',
              alpha=0.8, zorder=3, label='Bins (size ∝ count)')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')

    gap_max = max(abs(c - a) for c, a in zip(confs, accs)) if confs else 0
    ax.set_xlabel('Mean Confidence per Bin', fontsize=11)
    ax.set_ylabel('Accuracy per Bin', fontsize=11)
    ax.set_title(f'{title}\nECE={report.get("ECE", 0):.4f}  '
                 f'Gap(max)={gap_max:.4f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Right: confidence histogram (correct vs incorrect)
    ax2 = axes[1]
    raw = report.get("_raw", {})
    conf = raw.get("confidences", None)
    corr = raw.get("correctness", None)
    if conf is not None and corr is not None:
        conf_c = conf[corr == 1]
        conf_w = conf[corr == 0]
        bins_h = np.linspace(0, 1, 21)
        ax2.hist(conf_c, bins=bins_h, alpha=0.7, color='#4CAF50',
                label=f'Correct (n={len(conf_c)})')
        ax2.hist(conf_w, bins=bins_h, alpha=0.7, color='#F44336',
                label=f'Wrong (n={len(conf_w)})')
        ax2.set_xlabel('Confidence', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Confidence Distribution: Correct vs Wrong', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reliability diagram saved to: {save_path}")


def plot_uncertainty_error_map(vacuity_map, error_map, pred_map=None,
                                save_path="uncertainty_error_map.png"):
    """Overlay uncertainty with prediction errors.

    Args:
        vacuity_map: (H, W) data uncertainty
        error_map:   (H, W) boolean error mask
        pred_map:    (H, W) predicted class labels (optional)
    """
    import matplotlib.pyplot as plt

    n_cols = 3 if pred_map is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
    if n_cols == 2:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    im0 = axes[0].imshow(vacuity_map, cmap='hot', interpolation='nearest')
    axes[0].set_title('Vacuity (Data Uncertainty)', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(error_map, cmap='Reds', interpolation='nearest',
                          vmin=0, vmax=1)
    axes[1].set_title('Prediction Errors', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    if pred_map is not None:
        im2 = axes[2].imshow(pred_map, cmap='tab10', interpolation='nearest')
        axes[2].set_title('Predictions', fontsize=11)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Uncertainty-error map saved to: {save_path}")


def plot_rejection_curve(report, save_path="rejection_curve.png"):
    """Plot accuracy vs retention curve for uncertainty-based rejection."""
    import matplotlib.pyplot as plt

    curve = report.get("RejectionCurve", [])
    if not curve:
        print("No rejection curve data, skipping.")
        return

    ret = [p["retention"] for p in curve]
    acc = [p["accuracy"] for p in curve]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ret, acc, 'o-', linewidth=2, markersize=5, color='#2196F3')
    ax.fill_between(ret, acc, alpha=0.15, color='#2196F3')
    ax.set_xlabel('Retention Ratio (most confident pixels kept)', fontsize=11)
    ax.set_ylabel('Accuracy on Retained Pixels', fontsize=11)
    ax.set_title('Uncertainty Rejection Curve\n'
                 f'(OA full={report.get("OA", 0):.4f}, '
                 f'AUROC={report.get("AUROC_error_detection", 0):.4f})', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rejection curve saved to: {save_path}")


def plot_per_class_calibration(report, save_path="per_class_calibration.png"):
    """Bar chart of per-class ECE, accuracy, and mean confidence."""
    import matplotlib.pyplot as plt

    pc = report.get("PerClass", {})
    classes = [pc[k]["name"] for k in sorted(pc.keys()) if pc[k].get("ece") is not None]
    eces = [pc[k]["ece"] for k in sorted(pc.keys()) if pc[k].get("ece") is not None]
    accs = [pc[k]["acc"] for k in sorted(pc.keys()) if pc[k].get("ece") is not None]
    confs = [pc[k]["mean_conf"] for k in sorted(pc.keys()) if pc[k].get("ece") is not None]

    if not classes:
        print("No per-class data, skipping.")
        return

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w, accs, w, color='#4CAF50', label='Accuracy', alpha=0.85)
    ax.bar(x, confs, w, color='#2196F3', label='Mean Confidence', alpha=0.85)
    ax.bar(x + w, eces, w, color='#FF9800', label='ECE', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Per-Class Calibration', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class calibration chart saved to: {save_path}")


# =============================================================================
# Cross-Year Degradation Visualization
# =============================================================================
def plot_cross_year_degradation(degradation_summary,
                                 save_path="cross_year_degradation.png"):
    """Plot per-class IoU degradation across test years.

    Creates a 2-panel figure:
      Left: Per-class IoU comparison across years (grouped bar chart)
      Right: Stability ranking (horizontal bar chart)

    Args:
        degradation_summary: dict from ValidationStrategy.cross_year_degradation_analysis()
    """
    import matplotlib.pyplot as plt

    year_results = degradation_summary.get("year_results", {})
    stability = degradation_summary.get("stability_ranking", [])
    ref_year = degradation_summary.get("reference_year", "")

    if not year_results or not stability:
        print("No degradation data, skipping cross-year plot.")
        return

    years = list(year_results.keys())
    class_names = [s["class_name"] for s in stability]
    num_classes = len(class_names)

    # Build IoU matrix: rows=classes, cols=years
    iou_matrix = []
    for s in stability:
        cid = s["class_id"]
        row = []
        for y in years:
            yr_iou = year_results[y].get("IoU_per_class", [])
            row.append(yr_iou[cid - 1] if cid - 1 < len(yr_iou) else 0)
        iou_matrix.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: grouped bar chart: per-class IoU per year ──
    ax = axes[0]
    x = np.arange(num_classes)
    n_years = len(years)
    width = 0.8 / max(n_years, 1)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_years))

    for yi, year in enumerate(years):
        offset = (yi - n_years / 2 + 0.5) * width
        values = [iou_matrix[ci][yi] for ci in range(num_classes)]
        label = f"{year}" + (" (ref)" if year == ref_year else "")
        ax.bar(x + offset, values, width, color=colors[yi],
               alpha=0.85, label=label, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_title(f'Per-Class IoU Across Years (ref={ref_year})', fontsize=12)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    # ── Right: stability ranking ──
    ax2 = axes[1]
    scores = [s["stability_score"] for s in stability]
    degrads = [s["mean_degradation"] for s in stability]
    names_rev = list(reversed(class_names))
    scores_rev = list(reversed(scores))
    degrads_rev = list(reversed(degrads))

    bar_colors = []
    for d in degrads_rev:
        if d > 0.05:
            bar_colors.append('#F44336')
        elif d > 0.01:
            bar_colors.append('#FF9800')
        elif d < -0.01:
            bar_colors.append('#4CAF50')
        else:
            bar_colors.append('#2196F3')

    ax2.barh(range(num_classes), scores_rev, color=bar_colors, alpha=0.85,
             edgecolor='white')
    ax2.set_yticks(range(num_classes))
    ax2.set_yticklabels(names_rev, fontsize=9)
    ax2.set_xlabel('Stability Score (higher = more stable)', fontsize=11)
    ax2.set_title('Cross-Year Stability Ranking', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Improved (>1% IoU)'),
        Patch(facecolor='#2196F3', label='Stable (±1%)'),
        Patch(facecolor='#FF9800', label='Mild degradation (1-5%)'),
        Patch(facecolor='#F44336', label='Severe degradation (>5%)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=7, loc='lower right')

    # Annotate mean degradation on each bar
    for i, (s, d) in enumerate(zip(scores_rev, degrads_rev)):
        ax2.text(s + 0.01, i, f'Δ={d:+.3f}', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cross-year degradation plot saved to: {save_path}")


def plot_cross_year_degradation_heatmap(degradation_summary,
                                          save_path="cross_year_degradation_heatmap.png"):
    """Plot per-class IoU degradation as a heatmap (classes × test years)."""
    import matplotlib.pyplot as plt

    per_class = degradation_summary.get("per_class_degradation", {})
    year_results = degradation_summary.get("year_results", {})
    ref_year = degradation_summary.get("reference_year", "")

    if not per_class:
        print("No per-class degradation data, skipping heatmap.")
        return

    test_years = [y for y in per_class.keys()]
    if not test_years:
        return

    first_deltas = per_class[test_years[0]]
    class_names = [d["class_name"] for d in first_deltas]
    n_classes = len(class_names)
    n_years = len(test_years)

    delta_matrix = np.zeros((n_classes, n_years))
    for yi, year in enumerate(test_years):
        for ci, d in enumerate(per_class[year]):
            delta_matrix[ci, yi] = d["delta_IoU"]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_classes * 0.4)),
                              gridspec_kw={'width_ratios': [3, 1]})

    # ── Left: heatmap ──
    vmax = max(abs(delta_matrix).max(), 0.01)
    im = axes[0].imshow(delta_matrix, cmap='RdYlGn', aspect='auto',
                         vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[0].set_xticks(range(n_years))
    axes[0].set_xticklabels([f"{y}\nΔmIoU={degradation_summary.get('mIoU_degradation',{}).get(y,0):+.3f}"
                              for y in test_years], fontsize=8)
    axes[0].set_yticks(range(n_classes))
    axes[0].set_yticklabels(class_names, fontsize=9)
    axes[0].set_title(f'Per-Class IoU Degradation Heatmap (ref={ref_year})', fontsize=12)

    # Annotate cells
    for yi in range(n_years):
        for ci in range(n_classes):
            val = delta_matrix[ci, yi]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            axes[0].text(yi, ci, f'{val:+.3f}', ha='center', va='center',
                        fontsize=8, color=color, weight='bold')

    plt.colorbar(im, ax=axes[0], fraction=0.046, label='ΔIoU')

    # ── Right: mean degradation per class summary ──
    mean_deltas = delta_matrix.mean(axis=1)
    bar_colors = ['#F44336' if d > 0.03 else '#FF9800' if d > 0.01
                  else '#4CAF50' if d < -0.01 else '#2196F3'
                  for d in mean_deltas]
    axes[1].barh(range(n_classes), mean_deltas, color=bar_colors, alpha=0.85,
                 edgecolor='white')
    axes[1].set_yticks(range(n_classes))
    axes[1].set_yticklabels(class_names, fontsize=9)
    axes[1].set_xlabel('Mean ΔIoU across years', fontsize=10)
    axes[1].set_title('Mean Degradation', fontsize=11)
    axes[1].axvline(x=0, color='black', linewidth=0.5, linestyle='--')
    axes[1].grid(True, alpha=0.3, axis='x')

    for ci, (val, name) in enumerate(zip(mean_deltas, class_names)):
        axes[1].text(val + 0.002 * np.sign(val) if abs(val) > 0.001 else val + 0.002,
                     ci, f'{val:+.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cross-year degradation heatmap saved to: {save_path}")


# =============================================================================
# Interpretability plots
# =============================================================================
def plot_gradcam_heatmaps(gradcam_maps, class_names=None, opt_img=None,
                           save_path="gradcam_heatmaps.png"):
    """Plot Grad-CAM heatmaps for all classes in a grid.

    Args:
        gradcam_maps: dict {class_idx: (H, W) numpy array}
        class_names:  dict {class_idx: name}
        opt_img:      (C, H, W) or (H, W) background image
    """
    import matplotlib.pyplot as plt

    K = len(gradcam_maps)
    if class_names is None:
        class_names = {k: f"Class {k}" for k in gradcam_maps}

    n_cols = min(4, K)
    n_rows = (K + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.5, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (k, hm) in enumerate(sorted(gradcam_maps.items())):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]
        im = ax.imshow(hm, cmap='jet', interpolation='bilinear')
        ax.set_title(class_names.get(k, f"Class {k}"), fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    for idx in range(K, n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].axis('off')

    plt.suptitle('Grad-CAM: Per-Class Spatial Attribution', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM heatmaps saved to: {save_path}")


def plot_modality_contribution(ablation_results, save_path="modality_contribution.png"):
    """Bar chart of modality contributions from ablation study."""
    import matplotlib.pyplot as plt

    ri = ablation_results.get("relative_importance", {})
    if not ri:
        print("No relative importance data, skipping.")
        return

    modalities = list(ri.keys())
    values = [ri[m] for m in modalities]
    colors = {'optical': '#4CAF50', 'sar': '#2196F3', 'dem': '#FF9800'}
    bar_colors = [colors.get(m, '#9E9E9E') for m in modalities]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: relative importance
    axes[0].bar(modalities, values, color=bar_colors, alpha=0.85, edgecolor='black')
    axes[0].set_ylabel('Relative Importance', fontsize=11)
    axes[0].set_title('Modality Contribution (from Ablation)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    for m, v in zip(modalities, values):
        axes[0].text(m, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

    # Right: agreement & KL shift per configuration
    configs = ["full", "no_opt", "no_sar", "no_dem"]
    labels = ["Full", "w/o Optical", "w/o SAR", "w/o DEM"]
    agreements = [ablation_results.get(c, {}).get("agreement", 0) for c in configs]
    kl_shifts = [ablation_results.get(c, {}).get("prob_shift_kl", 0) for c in configs]

    x = np.arange(len(configs))
    w = 0.35
    axes[1].bar(x - w/2, agreements, w, color='#4CAF50', alpha=0.85,
               label='Agreement with Full')
    ax2 = axes[1].twinx()
    ax2.bar(x + w/2, kl_shifts, w, color='#F44336', alpha=0.85, label='KL Shift')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('Agreement', fontsize=11)
    ax2.set_ylabel('KL Divergence Shift', fontsize=11)
    axes[1].set_title('Ablation Impact', fontsize=12)
    axes[1].legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Modality contribution chart saved to: {save_path}")


def plot_temporal_importance(importance, save_path="temporal_importance.png"):
    """Plot per-timestep importance scores."""
    import matplotlib.pyplot as plt

    T = len(importance)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, T))
    bars = ax.bar(range(T), importance, color=colors, edgecolor='black', alpha=0.85)
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Importance (normalized)', fontsize=11)
    ax.set_title('Temporal Importance: Per-Timestep Contribution', fontsize=12)
    ax.set_xticks(range(T))
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight top-3 steps
    top_k = np.argsort(importance)[-3:]
    for idx in top_k:
        bars[idx].set_edgecolor('#F44336')
        bars[idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Temporal importance plot saved to: {save_path}")


def plot_band_importance(band_importance, save_path="band_importance.png"):
    """Bar chart of spectral band importance scores."""
    import matplotlib.pyplot as plt

    optical_labels = ['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR',
                      'NDVI', 'NDWI', 'EVI', 'LSWI', 'BSI', 'NBR']
    sar_labels = ['VV', 'VH', 'VV/VH', 'RVI', 'NLI']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    opt_imp = band_importance.get("optical_bands", [])
    sar_imp = band_importance.get("sar_bands", [])

    if opt_imp:
        x_opt = range(len(opt_imp))
        colors_opt = ['#1B5E20' if i < 4 else '#4CAF50' for i in x_opt]
        axes[0].bar(x_opt, opt_imp, color=colors_opt, alpha=0.85, edgecolor='black')
        labels_opt = optical_labels[:len(opt_imp)]
        axes[0].set_xticks(list(x_opt))
        axes[0].set_xticklabels(labels_opt, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Importance', fontsize=11)
        axes[0].set_title('Optical Band Importance', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')

    if sar_imp:
        x_sar = range(len(sar_imp))
        axes[1].bar(x_sar, sar_imp, color='#1565C0', alpha=0.85, edgecolor='black')
        labels_sar = sar_labels[:len(sar_imp)]
        axes[1].set_xticks(list(x_sar))
        axes[1].set_xticklabels(labels_sar, fontsize=9)
        axes[1].set_ylabel('Importance', fontsize=11)
        axes[1].set_title('SAR Band Importance', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Spectral Band Importance (Occlusion-Based)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Band importance chart saved to: {save_path}")


def plot_pixel_explanation(report, save_path="pixel_explanation.png"):
    """Compare vacuity, dissonance, margin distributions: correct vs incorrect."""
    import matplotlib.pyplot as plt

    corr = report.get("correct", {})
    incorr = report.get("incorrect", {})

    metrics = ["vacuity", "dissonance", "margin", "entropy", "confidence"]
    labels = ["Vacuity", "Dissonance", "Top2 Margin", "Entropy", "Confidence"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, label in zip(axes, metrics, labels):
        c = corr.get(metric, {})
        w = incorr.get(metric, {})

        x_pos = [0, 1]
        means = [c.get("mean", 0), w.get("mean", 0)]
        stds = [c.get("std", 0), w.get("std", 0)]

        ax.bar(x_pos, means, yerr=stds, color=['#4CAF50', '#F44336'],
               alpha=0.85, edgecolor='black', capsize=8, width=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Correct', 'Wrong'], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add text for median values
        ax.text(0, means[0] + stds[0] + 0.02,
                f"med={c.get('median', 0):.3f}", ha='center', fontsize=7)
        ax.text(1, means[1] + stds[1] + 0.02,
                f"med={w.get('median', 0):.3f}", ha='center', fontsize=7)

    plt.suptitle('Pixel-Level Explanation: Correct vs Incorrect Predictions',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pixel explanation chart saved to: {save_path}")


# =============================================================================
# Full calibration + interpretability visualization pipeline
# =============================================================================
def run_full_analysis(model, opt_seq, sar_seq, dem, doy, label_map,
                       num_classes=7, output_dir="analysis_output", device="cpu"):
    """Run complete calibration + interpretability analysis and save all plots.

    Args:
        model:       FusionCropNetV5EDL
        opt_seq:     (B, T, C_opt, H, W) numpy or tensor
        sar_seq:     (B, T, C_sar, H, W) numpy or tensor
        dem:         (B, C_dem, H, W) numpy or tensor
        doy:         (B, T) numpy or tensor
        label_map:   (B, H, W) numpy ground truth
        num_classes: int
        output_dir:  str
        device:      str or torch.device
    """
    from pathlib import Path
    from models.fusion_net_v5_edl import dirichlet_to_predictions
    from utils.calibration import calibration_report, print_calibration_report
    from utils.interpretability import (modality_ablation, temporal_importance,
                                         spectral_band_importance,
                                         pixel_explanation_report,
                                         gradcam_per_class,
                                         confusion_region_analysis)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Full Calibration & Interpretability Analysis")
    print(f"{'='*60}")

    device = torch.device(device) if isinstance(device, str) else device
    if device.type != "cpu" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model.to(device).eval()

    # Convert to tensors
    if isinstance(opt_seq, np.ndarray):
        opt_t = torch.from_numpy(opt_seq).float().to(device)
        sar_t = torch.from_numpy(sar_seq).float().to(device)
        dem_t = torch.from_numpy(dem).float().to(device)
        doy_t = torch.from_numpy(doy).float().to(device)
        lbl = label_map
    else:
        opt_t = opt_seq.to(device)
        sar_t = sar_seq.to(device)
        dem_t = dem.to(device)
        doy_t = doy.to(device)
        lbl = label_map.cpu().numpy() if isinstance(label_map, torch.Tensor) else label_map

    # 1. Get predictions and alpha
    print("\n[1/6] Running inference...")
    with torch.no_grad():
        alpha = model(opt_t, sar_t, dem_t, doy_t)
        if isinstance(alpha, tuple):
            alpha = alpha[0]
        preds = dirichlet_to_predictions(alpha)

    alpha_np = alpha.cpu().numpy()
    pred_np = preds["pred_class"].cpu().numpy()
    vacuity_np = preds["vacuity"].cpu().numpy()
    dissonance_np = preds["dissonance"].cpu().numpy()

    # 2. Calibration analysis
    print("[2/6] Computing calibration metrics...")
    report = calibration_report(alpha_np, lbl, num_classes,
                                 n_bins=15)
    print_calibration_report(report)

    # Plot calibration
    plot_reliability_diagram(report,
                             f"{output_dir}/reliability_diagram.png")
    plot_per_class_calibration(report,
                               f"{output_dir}/per_class_calibration.png")
    plot_rejection_curve(report,
                         f"{output_dir}/rejection_curve.png")

    # Error map overlay
    error_map = (pred_np[0] != lbl[0]).astype(np.float32) if lbl.ndim == 3 else (
        pred_np[0] != lbl).astype(np.float32)
    plot_uncertainty_error_map(
        vacuity_np[0], error_map, pred_np[0],
        f"{output_dir}/uncertainty_error_overlay.png")

    # 3. Pixel-level explanation
    print("[3/6] Computing pixel-level explanations...")
    px_report = pixel_explanation_report(alpha_np, lbl, num_classes)
    plot_pixel_explanation(px_report,
                           f"{output_dir}/pixel_explanation.png")

    # 4. Modality ablation
    print("[4/6] Running modality ablation...")
    try:
        abl_results = modality_ablation(model, opt_t, sar_t, dem_t, doy_t,
                                        num_classes=num_classes, device=device)
        plot_modality_contribution(abl_results,
                                   f"{output_dir}/modality_contribution.png")
    except Exception as e:
        print(f"  Modality ablation skipped: {e}")

    # 5. Temporal importance
    print("[5/6] Computing temporal importance...")
    try:
        t_imp, _ = temporal_importance(model, opt_t, sar_t, dem_t, doy_t,
                                       device=device)
        plot_temporal_importance(t_imp,
                                 f"{output_dir}/temporal_importance.png")
    except Exception as e:
        print(f"  Temporal importance skipped: {e}")

    # 6. Spectral band importance
    print("[6/6] Computing band importance...")
    try:
        band_imp = spectral_band_importance(model, opt_t, sar_t, dem_t, doy_t,
                                            device=device)
        plot_band_importance(band_imp,
                             f"{output_dir}/band_importance.png")
    except Exception as e:
        print(f"  Band importance skipped: {e}")

    # Confusion region analysis
    print("\nConfusion region analysis...")
    confusion = confusion_region_analysis(alpha_np, lbl, num_classes)
    top_confused = sorted(confusion.items(),
                          key=lambda x: x[1]["n"], reverse=True)[:5]
    print("Top-5 confused class pairs:")
    for pair_name, info in top_confused:
        print(f"  Classes {pair_name}: n={info['n']}, "
              f"vac={info.get('mean_vacuity', 'N/A')}")

    # Save report JSON
    import json
    serializable = {k: v for k, v in report.items() if k != "_raw"}
    serializable["confusion_pairs"] = {
        k: v for k, v in confusion.items()
        if v.get("mean_vacuity") is not None
    }
    with open(f"{output_dir}/analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"\nAnalysis complete. Results saved to: {output_dir}/")
    return report


if __name__ == "__main__":
    config = {
        "data_dir": "./data/processed",
        "model_path": "./best_model.pth",
        "label_path": "./data/processed/label.npy",
        "patch_size": 32,
        "num_classes": 7,
        "plot_architecture": True,
        "plot_attention": True,
        "plot_features": True,
        "plot_prediction": True,
        "plot_uncertainty": False,
    }
    main(config)
