#!/usr/bin/env python
# =============================================================================
# scripts/visualize_v6_experiments.py
# V6 Experiment Visualization Dashboard
#
# Generates publication-quality figures from v6_experiments_results.json.
#
# Figures (12+):
#   Fig 1a: Modality ablation heatmap (OA/mIoU per combination)
#   Fig 1b: Modality contribution waterfall
#   Fig 2:  Fusion mechanism ablation bar chart
#   Fig 3:  Feature derivation ablation bar chart
#   Fig 4a: Robustness — cloud cover degradation curve
#   Fig 4b: Robustness — missing timestep degradation curve
#   Fig 4c: Robustness — noise sensitivity curve
#   Fig 5a: Component ablation — leave-one-out impact
#   Fig 5b: Component ablation — cumulative addition
#   Fig 6a: Confusion matrix heatmap (raw counts)
#   Fig 6b: Confusion matrix heatmap (normalized)
#   Fig 6c: Per-class precision/recall/F1 bar chart
#
# Usage:
#   python scripts/visualize_v6_experiments.py --results v6_experiments_output/v6_experiments_results.json
# =============================================================================
import argparse, json, os, sys
import numpy as np

# ── Matplotlib setup (non-interactive backend for headless) ───────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# Style
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

CROP_NAMES = {0:"Background", 1:"Winter\nWheat", 2:"Summer\nCorn",
              3:"Rice", 4:"Soybean", 5:"Cotton", 6:"Other"}
CROP_NAMES_SHORT = {0:"BG", 1:"W.Wheat", 2:"S.Corn", 3:"Rice",
                    4:"Soybean", 5:"Cotton", 6:"Other"}

# Color palette (industrial-brutalist inspired)
C_DARK = "#1a1a2e"
C_ACCENT = "#e94560"
C_MID = "#0f3460"
C_LIGHT = "#16213e"
C_WHITE = "#eaeaea"
PALETTE = ["#e94560", "#0f3460", "#533483", "#00b4d8", "#ff6b35",
           "#2ec4b6", "#f77f00", "#7209b7", "#4cc9f0", "#e36414"]


def parse_args():
    p = argparse.ArgumentParser(description="V6 Experiment Visualization Dashboard")
    p.add_argument("--results", type=str, required=True,
                   help="Path to v6_experiments_results.json")
    p.add_argument("--output", type=str, default="./v6_experiments_output/figures",
                   help="Output directory for figures")
    p.add_argument("--format", type=str, default="png",
                   choices=["png", "pdf", "svg"], help="Output format")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────
def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(fig, name, args):
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f"{name}.{args.format}")
    fig.savefig(path, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")


def bar_labels(ax, values, fmt=".3f", offset=0.003):
    """Add value labels above/beside bars."""
    for i, v in enumerate(values):
        ax.text(i, v + offset, f"{v:{fmt}}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=C_DARK)


def extract_metric(configs_dict, metric="mIoU"):
    """Extract a metric from nested config dicts, returning (labels, values)."""
    labels, values = [], []
    for name, data in configs_dict.items():
        if isinstance(data, dict) and metric in data:
            labels.append(name)
            values.append(data[metric])
    return labels, values


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1a: Modality Ablation Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_modality_heatmap(data, args):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    configs = ["Full (Opt+SAR+DEM)", "Optical Only", "SAR Only", "DEM Only",
               "Opt+SAR (no DEM)", "Opt+DEM (no SAR)", "SAR+DEM (no Optical)"]
    metrics_list = ["OA", "mIoU", "Kappa", "vacuity_mean"]
    short_labels = ["Full", "Opt", "SAR", "DEM", "O+S", "O+D", "S+D"]

    # Heatmap data
    hm_data = np.zeros((len(configs), len(metrics_list)))
    for i, cfg in enumerate(configs):
        if cfg in data:
            for j, m in enumerate(metrics_list):
                hm_data[i, j] = data[cfg].get(m, 0)

    im = ax1.imshow(hm_data.T, aspect="auto", cmap="YlOrRd")
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(short_labels, fontsize=9)
    ax1.set_yticks(range(len(metrics_list)))
    ax1.set_yticklabels(["OA", "mIoU", "Kappa", "Vacuity"], fontsize=9)
    for i in range(len(configs)):
        for j in range(len(metrics_list)):
            ax1.text(i, j, f"{hm_data[i,j]:.3f}", ha="center", va="center",
                     fontsize=8, fontweight="bold",
                     color="white" if hm_data[i,j] > hm_data.max()/2 else C_DARK)
    ax1.set_title("Modality Combination Metrics", fontweight="bold", color=C_DARK)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # Bar chart: mIoU comparison
    miou_vals = [data.get(c, {}).get("mIoU", 0) for c in configs]
    bars = ax2.bar(range(len(configs)), miou_vals, color=PALETTE[:len(configs)], edgecolor="white")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(short_labels, fontsize=9)
    ax2.set_ylabel("mIoU")
    ax2.set_title("mIoU by Modality Combination", fontweight="bold", color=C_DARK)
    bar_labels(ax2, miou_vals)

    fig.suptitle("Experiment 1: Multi-Modal Combination Ablation",
                 fontweight="bold", fontsize=14, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig1a_modality_ablation", args)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Fusion Mechanism Ablation
# ═══════════════════════════════════════════════════════════════════════════
def fig_fusion_ablation(data, args):
    labels, miou_vals = extract_metric(data, "mIoU")
    _, oa_vals = extract_metric(data, "OA")

    # Shorten labels
    short = [l.replace(" (all on)", "").replace("No ", "−") for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(short))
    w = 0.35
    b1 = ax.bar(x - w/2, oa_vals, w, label="OA", color=PALETTE[0], edgecolor="white")
    b2 = ax.bar(x + w/2, miou_vals, w, label="mIoU", color=PALETTE[3], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Fusion Mechanism Ablation", fontweight="bold", color=C_DARK)
    ax.legend(frameon=False)
    bar_labels(ax, oa_vals, offset=0.01)
    bar_labels(ax, miou_vals, offset=0.01)

    fig.suptitle("Experiment 2: Fusion Mechanism Ablation",
                 fontweight="bold", fontsize=14, color=C_DARK)
    fig.tight_layout()
    save(fig, "fig2_fusion_ablation", args)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Feature Derivation Ablation
# ═══════════════════════════════════════════════════════════════════════════
def fig_feature_ablation(data, args):
    labels, miou_vals = extract_metric(data, "mIoU")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[0] if "All" in l else PALETTE[4] if "Visible" in l or "No" in l
              else PALETTE[3] for l in labels]
    bars = ax.barh(range(len(labels)), miou_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("mIoU")
    ax.invert_yaxis()
    for i, v in enumerate(miou_vals):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold", color=C_DARK)
    ax.set_title("Feature Band Group Ablation", fontweight="bold", color=C_DARK)
    fig.suptitle("Experiment 3: Feature Derivation Ablation",
                 fontweight="bold", fontsize=14, color=C_DARK)
    fig.tight_layout()
    save(fig, "fig3_feature_ablation", args)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4a-c: Robustness Analysis
# ═══════════════════════════════════════════════════════════════════════════
def fig_robustness(data, args):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 4a: Cloud cover
    if "cloud_cover" in data:
        cc = data["cloud_cover"]
        levels = [float(k.split("_")[1]) for k in cc.keys()]
        oa = [cc[k]["OA"] for k in sorted(cc.keys())]
        miou = [cc[k]["mIoU"] for k in sorted(cc.keys())]
        axes[0].plot(levels, oa, "o-", color=PALETTE[0], lw=2, markersize=8, label="OA")
        axes[0].plot(levels, miou, "s-", color=PALETTE[3], lw=2, markersize=8, label="mIoU")
        axes[0].set_xlabel("Cloud Cover Fraction")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Cloud Cover Degradation", fontweight="bold")
        axes[0].legend(frameon=False)
        axes[0].fill_between(levels, 0, miou, alpha=0.1, color=PALETTE[3])
        axes[0].axhline(y=miou[0], color="gray", ls="--", alpha=0.5, label=f"Clean mIoU={miou[0]:.3f}")

    # 4b: Missing timesteps
    if "missing_timesteps" in data:
        mt = data["missing_timesteps"]
        n_miss = [int(k.split("_")[1]) for k in mt.keys()]
        oa = [mt[k]["OA"] for k in sorted(mt.keys())]
        miou = [mt[k]["mIoU"] for k in sorted(mt.keys())]
        axes[1].plot(n_miss, oa, "o-", color=PALETTE[0], lw=2, markersize=8, label="OA")
        axes[1].plot(n_miss, miou, "s-", color=PALETTE[3], lw=2, markersize=8, label="mIoU")
        axes[1].set_xlabel("Missing Timesteps")
        axes[1].set_title("Temporal Robustness", fontweight="bold")
        axes[1].legend(frameon=False)

    # 4c: Noise
    if "noise" in data:
        ns = data["noise"]
        sigmas = [float(k.split("_")[1]) for k in ns.keys()]
        oa = [ns[k]["OA"] for k in sorted(ns.keys())]
        miou = [ns[k]["mIoU"] for k in sorted(ns.keys())]
        axes[2].plot(sigmas, oa, "o-", color=PALETTE[0], lw=2, markersize=8, label="OA")
        axes[2].plot(sigmas, miou, "s-", color=PALETTE[3], lw=2, markersize=8, label="mIoU")
        axes[2].set_xlabel("Gaussian Noise σ")
        axes[2].set_title("Noise Sensitivity", fontweight="bold")
        axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Experiment 4: Robustness Analysis",
                 fontweight="bold", fontsize=14, color=C_DARK)
    fig.tight_layout()
    save(fig, "fig4_robustness", args)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5a-b: Component Ablation
# ═══════════════════════════════════════════════════════════════════════════
def fig_component_ablation(data, args):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # 5a: Leave-one-out impact
    loo_items = {k: v for k, v in data.items()
                 if k.startswith("no_") and "V6 Full" not in k}
    if "V6 Full (all blocks)" in data:
        baseline_miou = data["V6 Full (all blocks)"]["mIoU"]
    else:
        baseline_miou = 0
    loo_labels = [k.replace("no_", "−") for k in loo_items.keys()]
    loo_deltas = [baseline_miou - data[k]["mIoU"] for k in loo_items.keys()]
    # Sort by impact
    sorted_idx = np.argsort(loo_deltas)[::-1]
    loo_labels = [loo_labels[i] for i in sorted_idx]
    loo_deltas = [loo_deltas[i] for i in sorted_idx]
    colors = [PALETTE[0] if d > 0 else PALETTE[3] for d in loo_deltas]
    ax1.barh(range(len(loo_labels)), loo_deltas, color=colors, edgecolor="white")
    ax1.set_yticks(range(len(loo_labels)))
    ax1.set_yticklabels(loo_labels, fontsize=9)
    ax1.set_xlabel("Δ mIoU (removing block)")
    ax1.axvline(x=0, color="gray", ls="-", alpha=0.5)
    ax1.set_title("Leave-One-Out Block Impact", fontweight="bold")
    for i, d in enumerate(loo_deltas):
        ax1.text(d + 0.001 * (1 if d >= 0 else -1), i, f"{d:+.4f}",
                 va="center", fontsize=8, fontweight="bold",
                 ha="left" if d >= 0 else "right", color=C_DARK)

    # 5b: Cumulative addition
    cumul_items = {k: v for k, v in data.items() if k.startswith("+")}
    v5_key = "V5EDL (no V6 blocks)"
    cumul_labels = []
    cumul_miou = []
    if v5_key in data:
        cumul_labels.append("V5EDL")
        cumul_miou.append(data[v5_key]["mIoU"])
    for k in sorted(cumul_items.keys()):
        cumul_labels.append(k.replace("+", ""))
        cumul_miou.append(data[k]["mIoU"])
    ax2.plot(range(len(cumul_labels)), cumul_miou, "o-", color=PALETTE[0],
             lw=2.5, markersize=10, markerfacecolor="white", markeredgewidth=2)
    ax2.set_xticks(range(len(cumul_labels)))
    ax2.set_xticklabels(cumul_labels, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("mIoU")
    ax2.set_title("Cumulative Block Addition", fontweight="bold")
    ax2.fill_between(range(len(cumul_labels)), cumul_miou, alpha=0.1, color=PALETTE[0])
    for i, v in enumerate(cumul_miou):
        ax2.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=8, fontweight="bold", color=C_DARK)

    fig.suptitle("Experiment 5: Model Component Ablation",
                 fontweight="bold", fontsize=14, color=C_DARK)
    fig.tight_layout()
    save(fig, "fig5_component_ablation", args)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6a-c: Confusion Matrix & Per-Class Analysis
# ═══════════════════════════════════════════════════════════════════════════
def fig_confusion_matrix(data, args):
    cm = np.array(data.get("confusion_matrix", [[0]]))
    cm_norm = np.array(data.get("confusion_matrix_normalized", [[0]]))
    K = cm.shape[0]
    class_names = [CROP_NAMES_SHORT.get(i, f"C{i}") for i in range(K)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 6a: Raw counts
    im1 = ax1.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(K)); ax1.set_yticks(range(K))
    ax1.set_xticklabels(class_names, fontsize=8)
    ax1.set_yticklabels(class_names, fontsize=8)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (Counts)", fontweight="bold")
    for i in range(K):
        for j in range(K):
            ax1.text(j, i, str(int(cm[i,j])), ha="center", va="center",
                     fontsize=7, fontweight="bold",
                     color="white" if cm[i,j] > cm.max()*0.5 else C_DARK)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 6b: Normalized
    im2 = ax2.imshow(cm_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(range(K)); ax2.set_yticks(range(K))
    ax2.set_xticklabels(class_names, fontsize=8)
    ax2.set_yticklabels(class_names, fontsize=8)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix (Normalized by Row)", fontweight="bold")
    for i in range(K):
        for j in range(K):
            ax2.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, fontweight="bold",
                     color="white" if cm_norm[i,j] > 0.5 else C_DARK)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Experiment 6: Confusion Matrix Analysis",
                 fontweight="bold", fontsize=14, color=C_DARK)
    fig.tight_layout()
    save(fig, "fig6a_confusion_matrix", args)


def fig_per_class_analysis(data, args):
    pc = data.get("per_class", {})
    if not pc:
        return

    classes = list(pc.keys())
    K = len(classes)
    x = np.arange(K)
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    prec = [pc[c]["precision"] for c in classes]
    rec = [pc[c]["recall"] for c in classes]
    f1 = [pc[c]["f1"] for c in classes]

    ax.bar(x - w, prec, w, label="Precision", color=PALETTE[0], edgecolor="white")
    ax.bar(x, rec, w, label="Recall (PA)", color=PALETTE[3], edgecolor="white")
    ax.bar(x + w, f1, w, label="F1 Score", color=PALETTE[4], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("\n", " ") for c in classes], fontsize=9, rotation=20)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Per-Class Precision / Recall / F1", fontweight="bold", color=C_DARK)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.3)

    fig.suptitle("Experiment 6: Per-Class Performance",
                 fontweight="bold", fontsize=14, color=C_DARK)
    fig.tight_layout()
    save(fig, "fig6b_per_class_analysis", args)


def fig_per_class_iou(data, args):
    pc = data.get("per_class", {})
    if not pc:
        return

    classes = [c.replace("\n", " ") for c in pc.keys()]
    iou_vals = [pc[c]["iou"] for c in pc.keys()]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[0] if v >= np.mean(iou_vals) else PALETTE[4] for v in iou_vals]
    ax.bar(range(len(classes)), iou_vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU", fontweight="bold", color=C_DARK)
    bar_labels(ax, iou_vals, fmt=".3f")
    ax.axhline(y=np.mean(iou_vals), color="gray", ls="--", alpha=0.5,
               label=f"mIoU={np.mean(iou_vals):.3f}")
    ax.legend(frameon=False)

    fig.suptitle("Experiment 6: Per-Class IoU Distribution",
                 fontweight="bold", fontsize=14, color=C_DARK)
    fig.tight_layout()
    save(fig, "fig6c_per_class_iou", args)


# ═══════════════════════════════════════════════════════════════════════════
# Summary Dashboard (all figures in one)
# ═══════════════════════════════════════════════════════════════════════════
def fig_summary_dashboard(all_data, args):
    """One-page summary with key metrics."""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("FusionCropNet V6 — Comprehensive Experiment Dashboard",
                 fontweight="bold", fontsize=16, color=C_DARK, y=0.98)

    meta = all_data.get("meta", {})
    meta_text = (f"Model: {meta.get('model','?')} | {meta.get('params_M','?')}M params | "
                 f"{meta.get('timestamp','?')} | Device: {meta.get('device','?')}")
    fig.text(0.5, 0.96, meta_text, ha="center", fontsize=9, color="gray")

    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    # 1. Modality ablation mIoU
    ax1 = fig.add_subplot(gs[0, 0])
    exp1 = all_data.get("exp1_modality_ablation", {})
    if exp1:
        configs = list(exp1.keys())[:7]
        short = ["Full","Opt","SAR","DEM","O+S","O+D","S+D"]
        miou = [exp1.get(c,{}).get("mIoU",0) for c in configs[:7]]
        ax1.bar(range(len(miou)), miou, color=PALETTE[:len(miou)], edgecolor="white")
        ax1.set_xticks(range(len(miou)))
        ax1.set_xticklabels(short[:len(miou)], fontsize=7)
        ax1.set_title("1. Modality Ablation (mIoU)", fontweight="bold", fontsize=10)
        for i, v in enumerate(miou):
            ax1.text(i, v+0.002, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # 2. Fusion ablation mIoU
    ax2 = fig.add_subplot(gs[0, 1])
    exp2 = all_data.get("exp2_fusion_ablation", {})
    if exp2:
        configs = list(exp2.keys())[:8]
        short = ["Full","−CrossAtn","−LateFus","−EarlyFus","LateOnly","CrossOnly","EarlyOnly","None"]
        miou = [exp2.get(c,{}).get("mIoU",0) for c in configs[:8]]
        ax2.barh(range(len(miou)), miou, color=PALETTE[:len(miou)], edgecolor="white")
        ax2.set_yticks(range(len(miou)))
        ax2.set_yticklabels(short[:len(miou)], fontsize=7)
        ax2.set_title("2. Fusion Ablation (mIoU)", fontweight="bold", fontsize=10)
        ax2.invert_yaxis()

    # 3. Robustness curves
    ax3 = fig.add_subplot(gs[0, 2])
    exp4 = all_data.get("exp4_robustness", {})
    if "cloud_cover" in exp4:
        cc = exp4["cloud_cover"]
        levels = sorted([float(k.split("_")[1]) for k in cc.keys()])
        miou_c = [cc[f"cloud_{l:.2f}"]["mIoU"] for l in levels]
        ax3.plot(levels, miou_c, "o-", color=PALETTE[0], lw=2, label="Cloud")
    if "missing_timesteps" in exp4:
        mt = exp4["missing_timesteps"]
        n = sorted([int(k.split("_")[1]) for k in mt.keys()])
        miou_t = [mt[f"missing_{x}"]["mIoU"] for x in n]
        ax3_t = ax3.twinx()
        ax3_t.plot(n, miou_t, "s-", color=PALETTE[3], lw=2, label="Timesteps")
        ax3_t.set_ylabel("mIoU (timesteps)", color=PALETTE[3])
    ax3.set_title("3. Robustness", fontweight="bold", fontsize=10)
    ax3.set_xlabel("Cloud fraction")
    ax3.legend(loc="upper right", fontsize=7)

    # 4. Component LOO impact
    ax4 = fig.add_subplot(gs[1, 0])
    exp5 = all_data.get("exp5_component_ablation", {})
    loo = [(k, v) for k, v in exp5.items() if k.startswith("no_")]
    if loo and "V6 Full (all blocks)" in exp5:
        bl = exp5["V6 Full (all blocks)"]["mIoU"]
        labels = [k.replace("no_","") for k, _ in loo]
        deltas = [bl - v["mIoU"] for _, v in loo]
        idx = np.argsort(deltas)[::-1]
        labels = [labels[i] for i in idx]
        deltas = [deltas[i] for i in idx]
        colors = [PALETTE[0] if d>0 else PALETTE[3] for d in deltas]
        ax4.barh(range(len(labels)), deltas, color=colors, edgecolor="white")
        ax4.set_yticks(range(len(labels)))
        ax4.set_yticklabels(labels, fontsize=7)
        ax4.axvline(x=0, color="gray", ls="-", alpha=0.5)
        ax4.set_title("4. Block Impact (ΔmIoU)", fontweight="bold", fontsize=10)

    # 5. Cumulative
    ax5 = fig.add_subplot(gs[1, 1])
    cumul = [(k, v) for k, v in exp5.items() if k.startswith("+") or "V5EDL" in k]
    if cumul:
        labels = [k.replace("+","") for k, _ in cumul]
        miou = [v["mIoU"] for _, v in cumul]
        ax5.plot(range(len(labels)), miou, "o-", color=PALETTE[0], lw=2, markersize=8)
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
        ax5.set_title("5. Cumulative Addition", fontweight="bold", fontsize=10)

    # 6. Confusion matrix (normalized, mini)
    ax6 = fig.add_subplot(gs[1, 2])
    exp6 = all_data.get("exp6_confusion_analysis", {})
    cm_norm = np.array(exp6.get("confusion_matrix_normalized", [[0]]))
    if cm_norm.size > 1:
        K = cm_norm.shape[0]
        class_names = [CROP_NAMES_SHORT.get(i,f"C{i}") for i in range(K)]
        im = ax6.imshow(cm_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax6.set_xticks(range(K)); ax6.set_yticks(range(K))
        ax6.set_xticklabels(class_names, fontsize=6)
        ax6.set_yticklabels(class_names, fontsize=6)
        ax6.set_title("6. Confusion Matrix (norm)", fontweight="bold", fontsize=10)
        for i in range(K):
            for j in range(K):
                ax6.text(j, i, f"{cm_norm[i,j]:.1f}", ha="center", va="center",
                         fontsize=6, color="white" if cm_norm[i,j]>0.5 else C_DARK)

    # 7. Per-class IoU
    ax7 = fig.add_subplot(gs[2, :2])
    pc = exp6.get("per_class", {})
    if pc:
        classes = [c.replace("\n"," ") for c in pc.keys()]
        iou = [pc[c]["iou"] for c in pc.keys()]
        ax7.bar(range(len(classes)), iou, color=PALETTE[:len(classes)], edgecolor="white")
        ax7.set_xticks(range(len(classes)))
        ax7.set_xticklabels(classes, fontsize=8)
        ax7.set_title("7. Per-Class IoU", fontweight="bold", fontsize=10)
        for i, v in enumerate(iou):
            ax7.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    # 8. Global metrics table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    gl = exp6.get("global", {})
    cal = exp6.get("calibration", {})
    lines = ["Global Metrics:", "─"*25]
    for k in ["OA","mIoU","Kappa"]:
        if k in gl:
            lines.append(f"  {k}: {gl[k]:.4f}")
    if "ECE" in cal:
        lines.append(f"  ECE: {cal['ECE']:.4f}")
        lines.append(f"  NLL: {cal.get('NLL',0):.4f}")
        lines.append(f"  AUROC: {cal.get('AUROC',0):.3f}")
    ax8.text(0, 1, "\n".join(lines), transform=ax8.transAxes,
             fontsize=9, fontfamily="monospace", va="top", color=C_DARK)

    save(fig, "fig0_summary_dashboard", args)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    data = load_results(args.results)
    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("  V6 Experiment Visualization Dashboard")
    print("="*60)

    if "exp1_modality_ablation" in data:
        fig_modality_heatmap(data["exp1_modality_ablation"], args)
    if "exp2_fusion_ablation" in data:
        fig_fusion_ablation(data["exp2_fusion_ablation"], args)
    if "exp3_feature_ablation" in data:
        fig_feature_ablation(data["exp3_feature_ablation"], args)
    if "exp4_robustness" in data:
        fig_robustness(data["exp4_robustness"], args)
    if "exp5_component_ablation" in data:
        fig_component_ablation(data["exp5_component_ablation"], args)
    if "exp6_confusion_analysis" in data:
        exp6 = data["exp6_confusion_analysis"]
        fig_confusion_matrix(exp6, args)
        fig_per_class_analysis(exp6, args)
        fig_per_class_iou(exp6, args)

    # Summary dashboard always last
    fig_summary_dashboard(data, args)
    print(f"\nAll figures saved to: {args.output}/")


if __name__ == "__main__":
    main()
