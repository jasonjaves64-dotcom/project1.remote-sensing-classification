# DEM Ablation Experiment Visualization
"""
Visualization script for DEM ablation experiment results.
Reads ablation_output/dem_ablation_results.json and produces a 3-panel
academic-style figure: mIoU bar chart, delta horizontal-bar chart, summary table.

Usage:
    python scripts/visualize_v6_experiments.py
    python scripts/visualize_v6_experiments.py -i ablation_output/dem_ablation_results.json
    python scripts/visualize_v6_experiments.py -o ablation_output/dem_ablation_visualization.png
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Chinese-capable font
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                                    "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_results(path: str) -> dict:
    """Load ablation results JSON and validate structure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    exps = data.get("experiments")
    if not isinstance(exps, list) or len(exps) < 2:
        raise ValueError("JSON must contain an 'experiments' list (>= 2 entries).")
    for i, exp in enumerate(exps):
        if "name" not in exp or "mIoU" not in exp:
            raise KeyError(f"Experiment[{i}] missing 'name' or 'mIoU'.")
        exp["mIoU"] = float(exp["mIoU"])
    return data


def _assign_colors(exps: list[dict]) -> list[str]:
    """Colour each bar: baseline=blue, mIoU>=baseline=green, mIoU<baseline=red."""
    bl = exps[0]["mIoU"]
    colors = ["#2196F3"]  # baseline
    for e in exps[1:]:
        colors.append("#4CAF50" if e["mIoU"] >= bl else "#F44336")
    return colors


def plot_experiments(exps: list[dict], save_path: str, title: str = "DEM 消融实验可视化"):
    """Create the 3-panel figure and save to disk."""

    names = [e["name"] for e in exps]
    mious = np.array([e["mIoU"] for e in exps])
    bl = mious[0]
    deltas = bl - mious                     # positive → degradation，negative → improvement
    colors = _assign_colors(exps)

    # Delta bar colours: red (degradation), green (improvement), grey (baseline)
    delta_colors = ["#F44336" if d > 0 else "#4CAF50" for d in deltas]
    delta_colors[0] = "#9E9E9E"

    y = np.arange(len(exps))

    # ── Figure & style ────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 10))

    # ---- Panel 1: mIoU bar chart ----
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(y, mious, color=colors, edgecolor="white", linewidth=0.8, alpha=0.90)
    ax1.set_xticks(y)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("mIoU", fontsize=11)
    ax1.set_title("各实验配置 mIoU 对比", fontsize=13, fontweight="bold", pad=12)
    for b, v in zip(bars, mious):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(y=bl, color="#2196F3", linestyle="--", linewidth=1.2, alpha=0.7,
                label=f"Baseline mIoU = {bl:.4f}")
    ax1.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white",
               edgecolor="#ddd")
    ax1.set_ylim(0, max(mious) * 1.12)

    # ---- Panel 2: Delta horizontal bar chart ----
    ax2 = fig.add_subplot(2, 2, 2)
    barh = ax2.barh(y, deltas, color=delta_colors, edgecolor="white",
                    linewidth=0.8, alpha=0.90, height=0.55)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.axvline(x=0, color="black", linewidth=1.0)
    ax2.set_xlabel(r"$\Delta$ mIoU  (Baseline − 实验)", fontsize=11)
    ax2.set_title("相对基线 mIoU 变化量", fontsize=13, fontweight="bold", pad=12)
    ax2.invert_yaxis()
    for bar, d in zip(barh, deltas):
        x_pos = bar.get_width()
        offset = 0.002 if x_pos >= 0 else -0.002
        ha = "left" if x_pos >= 0 else "right"
        color = "#333" if abs(d) < 1e-6 else ("#C62828" if d > 0 else "#2E7D32")
        ax2.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                 f"{d:+.4f}", va="center", ha=ha, fontsize=8, color=color)
    legend_patches = [
        Patch(facecolor="#F44336", alpha=0.85, label="性能下降 (Degradation)"),
        Patch(facecolor="#4CAF50", alpha=0.85, label="性能提升 (Improvement)"),
        Patch(facecolor="#9E9E9E", alpha=0.85, label="基线 (Baseline)"),
    ]
    ax2.legend(handles=legend_patches, loc="lower right", fontsize=8,
               frameon=True, facecolor="white", edgecolor="#ddd")

    # ---- Panel 3: Summary table ----
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis("off")

    cell_text = []
    for i, e in enumerate(exps):
        d = deltas[i]
        if i == 0:
            cat = "Baseline"
        elif d > 0:
            cat = "Degradation"
        else:
            cat = "Improvement"
        cell_text.append([e["name"], f"{e['mIoU']:.4f}", f"{d:+.4f}", cat])

    col_labels = ["实验名称", "mIoU", "Δ mIoU", "类别"]
    table = ax3.table(cellText=cell_text, colLabels=col_labels,
                      colColours=["#ECEFF1"] * 4, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    # Colour-code category column
    for i in range(len(exps)):
        cell = table[(i + 1, 3)]
        txt = cell.get_text().get_text()
        face = {"Baseline": "#BBDEFB", "Degradation": "#FFCDD2", "Improvement": "#C8E6C9"}
        cell.set_facecolor(face.get(txt, "white"))

    # Header row style
    for j in range(4):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].get_text().set_color("white")
        table[(0, j)].set_fontsize(10)

    ax3.set_title("消融实验汇总表", fontsize=13, fontweight="bold", pad=20, loc="center")

    # ── Save ──────────────────────────────────────────────────────────
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DEM Ablation Experiment Visualization")
    parser.add_argument("-i", "--input", default="ablation_output/dem_ablation_results.json",
                        help="Path to ablation results JSON")
    parser.add_argument("-o", "--output", default="ablation_output/dem_ablation_visualization.png",
                        help="Output image path")
    parser.add_argument("-t", "--title", default="DEM 消融实验可视化",
                        help="Figure super-title")
    args = parser.parse_args()

    try:
        data = _load_results(args.input)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    plot_experiments(data["experiments"], args.output, title=args.title)


if __name__ == "__main__":
    main()
