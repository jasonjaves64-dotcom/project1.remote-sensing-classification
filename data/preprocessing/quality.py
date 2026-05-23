"""
质量控制模块 - 数据有效性检查、统计报告、可视化诊断
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, List

CROP_CLASSES = {0: "背景", 1: "冬小麦", 2: "夏玉米", 3: "水稻", 4: "大豆", 5: "棉花", 6: "其他"}


class QualityControl:
    """数据质量控制"""

    def __init__(self):
        pass

    def check_validity(self, data: np.ndarray) -> Dict[str, Any]:
        valid_pixels = int(np.sum(np.isfinite(data)))
        invalid_pixels = int(np.sum(~np.isfinite(data)))
        total_pixels = int(data.size)
        return {
            "valid_pixels": valid_pixels,
            "invalid_pixels": invalid_pixels,
            "valid_percentage": 100 * valid_pixels / total_pixels if total_pixels > 0 else 0,
            "total_pixels": total_pixels
        }

    def generate_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "percentiles": {}}
        return {
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "median": float(np.median(valid_data)),
            "percentiles": {
                "p5": float(np.percentile(valid_data, 5)),
                "p25": float(np.percentile(valid_data, 25)),
                "p50": float(np.percentile(valid_data, 50)),
                "p75": float(np.percentile(valid_data, 75)),
                "p95": float(np.percentile(valid_data, 95))
            }
        }

    def detect_outliers(self, data: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return np.zeros_like(data, dtype=bool)
        mean, std = np.mean(valid_data), np.std(valid_data)
        if std == 0:
            return np.zeros_like(data, dtype=bool)
        return np.abs((data - mean) / std) > z_threshold

    def generate_report(self, data: np.ndarray,
                        dataset_name: str = "unknown") -> Dict[str, Any]:
        return {
            "dataset_name": dataset_name,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "validity": self.check_validity(data),
            "statistics": self.generate_statistics(data),
            "outlier_count": int(np.sum(self.detect_outliers(data)))
        }


class PreprocessingQualityReport:
    """预处理质量报告与可视化"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_cloud_coverage_timeline(self, cloud_pcts: List[float],
                                     doys: List[int], year: int = 2023):
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#e74c3c" if p > 0.3 else "#2ecc71" for p in cloud_pcts]
        ax.bar(range(len(doys)), [p * 100 for p in cloud_pcts],
               color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(30, color="red", linestyle="--", linewidth=1, label="30%")
        ax.set_title(f"{year} 云量统计")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cloud_coverage_timeline.png", dpi=120)
        plt.close()

    def plot_phenology_curves(self, opt_sequence: np.ndarray, label_map: np.ndarray,
                              doys: np.ndarray, ndvi_channel: int = 6):
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#FFD700", "#228B22", "#4682B4", "#9ACD32", "#FF8C00", "#A9A9A9"]
        for cls_id, cls_name in CROP_CLASSES.items():
            if cls_id == 0: continue
            mask = label_map == cls_id
            if mask.sum() < 10: continue
            ndvi = opt_sequence[:, ndvi_channel, :, :] if opt_sequence.shape[1] > ndvi_channel \
                else opt_sequence[:, 0, :, :]
            mean = np.nanmean(ndvi[:, mask], axis=1)
            ax.plot(doys, mean, marker="o", markersize=4,
                    label=cls_name, color=colors[(cls_id - 1) % len(colors)])
        ax.set_xlabel("DOY"); ax.set_ylabel("NDVI"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/phenology_curves.png", dpi=120)
        plt.close()

    def plot_missing_value_heatmap(self, opt_sequence: np.ndarray):
        T = opt_sequence.shape[0]
        cols = min(6, T); rows = (T + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        axes = np.array(axes).flatten()
        for t in range(T):
            axes[t].imshow(np.isnan(opt_sequence[t, 0]).astype(float), cmap="Reds", vmin=0, vmax=1)
            axes[t].set_title(f"T={t + 1}", fontsize=8); axes[t].axis("off")
        for t in range(T, len(axes)): axes[t].axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/missing_value_heatmap.png", dpi=100)
        plt.close()

    def plot_normalization_comparison(self, raw_seq: np.ndarray, norm_seq: np.ndarray,
                                      channel_names: Optional[List[str]] = None):
        show_channels = min(4, raw_seq.shape[1] if raw_seq.ndim == 4 else 1)
        fig, axes = plt.subplots(2, show_channels, figsize=(show_channels * 3.5, 6))
        if show_channels == 1: axes = axes.reshape(2, 1)
        for i in range(show_channels):
            name = channel_names[i] if channel_names else f"CH_{i}"
            raw_vals = raw_seq[:, i].flatten() if raw_seq.ndim == 4 else raw_seq.flatten()
            norm_vals = norm_seq[:, i].flatten() if norm_seq.ndim == 4 else norm_seq.flatten()
            raw_vals = raw_vals[~np.isnan(raw_vals)]
            norm_vals = norm_vals[~np.isnan(norm_vals)]
            axes[0, i].hist(raw_vals, bins=50, color="#3498db", alpha=0.7)
            axes[0, i].set_title(f"{name} 归一化前", fontsize=9)
            axes[1, i].hist(norm_vals, bins=50, color="#e74c3c", alpha=0.7)
            axes[1, i].set_title(f"{name} 归一化后", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/normalization_comparison.png", dpi=120)
        plt.close()

    def plot_spatial_split(self, train_mask: np.ndarray, val_mask: np.ndarray,
                           test_mask: np.ndarray):
        split_map = np.zeros(train_mask.shape, dtype=np.uint8)
        split_map[train_mask], split_map[val_mask], split_map[test_mask] = 1, 2, 3
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(split_map, cmap=plt.cm.get_cmap("Set1", 4), vmin=0, vmax=3)
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(["背景", "训练", "验证", "测试"])
        ax.set_title("空间划分"); ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/spatial_split.png", dpi=120)
        plt.close()

    def plot_class_distribution(self, label_map: np.ndarray):
        counts = {}
        for cls_id, cls_name in CROP_CLASSES.items():
            if cls_id == 0: continue
            cnt = int((label_map == cls_id).sum())
            if cnt > 0: counts[cls_name] = cnt
        fig, ax = plt.subplots(figsize=(7, 7))
        wedge_colors = ["#FFD700", "#228B22", "#4682B4", "#9ACD32", "#FF8C00", "#A9A9A9"]
        labels, values = list(counts.keys()), list(counts.values())
        ax.pie(values, labels=labels, colors=wedge_colors[:len(labels)],
               autopct="%1.1f%%", startangle=90)
        ax.set_title("类别分布")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/class_distribution.png", dpi=120)
        plt.close()

    def generate_text_report(self, stats: dict):
        report_path = f"{self.output_dir}/preprocessing_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        txt_path = f"{self.output_dir}/preprocessing_report.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("农作物遥感分类 - 数据预处理质量报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for section_name, section_data in stats.items():
                f.write(f"[{section_name}]\n")
                if isinstance(section_data, dict):
                    for k, v in section_data.items():
                        f.write(f"  {k}: {v}\n")
                f.write("\n")
