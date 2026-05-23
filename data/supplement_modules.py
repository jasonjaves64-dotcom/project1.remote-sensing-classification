# =============================================================================
# data/supplement_modules.py
# 数据预处理补充模块：
# Module A : 地块边界混合像元精细处理 + 标签空间对齐质量评估
# Module C : 像元级有效观测追踪（精细云掩膜）
# =============================================================================
import os
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from pathlib import Path
from scipy.ndimage import binary_dilation, label as nd_label
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# MODULE A：地块边界混合像元精细处理
# =============================================================================
class MixedPixelHandler:
    def __init__(self, pixel_size_m: float = 30.0, purity_threshold: float = 0.8):
        self.pixel_size = pixel_size_m
        self.purity_thresh = purity_threshold
        self.pixel_area = pixel_size_m ** 2

    def compute_pixel_purity(self, gdf: gpd.GeoDataFrame, profile: dict, 
                             class_field: str = "crop_type") -> tuple[np.ndarray, np.ndarray]:
        from shapely.geometry import box as sg_box
        H, W = profile["height"], profile["width"]
        T, crs = profile["transform"], profile["crs"]
        gdf_proj = gdf.to_crs(crs)
        
        class_area = {}
        for cls in gdf_proj[class_field].unique():
            class_area[int(cls)] = np.zeros((H, W), dtype=np.float32)

        for _, row in gdf_proj.iterrows():
            geom, cls_id = row.geometry, int(row[class_field])
            if geom is None or geom.is_empty:
                continue
            
            minx, miny, maxx, maxy = geom.bounds
            col_min, row_max = ~T * (minx, miny)
            col_max, row_min = ~T * (maxx, maxy)
            r0, r1 = max(0, int(row_min) - 1), min(H, int(row_max) + 2)
            c0, c1 = max(0, int(col_min) - 1), min(W, int(col_max) + 2)

            for r in range(r0, r1):
                for c in range(c0, c1):
                    px_left, px_top = T.c + c * T.a, T.f + r * T.e
                    pixel_box = sg_box(px_left, px_top, px_left + T.a, px_top + T.e)
                    intersection = geom.intersection(pixel_box)
                    if not intersection.is_empty:
                        class_area[cls_id][r, c] += min(intersection.area / self.pixel_area, 1.0)

        all_areas = np.stack([class_area[k] for k in sorted(class_area.keys())], axis=0)
        cls_ids = sorted(class_area.keys())
        purity_map = all_areas.max(axis=0)
        dominant_map = np.array(cls_ids)[all_areas.argmax(axis=0)].astype(np.uint8)
        
        dominant_map[all_areas.sum(axis=0) < 0.01] = 0
        purity_map[all_areas.sum(axis=0) < 0.01] = 0.0
        return purity_map, dominant_map

    def detect_multi_class_boundaries(self, label_map: np.ndarray, buffer_px: int = 1) -> np.ndarray:
        boundary_map = np.zeros(label_map.shape, dtype=bool)
        valid_mask = (label_map > 0) & (label_map < 255)
        
        shifts = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for dr, dc in shifts:
            shifted = np.roll(np.roll(label_map, dr, axis=0), dc, axis=1)
            boundary_map |= (label_map != shifted) & valid_mask & (shifted > 0) & (shifted < 255)
        
        if buffer_px > 0:
            boundary_map = binary_dilation(boundary_map, iterations=buffer_px)
        print(f" 跨类别边界像元: {boundary_map.sum():,} ({100*boundary_map.mean():.2f}%)")
        return boundary_map

    def build_final_label(self, dominant_map: np.ndarray, purity_map: np.ndarray,
                          cross_boundary: np.ndarray, qa_mask: np.ndarray) -> np.ndarray:
        final = dominant_map.copy()
        final[purity_map < self.purity_thresh] = 255
        final[cross_boundary] = 255
        final[qa_mask == 255] = 255
        
        n_valid = ((final > 0) & (final < 255)).sum()
        n_ignored = (final == 255).sum()
        print(f"最终标签: 有效{n_valid:,} 忽略{n_ignored:,} ({100*n_ignored/final.size:.1f}%)")
        return final

    def visualize_purity(self, purity_map, cross_boundary, final_label, save_path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(purity_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[0].set_title("像元纯度图")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        axes[1].imshow(cross_boundary.astype(np.uint8), cmap="Reds", vmin=0, vmax=1)
        axes[1].set_title("跨类别边界")
        axes[1].axis("off")
        
        axes[2].imshow(final_label, cmap=plt.cm.get_cmap("tab10", 8), vmin=0, vmax=7)
        axes[2].set_title("最终标签")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f" ✓ 混合像元可视化: {save_path}")

# =============================================================================
# MODULE C：像元级有效观测追踪
# =============================================================================
class PixelLevelCloudTracker:
    QA_BITS = {
        "fill": 1 << 0, "dilated_cloud": 1 << 1, "cirrus": 1 << 2,
        "cloud": 1 << 3, "cloud_shadow": 1 << 4, "snow": 1 << 5,
        "clear": 1 << 6, "water": 1 << 7,
    }

    def __init__(self, min_valid_obs: int = 4):
        self.min_valid_obs = min_valid_obs

    def parse_qa_pixel(self, qa_band: np.ndarray) -> dict:
        masks = {name: (qa_band & bit).astype(bool) for name, bit in self.QA_BITS.items()}
        masks["valid"] = ~(masks["fill"] | masks["cloud"] | masks["cloud_shadow"] | 
                          masks["snow"] | masks["dilated_cloud"] | masks["cirrus"])
        return masks

    def build_observation_quality_matrix(self, qa_paths: list, profile: dict):
        T, H, W = len(qa_paths), profile["height"], profile["width"]
        obs_matrix = np.zeros((T, H, W), dtype=np.uint8)
        
        for t, qa_path in enumerate(qa_paths):
            with rasterio.open(qa_path) as src:
                masks = self.parse_qa_pixel(src.read(1).astype(np.uint16))
                obs_matrix[t][masks["valid"]] = 0
                obs_matrix[t][masks["cloud"]] = 1
                obs_matrix[t][masks["cloud_shadow"]] = 2
                obs_matrix[t][masks["snow"]] = 3
                obs_matrix[t][masks["fill"]] = 4
        
        valid_count = (obs_matrix == 0).sum(axis=0).astype(np.uint8)
        quality_map = self._compute_quality_score(obs_matrix)
        return obs_matrix, valid_count, quality_map

    def _compute_quality_score(self, obs_matrix):
        T = obs_matrix.shape[0]
        valid_rate = (obs_matrix == 0).mean(axis=0)
        
        quarter_size = max(T // 4, 1)
        quarter_has_obs = np.zeros((4, obs_matrix.shape[1], obs_matrix.shape[2]), dtype=bool)
        for q in range(4):
            t_start, t_end = q * quarter_size, (q + 1) * quarter_size if q < 3 else T
            quarter_has_obs[q] = (obs_matrix[t_start:t_end] == 0).max(axis=0) > 0
        
        return (valid_rate * quarter_has_obs.mean(axis=0)).astype(np.float32)

# =============================================================================
# 整合入口
# =============================================================================
def run_supplement_modules(opt_sequence: np.ndarray, sar_sequence: np.ndarray,
                           doy_norm: np.ndarray, label_map: np.ndarray,
                           gdf_path: str, profile: dict, output_dir: str,
                           class_field: str = "crop_type") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(gdf_path):
        print(" ⚠ 未找到矢量文件，跳过混合像元处理")
        return {"final_label": label_map}

    print("\n" + "="*60)
    print("Module A：混合像元精细处理")
    print("="*60)
    handler = MixedPixelHandler(pixel_size_m=30.0, purity_threshold=0.8)
    print(" 计算亚像元面积占比...")
    purity_map, dominant_map = handler.compute_pixel_purity(gdf_path=gpd.read_file(gdf_path), 
                                                           profile=profile, 
                                                           class_field=class_field)
    
    print(" 检测跨类别边界...")
    cross_boundary = handler.detect_multi_class_boundaries(dominant_map, buffer_px=1)
    
    final_label = handler.build_final_label(dominant_map, purity_map, cross_boundary, np.zeros_like(label_map))
    handler.visualize_purity(purity_map, cross_boundary, final_label, 
                             save_path=f"{output_dir}/mixed_pixel_analysis.png")

    print("\n" + "="*60)
    print("Module B：空间K折验证划分（已移至utils/evaluation.py）")
    print("="*60)
    from utils.evaluation import ValidationStrategy
    validator = ValidationStrategy()
    print(" 生成空间5折交叉验证划分...")
    splits = validator.spatial_kfold_split(final_label, k=5, block_size_px=64)

    np.save(f"{output_dir}/final_label.npy", final_label.astype(np.uint8))
    np.save(f"{output_dir}/purity_map.npy", purity_map.astype(np.float32))
    
    print(f"\n 补充模块完成！输出目录: {output_dir}")
    return {
        "final_label": final_label,
        "purity_map": purity_map,
        "kfold_splits": splits,
        "validator": validator
    }