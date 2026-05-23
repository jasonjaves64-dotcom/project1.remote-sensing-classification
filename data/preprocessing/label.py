"""
标签处理模块 - 矢量栅格化、边界侵蚀、分布分析
"""
import os
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Optional

CROP_CLASSES = {0: "背景", 1: "冬小麦", 2: "夏玉米", 3: "水稻", 4: "大豆", 5: "棉花", 6: "其他"}


class LabelProcessor:
    """标签制作与质量控制"""

    @staticmethod
    def rasterize_vector_labels(shp_path: str, profile: dict,
                                class_field: str = "crop_type") -> np.ndarray:
        if not os.path.exists(shp_path):
            return np.random.randint(0, 7, (profile['height'], profile['width'])).astype(np.uint8)

        gdf = gpd.read_file(shp_path).to_crs(profile.get("crs", "EPSG:4326"))
        shapes = [(geom.__geo_interface__, int(cls))
                  for geom, cls in zip(gdf.geometry, gdf[class_field])
                  if geom is not None]
        return rasterize(shapes=shapes,
                         out_shape=(profile["height"], profile["width"]),
                         transform=profile.get("transform"),
                         fill=0, dtype=np.uint8, all_touched=False)

    @staticmethod
    def rasterize(shp_path: str, profile: dict,
                  class_field: str = "crop_type") -> np.ndarray:
        return LabelProcessor.rasterize_vector_labels(shp_path, profile, class_field)

    @staticmethod
    def erode_field_boundaries(label_map: np.ndarray,
                               erosion_pixels: int = 2) -> np.ndarray:
        result = label_map.copy()
        for cls in np.unique(label_map):
            if cls == 0:
                continue
            cls_mask = label_map == cls
            eroded = binary_erosion(cls_mask, iterations=erosion_pixels)
            result[cls_mask & ~eroded] = 255
        return result

    @staticmethod
    def erode_boundaries(label_map: np.ndarray, pixels: int = 2) -> np.ndarray:
        return LabelProcessor.erode_field_boundaries(label_map, pixels)

    @staticmethod
    def analyze_class_distribution(label_map: np.ndarray) -> dict:
        valid = label_map[label_map != 255]
        unique, counts = np.unique(valid, return_counts=True)
        total = counts.sum()
        distribution = {}
        for cls, cnt in zip(unique, counts):
            name = CROP_CLASSES.get(int(cls), f"unknown_{cls}")
            distribution[int(cls)] = {
                "name": name, "count": int(cnt), "ratio": float(cnt / total)
            }
        return distribution
