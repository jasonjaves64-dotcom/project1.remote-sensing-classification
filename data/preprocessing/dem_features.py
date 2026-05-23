"""
DEM 特征提取模块
"""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


class DEMFeatureExtractor:
    """DEM 地形特征提取（海拔、坡度、坡向、曲率、粗糙度）"""

    def __init__(self, dem_path: str):
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype(np.float32)
            self.res = src.res[0]
            self.profile = src.profile

    def extract_features(self, target_profile: dict) -> np.ndarray:
        H, W = target_profile["height"], target_profile["width"]
        if self.dem.shape != (H, W):
            dem_reproj = np.zeros((H, W), dtype=np.float32)
            reproject(
                source=self.dem, destination=dem_reproj,
                src_transform=self.profile["transform"],
                src_crs=self.profile["crs"],
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
                resampling=Resampling.bilinear
            )
        else:
            dem_reproj = self.dem

        dy, dx = np.gradient(dem_reproj, self.res)
        slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        slope_deg = np.degrees(slope_rad)
        aspect_rad = np.arctan2(-dx, dy) % (2 * np.pi)
        aspect_deg = np.degrees(aspect_rad)
        curvature = np.gradient(dx, self.res, axis=1) + np.gradient(dy, self.res, axis=0)
        roughness = np.sqrt(dx ** 2 + dy ** 2)

        dem_norm = (dem_reproj - dem_reproj.min()) / (dem_reproj.max() - dem_reproj.min() + 1e-6)
        slope_norm = slope_deg / 90.0
        aspect_norm = aspect_deg / 360.0
        curvature_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-6)
        roughness_norm = roughness / (roughness.max() + 1e-6)

        return np.stack([dem_norm, slope_norm, aspect_norm, curvature_norm, roughness_norm], axis=0)
