"""
光学影像处理器 - GeoTIFF读取、云掩膜、光谱指数、归一化
"""
import os
import json
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
from .base import BasePreprocessor

LS_SCALE = 0.0000275
LS_OFFSET = -0.2
TARGET_CRS = "EPSG:32650"
TARGET_RES = 30

BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
BAND_IDX = [1, 2, 3, 4, 5, 6]


class GeoTIFFReader:
    """GeoTIFF 读取与重投影"""

    @staticmethod
    def parse_date_from_filename(filename: str) -> datetime:
        stem = Path(filename).stem
        for part in stem.split("_"):
            if len(part) == 8 and part.isdigit():
                return datetime.strptime(part, "%Y%m%d")
            if len(part) == 6 and part.isdigit():
                return datetime.strptime(part + "15", "%Y%m%d")
        return datetime(2023, 1, 1)

    @staticmethod
    def read_and_reproject(filepath: str, target_crs: str = TARGET_CRS,
                           target_res: float = TARGET_RES) -> Tuple[np.ndarray, dict]:
        with rasterio.open(filepath) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height,
                *src.bounds, resolution=target_res
            )
            profile = src.profile.copy()
            profile.update({"crs": target_crs, "transform": transform,
                            "width": width, "height": height})

            data = np.zeros((src.count, height, width), dtype=np.float32)
            for i in range(src.count):
                reproject(
                    source=rasterio.band(src, i + 1),
                    destination=data[i],
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )
        return data, profile


class CloudMaskProcessor:
    """云/云影掩膜处理"""

    @staticmethod
    def get_cloud_coverage(qa_band: np.ndarray) -> float:
        cloud = (qa_band & 0x8000).astype(bool)
        shadow = (qa_band & (1 << 3)).astype(bool)
        fill = (qa_band & (1 << 1)).astype(bool)
        invalid = cloud | shadow | fill
        return invalid.mean()

    @staticmethod
    def create_mask(qa_band: np.ndarray, include_shadow: bool = True,
                    include_snow: bool = True) -> np.ndarray:
        invalid = (qa_band & (1 << 1)).astype(bool)
        cloud = (qa_band & ((1 << 5) | (1 << 6))).astype(bool)
        shadow = (qa_band & (1 << 3)).astype(bool) if include_shadow else False
        snow = (qa_band & (1 << 4)).astype(bool) if include_snow else False
        return ~(invalid | cloud | shadow | snow)

    @staticmethod
    def morphological_expand(qa_band: np.ndarray, expand_pixels: int = 3) -> np.ndarray:
        from scipy.ndimage import binary_dilation
        cloud_mask = ~CloudMaskProcessor.create_mask(qa_band)
        expanded = binary_dilation(cloud_mask, iterations=expand_pixels)
        return ~expanded

    @staticmethod
    def apply_mask(bands: np.ndarray, qa_band: np.ndarray,
                   fill_value: float = np.nan) -> np.ndarray:
        valid_mask = CloudMaskProcessor.create_mask(qa_band)
        result = bands.copy().astype(np.float32)
        result[:, ~valid_mask] = fill_value
        return result


def compute_spectral_indices(bands: np.ndarray) -> np.ndarray:
    """计算植被指数（NDVI, EVI, LSWI, NDWI, SAVI, NDRE）"""
    eps = 1e-6
    blue, green, red, nir, swir1, swir2 = [bands[:, i] for i in range(6)]

    ndvi = (nir - red) / (nir + red + eps)
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1.0 + eps)
    lswi = (nir - swir1) / (nir + swir1 + eps)
    ndwi = (green - nir) / (green + nir + eps)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    ndre = (nir - red) / (nir + red + eps)

    indices = np.stack([ndvi, evi, lswi, ndwi, savi, ndre], axis=1)
    return np.concatenate([bands, indices], axis=1)


class DataNormalizer:
    """多方法数据归一化"""

    def __init__(self, method: str = "robust"):
        self.method = method
        self.stats: dict = {}

    def fit(self, sequence: np.ndarray, channel_names: Optional[List[str]] = None):
        T, C, H, W = sequence.shape
        flat = sequence.reshape(T * H * W, C)
        valid = flat[~np.isnan(flat).any(axis=1)]

        for c in range(C):
            ch_data = valid[:, c]
            name = channel_names[c] if channel_names else f"ch_{c}"

            if self.method == "minmax":
                self.stats[name] = {"min": float(ch_data.min()), "max": float(ch_data.max())}
            elif self.method == "zscore":
                self.stats[name] = {"mean": float(ch_data.mean()),
                                    "std": float(ch_data.std() + 1e-6)}
            elif self.method == "robust":
                self.stats[name] = {
                    "median": float(np.median(ch_data)),
                    "iqr": float(np.percentile(ch_data, 75) -
                                 np.percentile(ch_data, 25) + 1e-6)
                }
            elif self.method == "percentile":
                self.stats[name] = {"p2": float(np.percentile(ch_data, 2)),
                                    "p98": float(np.percentile(ch_data, 98))}

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        result = sequence.copy().astype(np.float32)
        names = list(self.stats.keys())
        for c, name in enumerate(names):
            s = self.stats[name]
            if self.method == "minmax":
                result[:, c] = (result[:, c] - s["min"]) / (s["max"] - s["min"] + 1e-6)
            elif self.method == "zscore":
                result[:, c] = (result[:, c] - s["mean"]) / s["std"]
            elif self.method == "robust":
                result[:, c] = (result[:, c] - s["median"]) / s["iqr"]
            elif self.method == "percentile":
                result[:, c] = (result[:, c] - s["p2"]) / (s["p98"] - s["p2"] + 1e-6)
                result[:, c] = np.clip(result[:, c], 0, 1)
        return result

    def fit_transform(self, sequence: np.ndarray,
                      channel_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(sequence, channel_names)
        return self.transform(sequence)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"method": self.method, "stats": self.stats}, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.method = data["method"]
        self.stats = data["stats"]
        return self


class OpticalProcessor(BasePreprocessor):
    """光学影像处理器"""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

    def process(self, data: dict) -> dict:
        self.validate_input(data)
        if 'qa_pixel' in data:
            data['mask'] = self.apply_cloud_mask(data['qa_pixel'])
        if 'image' in data:
            nir = data['image'][..., 7] if data['image'].ndim >= 3 and data['image'].shape[-1] > 7 \
                else data['image'][..., 3]
            red = data['image'][..., 3] if data['image'].ndim >= 3 and data['image'].shape[-1] > 3 \
                else data['image'][..., 2]
            blue = data['image'][..., 0]
            data['ndvi'] = self.calculate_ndvi(nir, red)
            data['evi'] = self.calculate_evi(nir, red, blue)
        return data

    def calculate_ndvi(self, nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
        ndvi = np.nan_to_num(ndvi, nan=0.0)
        return np.clip(ndvi, -1, 1)

    def calculate_evi(self, nir: np.ndarray, red: np.ndarray,
                      blue: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
        evi = np.nan_to_num(evi, nan=0.0)
        return evi

    def apply_cloud_mask(self, qa_pixel: np.ndarray) -> np.ndarray:
        cloud_mask = (qa_pixel & (1 << 10)) == 0
        shadow_mask = (qa_pixel & (1 << 11)) == 0
        return cloud_mask & shadow_mask

    def normalize(self, data: np.ndarray, min_val: float = 0.0,
                  max_val: float = 1.0) -> np.ndarray:
        data_min, data_max = np.min(data), np.max(data)
        if data_max == data_min:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
