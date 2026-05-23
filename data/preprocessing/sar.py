"""
SAR 影像处理器 - 斑噪滤波、空间配准、特征计算
"""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter, median_filter
from typing import Optional, Tuple
from .base import BasePreprocessor


class SARProcessor(BasePreprocessor):
    """SAR 影像处理器"""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

    def process(self, data: dict) -> dict:
        self.validate_input(data)
        if 'image' in data:
            speckle_method = self.config.get('sar_speckle_method', 'refined_lee')
            data['image'] = self.apply_speckle_filter(data['image'], method=speckle_method)
            if self.config.get('sar_log_transform', True):
                data['image'] = self.normalize_db(data['image'])
        return data

    def apply_speckle_filter(self, data: np.ndarray, window_size: int = 5,
                             method: str = "refined_lee") -> np.ndarray:
        if method == "boxcar":
            return SAROpticalAligner._boxcar_filter(data, window_size)
        elif method == "median":
            return SAROpticalAligner._median_filter(data, window_size)
        elif method == "refined_lee":
            return SAROpticalAligner.refined_lee_filter(data, window_size)
        return data

    def orthorectify(self, data: np.ndarray) -> np.ndarray:
        return data.copy()

    def normalize_db(self, data: np.ndarray) -> np.ndarray:
        data_db = 10 * np.log10(data + 1e-10)
        min_db = np.percentile(data_db, 1)
        max_db = np.percentile(data_db, 99)
        return np.clip((data_db - min_db) / (max_db - min_db + 1e-6), 0, 1)


class SAROpticalAligner:
    """SAR 与光学影像空间配准"""

    @staticmethod
    def align_sar_to_optical(sar_array: np.ndarray, sar_profile: dict,
                             opt_profile: dict) -> np.ndarray:
        H_opt, W_opt = opt_profile["height"], opt_profile["width"]
        C_sar = sar_array.shape[0]
        aligned = np.zeros((C_sar, H_opt, W_opt), dtype=np.float32)
        for i in range(C_sar):
            reproject(
                source=sar_array[i], destination=aligned[i],
                src_transform=sar_profile["transform"],
                src_crs=sar_profile["crs"],
                dst_transform=opt_profile["transform"],
                dst_crs=opt_profile["crs"],
                resampling=Resampling.average
            )
        return aligned

    @staticmethod
    def _boxcar_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        result = data.copy()
        for c in range(data.shape[0]):
            result[c] = uniform_filter(data[c], size=window_size)
        return result

    @staticmethod
    def _median_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        result = data.copy()
        for c in range(data.shape[0]):
            result[c] = median_filter(data[c], size=window_size)
        return result

    @staticmethod
    def refined_lee_filter(img: np.ndarray, win_size: int = 5) -> np.ndarray:
        if img.ndim == 3:
            result = img.copy()
            for c in range(img.shape[0]):
                result[c] = SAROpticalAligner._refined_lee_single(img[c], win_size)
            return result
        return SAROpticalAligner._refined_lee_single(img, win_size)

    @staticmethod
    def _refined_lee_single(img: np.ndarray, win_size: int = 5) -> np.ndarray:
        img_mean = uniform_filter(img, size=win_size)
        img_sq_mean = uniform_filter(img ** 2, size=win_size)
        img_var = img_sq_mean - img_mean ** 2
        overall_var = np.var(img)
        noise_var = overall_var / (1 + overall_var + 1e-10)
        weight = img_var / (img_var + noise_var + 1e-10)
        return img_mean + weight * (img - img_mean)

    @staticmethod
    def compute_sar_features(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        eps = 1e-6
        ratio = vv - vh
        rvi = (4 * vh) / (vv + vh + eps)
        rfdi = (vv - vh) / (vv + vh + eps)
        return np.stack([vv, vh, ratio, rvi, rfdi], axis=0)
