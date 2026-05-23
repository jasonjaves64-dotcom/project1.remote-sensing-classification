"""
数据增强模块 - 空间变换（归一化前）
"""
import numpy as np
from .base import DataSample


class DataAugmenter:
    """空间数据增强（翻转、旋转）"""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def augment(self, sample: DataSample) -> DataSample:
        if np.random.rand() > self.prob:
            return sample

        opt = sample.opt_seq.copy()
        sar = sample.sar_seq.copy()
        dem = sample.dem.copy()
        label = sample.label.copy() if sample.label is not None else None
        cloud_mask = sample.cloud_mask.copy() if sample.cloud_mask is not None else None
        is_interpolated = sample.is_interpolated.copy() if sample.is_interpolated is not None else None

        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-1)
            sar = np.flip(sar, axis=-1)
            dem = np.flip(dem, axis=-1)
            if label is not None:
                label = np.flip(label, axis=-1)
            if cloud_mask is not None:
                cloud_mask = np.flip(cloud_mask, axis=-1)
            if is_interpolated is not None:
                is_interpolated = np.flip(is_interpolated, axis=-1)

        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-2)
            sar = np.flip(sar, axis=-2)
            dem = np.flip(dem, axis=-2)
            if label is not None:
                label = np.flip(label, axis=-2)
            if cloud_mask is not None:
                cloud_mask = np.flip(cloud_mask, axis=-2)
            if is_interpolated is not None:
                is_interpolated = np.flip(is_interpolated, axis=-2)

        return DataSample(
            opt_seq=opt, sar_seq=sar, dem=dem, doy=sample.doy,
            label=label, cloud_mask=cloud_mask,
            valid_count=sample.valid_count,
            is_interpolated=is_interpolated
        )
