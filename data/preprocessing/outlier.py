"""
异常值检测与修复模块
"""
import numpy as np


class OutlierDetector:
    """多策略异常值检测与修复"""

    def __init__(self, z_thresh: float = 3.5, temporal_diff_thresh: float = 0.3):
        self.z_thresh = z_thresh
        self.temporal_diff_thresh = temporal_diff_thresh

    def detect_physical_outliers(self, sequence: np.ndarray) -> np.ndarray:
        return (sequence < -0.05) | (sequence > 1.05)

    def detect_temporal_outliers(self, sequence: np.ndarray) -> np.ndarray:
        diff = np.abs(np.diff(sequence, axis=0))
        diff_fwd = np.concatenate([diff, diff[-1:]], axis=0)
        diff_bwd = np.concatenate([diff[:1], diff], axis=0)
        return (diff_fwd > self.temporal_diff_thresh) & (diff_bwd > self.temporal_diff_thresh)

    def detect_zscore_outliers(self, sequence: np.ndarray) -> np.ndarray:
        mean = np.nanmean(sequence, axis=0, keepdims=True)
        std = np.nanstd(sequence, axis=0, keepdims=True) + 1e-6
        return np.abs((sequence - mean) / std) > self.z_thresh

    def detect(self, sequence: np.ndarray) -> np.ndarray:
        m1 = self.detect_physical_outliers(sequence)
        m2 = self.detect_temporal_outliers(sequence)
        m3 = self.detect_zscore_outliers(sequence)
        return m1 | m2 | m3

    def run_all_checks(self, sequence: np.ndarray) -> np.ndarray:
        return self.detect(sequence)

    def fix_outliers(self, sequence: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        result = sequence.copy()
        result[outlier_mask] = np.nan
        return result

    def fix(self, sequence: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        return self.fix_outliers(sequence, outlier_mask)
