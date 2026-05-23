"""
时序处理模块 - 缺失值插补、光学/SAR时相对齐
"""
import numpy as np
from scipy.interpolate import interp1d
from typing import Tuple, Optional


class TemporalInterpolator:
    """时序缺失值插补"""

    def __init__(self, max_gap: int = 30, method: str = "linear",
                 mask_long_gaps: bool = True, long_gap_threshold: int = 60):
        self.max_gap = max_gap
        self.method = method
        self.mask_long_gaps = mask_long_gaps
        self.long_gap_threshold = long_gap_threshold

    def interpolate(self, opt_seq: np.ndarray, doy: np.ndarray,
                    cloud_mask: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T, C, H, W = opt_seq.shape
        filled_seq = opt_seq.copy()
        is_interpolated = np.zeros((T, H, W), dtype=bool)
        updated_mask = cloud_mask.copy()
        valid_mask = ~cloud_mask

        for h in range(H):
            for w in range(W):
                for c in range(C):
                    valid_indices = np.where(valid_mask[:, h, w])[0]
                    if len(valid_indices) < 2:
                        continue
                    filled_seq[:, c, h, w] = np.interp(
                        np.arange(T), valid_indices,
                        opt_seq[valid_indices, c, h, w],
                        left=opt_seq[valid_indices[0], c, h, w],
                        right=opt_seq[valid_indices[-1], c, h, w]
                    )
                    interpolated_mask = np.ones(T, dtype=bool)
                    interpolated_mask[valid_indices] = False
                    is_interpolated[interpolated_mask, h, w] = True
                    updated_mask[interpolated_mask, h, w] = True

        if self.mask_long_gaps:
            for h in range(H):
                for w in range(W):
                    valid_indices = np.where(~updated_mask[:, h, w])[0]
                    if len(valid_indices) < 2:
                        updated_mask[:, h, w] = True
                        continue
                    gaps = np.diff(valid_indices) * (doy[1] - doy[0]) * 365
                    long_gap_mask = gaps > self.long_gap_threshold
                    for i, is_long_gap in enumerate(long_gap_mask):
                        if is_long_gap:
                            start, end = valid_indices[i] + 1, valid_indices[i + 1]
                            updated_mask[start:end, h, w] = True
                            is_interpolated[start:end, h, w] = True

        valid_count = (~updated_mask).sum(axis=0).astype(np.float32)
        return filled_seq, updated_mask, is_interpolated, valid_count

    @staticmethod
    def interpolate_timeseries(sequence: np.ndarray, doy: np.ndarray,
                               method: str = "cubic",
                               max_gap_days: int = 45
                               ) -> Tuple[np.ndarray, np.ndarray]:
        T, C, H, W = sequence.shape
        filled = sequence.copy()
        valid_ratio = (~np.isnan(sequence[:, 0])).mean(axis=0)

        for r in range(H):
            for c in range(W):
                for ch in range(C):
                    ts = sequence[:, ch, r, c]
                    valid_idx = np.where(~np.isnan(ts))[0]
                    if len(valid_idx) < 2:
                        filled[:, ch, r, c] = np.nanmean(ts) if len(valid_idx) > 0 else 0.0
                        continue
                    interp_fn = interp1d(doy[valid_idx], ts[valid_idx],
                                         kind=method, bounds_error=False,
                                         fill_value=(ts[valid_idx[0]], ts[valid_idx[-1]]))
                    nan_idx = np.where(np.isnan(ts))[0]
                    if len(nan_idx) > 0:
                        filled[nan_idx, ch, r, c] = interp_fn(doy[nan_idx])
        return filled, valid_ratio

    @staticmethod
    def quality_check_sequence(sequence: np.ndarray,
                               min_valid_ratio: float = 0.5) -> np.ndarray:
        valid_ratio = (~np.isnan(sequence[:, 0])).mean(axis=0)
        qa_mask = np.zeros(valid_ratio.shape, dtype=np.uint8)
        qa_mask[valid_ratio < min_valid_ratio] = 255
        return qa_mask


class TemporalAligner:
    """光学/SAR 时相对齐"""

    def __init__(self, max_gap_days: int = 16):
        self.max_gap_days = max_gap_days

    def align(self, opt_sequence: np.ndarray, opt_doys: np.ndarray,
              sar_sequence: np.ndarray, sar_doys: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        T_opt = opt_sequence.shape[0]
        opt_aligned_list, sar_aligned_list, common_doy_list = [], [], []

        for t_opt in range(T_opt):
            doy_opt = opt_doys[t_opt]
            doy_diffs = np.abs(sar_doys - doy_opt)
            nearest_t = int(doy_diffs.argmin())
            min_gap = float(doy_diffs[nearest_t])

            if min_gap > self.max_gap_days:
                continue

            if min_gap <= 3:
                sar_for_this = sar_sequence[nearest_t]
            else:
                before_mask = sar_doys <= doy_opt
                after_mask = sar_doys >= doy_opt
                if before_mask.any() and after_mask.any():
                    t_before = int(np.where(before_mask)[0][-1])
                    t_after = int(np.where(after_mask)[0][0])
                    d_before = doy_opt - sar_doys[t_before]
                    d_after = sar_doys[t_after] - doy_opt
                    total = d_before + d_after + 1e-6
                    w_before = d_after / total
                    w_after = d_before / total
                    sar_for_this = (w_before * sar_sequence[t_before] +
                                    w_after * sar_sequence[t_after])
                else:
                    sar_for_this = sar_sequence[nearest_t]

            opt_aligned_list.append(opt_sequence[t_opt])
            sar_aligned_list.append(sar_for_this)
            common_doy_list.append(doy_opt)

        opt_aligned = np.stack(opt_aligned_list, axis=0)
        sar_aligned = np.stack(sar_aligned_list, axis=0)
        common_doys = np.array(common_doy_list)
        return opt_aligned, sar_aligned, common_doys
