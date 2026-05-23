"""
空间无泄漏数据集划分模块
"""
import numpy as np
import rasterio
from typing import Tuple


class SpatialDataSplitter:
    """块状空间划分（防空间自相关泄漏）"""

    def __init__(self, block_size_px: int = 64):
        self.block_size = block_size_px

    def split(self, label_map: np.ndarray, train_ratio: float = 0.70,
              val_ratio: float = 0.15, seed: int = 42
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = label_map.shape
        bs = self.block_size
        n_rows, n_cols = (H + bs - 1) // bs, (W + bs - 1) // bs
        n_blocks = n_rows * n_cols

        rng = np.random.default_rng(seed)
        order = rng.permutation(n_blocks)
        n_train, n_val = int(n_blocks * train_ratio), int(n_blocks * val_ratio)

        train_blocks = set(order[:n_train].tolist())
        val_blocks = set(order[n_train:n_train + n_val].tolist())
        test_blocks = set(order[n_train + n_val:].tolist())

        train_mask = np.zeros((H, W), dtype=bool)
        val_mask = np.zeros((H, W), dtype=bool)
        test_mask = np.zeros((H, W), dtype=bool)

        for idx in range(n_blocks):
            row, col = idx // n_cols, idx % n_cols
            r0, c0 = row * bs, col * bs
            r1, c1 = min(r0 + bs, H), min(c0 + bs, W)

            if idx in train_blocks:
                train_mask[r0:r1, c0:c1] = True
            elif idx in val_blocks:
                val_mask[r0:r1, c0:c1] = True
            else:
                test_mask[r0:r1, c0:c1] = True

        valid = (label_map != 0) & (label_map != 255)
        train_mask &= valid
        val_mask &= valid
        test_mask &= valid

        return train_mask, val_mask, test_mask

    def save_split_masks(self, train_mask: np.ndarray, val_mask: np.ndarray,
                         test_mask: np.ndarray, output_dir: str,
                         profile: dict) -> None:
        split_map = np.zeros(train_mask.shape, dtype=np.uint8)
        split_map[train_mask], split_map[val_mask], split_map[test_mask] = 1, 2, 3

        import os
        p = profile.copy()
        p.update({"count": 1, "dtype": "uint8"})
        with rasterio.open(f"{output_dir}/spatial_split_map.tif", "w", **p) as dst:
            dst.write(split_map, 1)

        np.save(f"{output_dir}/train_mask.npy", train_mask)
        np.save(f"{output_dir}/val_mask.npy", val_mask)
        np.save(f"{output_dir}/test_mask.npy", test_mask)
