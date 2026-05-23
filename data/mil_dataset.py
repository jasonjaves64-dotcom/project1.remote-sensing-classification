"""
MIL (Multi-Instance Learning) Dataset classes.

Supports bag-instance data structures for weakly supervised learning.
Each image is a "bag" and its sub-region patches are "instances".
Only image-level labels are required.

Classes:
  MILCropDataset      — Builds bags from pre-loaded numpy arrays
  MILCropDatasetFiles — Builds bags from file paths (for large datasets)
  mil_collate_fn      — Custom collate for variable-length bags
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path


CROP_CLASSES = {
    0: "背景", 1: "冬小麦", 2: "夏玉米",
    3: "水稻", 4: "大豆", 5: "棉花", 6: "其他"
}


def mil_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate function for variable-length bags.

    Pads each bag to max_instances across the batch and creates an
    instance_mask to indicate real vs padding positions.

    Args:
        batch: list of dicts, each with keys:
            opt, sar, dem, doy, bag_label, (optional) pixel_labels

    Returns:
        dict with padded tensors + instance_mask
    """
    max_n = max(item["opt"].shape[0] for item in batch)
    B = len(batch)

    opt_bag = torch.zeros(B, max_n, *batch[0]["opt"].shape[1:])
    sar_bag = torch.zeros(B, max_n, *batch[0]["sar"].shape[1:])
    dem_bag = torch.zeros(B, max_n, *batch[0]["dem"].shape[1:])
    doy_bag = torch.zeros(B, max_n, *batch[0]["doy"].shape[1:])
    instance_mask = torch.zeros(B, max_n, dtype=torch.bool)
    bag_labels = torch.zeros(B, dtype=torch.long)

    has_pixel_labels = "pixel_labels" in batch[0] and batch[0]["pixel_labels"] is not None

    if has_pixel_labels:
        pixel_labels_bag = torch.full(
            (B, max_n, *batch[0]["pixel_labels"].shape[1:]),
            255, dtype=torch.long)  # Match EDLLoss.ignore_index
    else:
        pixel_labels_bag = None

    for i, item in enumerate(batch):
        n = item["opt"].shape[0]
        opt_bag[i, :n] = item["opt"]
        sar_bag[i, :n] = item["sar"]
        dem_bag[i, :n] = item["dem"]
        doy_bag[i, :n] = item["doy"]
        instance_mask[i, :n] = True
        bag_labels[i] = item["bag_label"]

        if has_pixel_labels:
            pixel_labels_bag[i, :n] = item["pixel_labels"]

    return {
        "opt": opt_bag,
        "sar": sar_bag,
        "dem": dem_bag,
        "doy": doy_bag,
        "instance_mask": instance_mask,
        "bag_label": bag_labels,
        "pixel_labels": pixel_labels_bag,
    }


class MILCropDataset(Dataset):
    """MIL dataset from pre-loaded numpy arrays.

    Splits a large remote sensing image into bags. Each bag contains
    patch instances extracted via sliding window. The bag label is
    derived from the majority crop class among valid pixels in the bag area.

    Args:
        opt_seq:    (T, C_opt, H, W) optical time-series
        sar_seq:    (T, C_sar, H, W) SAR time-series
        doy_norm:   (T,) normalized DOY values
        label:      (H, W) pixel-level labels (only used to derive bag labels)
        dem_data:   (C_dem, H, W) or None
        bag_size:   spatial size of each bag (e.g., 128×128 pixels)
        patch_size: size of each instance/patch (e.g., 32×32 pixels)
        stride:     stride for sliding window within each bag
        augment:    apply data augmentation
        min_valid_pct: minimum fraction of valid (non-zero) pixels to keep a bag
        return_pixel_labels: if True, also return pixel-level labels for hybrid training
    """

    CROP_CLASSES = CROP_CLASSES

    def __init__(self, opt_seq: np.ndarray, sar_seq: np.ndarray,
                 doy_norm: np.ndarray, label: np.ndarray,
                 dem_data: Optional[np.ndarray] = None,
                 bag_size: int = 128, patch_size: int = 32,
                 stride: int = 16, augment: bool = True,
                 min_valid_pct: float = 0.3,
                 return_pixel_labels: bool = False):
        self.opt_seq = opt_seq
        self.sar_seq = sar_seq
        self.doy_norm = doy_norm
        self.label = label
        self.dem_data = dem_data
        self.bag_size = bag_size
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.min_valid_pct = min_valid_pct
        self.return_pixel_labels = return_pixel_labels
        self.num_classes = 7

        H, W = label.shape
        self.bag_coords = []

        for r in range(0, H - bag_size + 1, bag_size // 2):
            for c in range(0, W - bag_size + 1, bag_size // 2):
                bag_label_area = label[r:r + bag_size, c:c + bag_size]
                valid = bag_label_area[bag_label_area != 255]
                if len(valid) == 0:
                    continue
                valid_frac = len(valid) / (bag_size * bag_size)
                if valid_frac < min_valid_pct:
                    continue
                # Bag label = majority class (excluding background=0)
                classes, counts = np.unique(valid, return_counts=True)
                non_bg = classes != 0
                if non_bg.sum() == 0:
                    continue
                majority_class = classes[non_bg][counts[non_bg].argmax()]
                self.bag_coords.append((r, c, int(majority_class)))

    def __len__(self):
        return len(self.bag_coords)

    def __getitem__(self, idx: int) -> dict:
        r, c, bag_label = self.bag_coords[idx]
        bs = self.bag_size
        ps = self.patch_size

        # Build instance list for this bag
        instances_opt = []
        instances_sar = []
        instances_dem = []
        instances_label = []

        for pr in range(0, bs - ps + 1, self.stride):
            for pc in range(0, bs - ps + 1, self.stride):
                abs_r, abs_c = r + pr, c + pc
                patch_opt = self.opt_seq[:, :, abs_r:abs_r + ps, abs_c:abs_c + ps].copy()
                patch_sar = self.sar_seq[:, :, abs_r:abs_r + ps, abs_c:abs_c + ps].copy()
                patch_label = self.label[abs_r:abs_r + ps, abs_c:abs_c + ps].copy()

                if self.dem_data is not None:
                    patch_dem = self.dem_data[:, abs_r:abs_r + ps, abs_c:abs_c + ps].copy()
                else:
                    patch_dem = np.zeros((5, ps, ps), dtype=np.float32)

                # Skip fully invalid patches
                patch_label_clean = patch_label.copy()
                patch_label_clean[patch_label_clean == 255] = 0

                instances_opt.append(patch_opt)
                instances_sar.append(patch_sar)
                instances_dem.append(patch_dem)
                instances_label.append(patch_label_clean)

        if len(instances_opt) == 0:
            # Fallback: at least one instance at center
            cr, cc = bs // 2 - ps // 2, bs // 2 - ps // 2
            instances_opt = [self.opt_seq[:, :, r + cr:r + cr + ps, c + cc:c + cc + ps].copy()]
            instances_sar = [self.sar_seq[:, :, r + cr:r + cr + ps, c + cc:c + cc + ps].copy()]
            patch_label = self.label[r + cr:r + cr + ps, c + cc:c + cc + ps].copy()
            patch_label[patch_label == 255] = 0
            instances_label = [patch_label]
            if self.dem_data is not None:
                instances_dem = [self.dem_data[:, r + cr:r + cr + ps, c + cc:c + cc + ps].copy()]
            else:
                instances_dem = [np.zeros((5, ps, ps), dtype=np.float32)]

        # Augmentation (applied consistently to all instances in the bag)
        if self.augment:
            instances_opt, instances_sar, instances_dem, instances_label = self._augment_bag(
                instances_opt, instances_sar, instances_dem, instances_label)

        opt_bag = torch.from_numpy(np.stack(instances_opt, axis=0)).float()
        sar_bag = torch.from_numpy(np.stack(instances_sar, axis=0)).float()
        dem_bag = torch.from_numpy(np.stack(instances_dem, axis=0)).float()
        doy_tensor = torch.from_numpy(self.doy_norm).float().unsqueeze(0).expand(
            len(instances_opt), -1)

        result = {
            "opt": opt_bag,       # (N, T, 10, P, P)
            "sar": sar_bag,       # (N, T, 5,  P, P)
            "dem": dem_bag,       # (N, 5, P, P)
            "doy": doy_tensor,    # (N, T)
            "bag_label": torch.tensor(bag_label, dtype=torch.long),
        }

        if self.return_pixel_labels:
            pixel_labels = np.stack(instances_label, axis=0)
            result["pixel_labels"] = torch.from_numpy(pixel_labels).long()
        else:
            result["pixel_labels"] = None

        return result

    def _augment_bag(self, instances_opt, instances_sar,
                     instances_dem, instances_label):
        """Apply same augmentation to all instances in a bag."""
        if np.random.rand() > 0.5:
            instances_opt = [np.flip(p, axis=-1).copy() for p in instances_opt]
            instances_sar = [np.flip(p, axis=-1).copy() for p in instances_sar]
            instances_dem = [np.flip(p, axis=-1).copy() for p in instances_dem]
            instances_label = [np.flip(p, axis=-1).copy() for p in instances_label]

        if np.random.rand() > 0.5:
            instances_opt = [np.flip(p, axis=-2).copy() for p in instances_opt]
            instances_sar = [np.flip(p, axis=-2).copy() for p in instances_sar]
            instances_dem = [np.flip(p, axis=-2).copy() for p in instances_dem]
            instances_label = [np.flip(p, axis=-2).copy() for p in instances_label]

        if np.random.rand() > 0.5:
            instances_opt = [np.transpose(p, (0, 1, 3, 2)).copy() for p in instances_opt]
            instances_sar = [np.transpose(p, (0, 1, 3, 2)).copy() for p in instances_sar]
            instances_dem = [np.transpose(p, (0, 2, 1)).copy() for p in instances_dem]
            instances_label = [np.transpose(p, (1, 0)).copy() for p in instances_label]

        # Timestep dropout
        T = instances_opt[0].shape[0]
        n_drop = np.random.randint(0, 4)
        drop_idx = np.random.choice(T, n_drop, replace=False)
        for i in range(len(instances_opt)):
            instances_opt[i][drop_idx] = 0.0

        return instances_opt, instances_sar, instances_dem, instances_label


class MILCropDatasetFiles(Dataset):
    """MIL dataset reading from individual GeoTIFF/npy file paths.

    Each file is a bag (e.g., a field polygon). Patches are extracted
    from each file and the file-level label is the bag label.

    Args:
        opt_paths:   list of paths to optical data files
        sar_paths:   list of paths to SAR data files
        dem_paths:   list of paths to DEM data files (or None)
        labels:      list of integer bag labels
        patch_size:  instance patch size
        n_instances: number of random instances to sample per bag
        augment:     apply data augmentation
    """

    def __init__(self, opt_paths: list[str], sar_paths: list[str],
                 labels: list[int], dem_paths: Optional[list[str]] = None,
                 patch_size: int = 32, n_instances: int = 49,
                 augment: bool = True):
        self.opt_paths = opt_paths
        self.sar_paths = sar_paths
        self.dem_paths = dem_paths
        self.labels = labels
        self.patch_size = patch_size
        self.n_instances = n_instances
        self.augment = augment

    def __len__(self):
        return len(self.opt_paths)

    def __getitem__(self, idx: int) -> dict:
        bag_label = self.labels[idx]

        opt = self._load(self.opt_paths[idx])  # (T, C_opt, H, W)
        sar = self._load(self.sar_paths[idx])  # (T, C_sar, H, W)

        if self.dem_paths is not None and self.dem_paths[idx] is not None:
            dem = self._load(self.dem_paths[idx])  # (C_dem, H, W)
        else:
            dem = np.zeros((5, opt.shape[2], opt.shape[3]), dtype=np.float32)

        T, _, H, W = opt.shape
        ps = self.patch_size

        instances_opt = []
        instances_sar = []
        instances_dem = []

        for _ in range(self.n_instances):
            r = np.random.randint(0, max(1, H - ps))
            c = np.random.randint(0, max(1, W - ps))
            instances_opt.append(opt[:, :, r:r + ps, c:c + ps].copy())
            instances_sar.append(sar[:, :, r:r + ps, c:c + ps].copy())
            instances_dem.append(dem[:, r:r + ps, c:c + ps].copy())

        # DOY: placeholder array; real DOY should come from file metadata
        doy_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)

        opt_bag = torch.from_numpy(np.stack(instances_opt, axis=0)).float()
        sar_bag = torch.from_numpy(np.stack(instances_sar, axis=0)).float()
        dem_bag = torch.from_numpy(np.stack(instances_dem, axis=0)).float()
        doy_tensor = torch.from_numpy(doy_norm).float().unsqueeze(0).expand(
            self.n_instances, -1)

        return {
            "opt": opt_bag,
            "sar": sar_bag,
            "dem": dem_bag,
            "doy": doy_tensor,
            "bag_label": torch.tensor(bag_label, dtype=torch.long),
            "pixel_labels": None,
        }

    @staticmethod
    def _load(path: str) -> np.ndarray:
        path = str(path)
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith(('.tif', '.tiff')):
            import rasterio
            with rasterio.open(path) as src:
                return src.read().astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {path}")
