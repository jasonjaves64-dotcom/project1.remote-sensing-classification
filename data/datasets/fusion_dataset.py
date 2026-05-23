"""
Patch-based Fusion Crop Dataset with data augmentation.

Used by both train_fusion.py and train_fusion_edl.py.
Provides FusionCropDatasetEDL and shared compute_metrics.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class FusionCropDatasetEDL(Dataset):
    CROP_CLASSES = {
        0: "背景", 1: "冬小麦", 2: "夏玉米",
        3: "水稻", 4: "大豆", 5: "棉花", 6: "其他"
    }

    def __init__(self, opt_seq, sar_seq, doy_norm, label,
                 patch_size=32, augment=True, mask=None, dem_data=None):
        self.opt_seq = opt_seq
        self.sar_seq = sar_seq
        self.doy_norm = doy_norm
        self.label = label
        self.patch_size = patch_size
        self.augment = augment
        self.dem_data = dem_data

        H, W = label.shape
        p = patch_size
        self.coords = []

        for r in range(0, H - p + 1, p // 2):
            for c in range(0, W - p + 1, p // 2):
                patch_label = label[r:r+p, c:c+p]
                if patch_label.max() > 0 and patch_label.min() != 255:
                    if mask is None or mask[r + p//2, c + p//2]:
                        self.coords.append((r, c))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        p = self.patch_size
        opt = self.opt_seq[:, :, r:r+p, c:c+p].copy()
        sar = self.sar_seq[:, :, r:r+p, c:c+p].copy()
        y = self.label[r:r+p, c:c+p].copy()
        y[y == 255] = 0
        if self.dem_data is not None:
            dem = self.dem_data[:, r:r+p, c:c+p].copy()
        else:
            dem = np.zeros((5, p, p), dtype=np.float32)
        if self.augment:
            opt, sar, y, dem = self._augment(opt, sar, y, dem)
        return {
            "opt": torch.from_numpy(opt).float(),
            "sar": torch.from_numpy(sar).float(),
            "dem": torch.from_numpy(dem).float(),
            "doy": torch.from_numpy(self.doy_norm).float(),
            "y": torch.from_numpy(y).long()
        }

    def _augment(self, opt, sar, y, dem):
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-1).copy()
            sar = np.flip(sar, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()
            dem = np.flip(dem, axis=-1).copy()
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-2).copy()
            sar = np.flip(sar, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()
            dem = np.flip(dem, axis=-2).copy()
        if np.random.rand() > 0.5:
            opt = np.transpose(opt, (0, 1, 3, 2)).copy()
            sar = np.transpose(sar, (0, 1, 3, 2)).copy()
            y = np.transpose(y, (1, 0)).copy()
            dem = np.transpose(dem, (0, 2, 1)).copy()
        T = opt.shape[0]
        n_drop = np.random.randint(0, 4)
        opt_drop_idx = np.random.choice(T, n_drop, replace=False)
        opt[opt_drop_idx] = 0.0
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, sar.shape).astype(np.float32)
            sar = sar + noise
        if np.random.rand() > 0.3:
            noise = np.random.normal(0, 0.03, opt.shape).astype(np.float32)
            opt = opt + noise
        return opt, sar, y, dem


def compute_metrics(preds, labels, num_classes):
    valid = labels != 0
    p = preds[valid]
    l = labels[valid]
    oa = (p == l).float().mean().item()
    iou_list = []
    for cls in range(1, num_classes):
        tp = ((p == cls) & (l == cls)).sum().float()
        fp = ((p == cls) & (l != cls)).sum().float()
        fn = ((p != cls) & (l == cls)).sum().float()
        iou_list.append((tp / (tp + fp + fn + 1e-6)).item())
    return {
        "OA": oa,
        "mIoU": sum(iou_list)/len(iou_list) if iou_list else 0,
        "IoU_per_class": iou_list
    }
