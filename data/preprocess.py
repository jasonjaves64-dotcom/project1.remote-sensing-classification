"""
简化版预处理入口 - 构建时序序列并计算植被指数
向后兼容的薄包装，委托给统一预处理模块
"""
import numpy as np
from pathlib import Path
from datetime import datetime

# 委托给统一模块
from data.preprocessing.optical import BAND_NAMES, BAND_IDX, compute_spectral_indices


def build_time_sequence(img_dir: str, year: int = 2023,
                        cloud_threshold: float = 0.2
                        ) -> tuple[np.ndarray, np.ndarray]:
    """构建时序序列（向后兼容函数签名）"""
    tif_paths = sorted(Path(img_dir).glob("*.tif"))
    npy_paths = sorted(Path(img_dir).glob("*.npy"))

    if len(tif_paths) > 0:
        img_paths = tif_paths
        use_rasterio = True
    elif len(npy_paths) > 0:
        img_paths = npy_paths
        use_rasterio = False
    else:
        raise ValueError(f"未找到影像文件: {img_dir}")

    sequences, doys = [], []
    for path in img_paths:
        if use_rasterio:
            import rasterio
            with rasterio.open(path) as src:
                qa = src.read(7)
                cloud_mask = (qa & 0b11000) >> 3
                cloud_ratio = (cloud_mask >= 2).mean()
                if cloud_ratio > cloud_threshold:
                    continue
                bands = src.read(BAND_IDX).astype(np.float32)
                bands = bands / 10000.0
                bands = np.clip(bands, 0, 1)
        else:
            bands = np.load(path).astype(np.float32)
            bands = bands[:6]
            bands = bands / 10000.0
            bands = np.clip(bands, 0, 1)

        date_str = path.stem.split("_")[3]
        doy = datetime.strptime(date_str, "%Y%m%d").timetuple().tm_yday
        sequences.append(bands)
        doys.append(doy)

    sequence = np.stack(sequences, axis=0)
    sequence = compute_spectral_indices(sequence)
    doy_norm = np.array(doys, dtype=np.float32) / 365.0
    return sequence, doy_norm
