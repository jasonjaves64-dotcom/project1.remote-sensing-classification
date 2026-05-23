"""
预处理基类模块 - 公共基类、配置管理、数据验证、内存优化
"""
import os
import yaml
import json
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterator
from dataclasses import dataclass, field


# ── YAML 配置加载 ──────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """从 YAML 文件加载预处理配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: dict, config_path: str):
    """保存预处理配置到 YAML 文件"""
    os.makedirs(Path(config_path).parent, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ── 数据类 ──────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    """预处理配置数据类"""
    # 数据路径
    landsat_dir: str = "./data/raw/landsat"
    sar_dir: str = "./data/raw/sentinel1"
    dem_path: str = "./data/dem/dem_30m.tif"
    label_shp: str = "./data/labels/crop_parcels_2023.shp"
    output_dir: str = "./data/processed"
    class_field: str = "crop_type"
    year: int = 2023

    # 空间配准
    target_resolution: float = 30.0
    target_crs: str = "EPSG:32650"
    align_method: str = "bilinear"

    # 云检测
    cloud_threshold: float = 0.3
    max_cloud_pct: float = 0.5
    use_sar_for_cloud_mask: bool = True

    # SAR 处理
    sar_log_transform: bool = True
    sar_speckle_method: str = "refined_lee"

    # 时序插值
    max_gap: int = 30
    max_gap_days: int = 16
    interpolation_method: str = "linear"
    mask_long_gaps: bool = True
    long_gap_threshold: int = 60

    # 归一化
    normalize: bool = True
    norm_method: str = "robust"
    global_stats_path: Optional[str] = None
    freeze_stats: bool = False

    # 地形校正
    apply_terrain_correction: bool = False
    terrain_correction_method: str = "minnaert"
    solar_zenith: float = 30.0
    solar_azimuth: float = 150.0

    # 异常值检测
    outlier_z_thresh: float = 3.5
    temporal_diff_thresh: float = 0.3

    # 数据增强（归一化前）
    augment: bool = False
    augment_prob: float = 0.5

    # 输出
    output_format: str = "npy"

    # 空间划分
    split_block_size: int = 64
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # 标签处理
    erosion_pixels: int = 2
    purity_threshold: float = 0.8
    min_valid_observations: int = 4

    # 融合
    fusion_mode: str = "concat"
    enable_supplement: bool = True

    # 内存优化
    use_memmap: bool = False
    use_generator: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "PreprocessConfig":
        cfg = load_config(path)
        preproc_cfg = cfg.get('preprocessing', cfg)
        return cls(**{k: v for k, v in preproc_cfg.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class DataSample:
    """标准化数据样本"""
    opt_seq: np.ndarray
    sar_seq: np.ndarray
    dem: np.ndarray
    doy: np.ndarray
    cloud_mask: Optional[np.ndarray] = None
    valid_count: Optional[np.ndarray] = None
    is_interpolated: Optional[np.ndarray] = None
    label: Optional[np.ndarray] = None


# ── 抽象基类 ────────────────────────────────────────────────────

class BasePreprocessor(ABC):
    """预处理基类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else {}

    @abstractmethod
    def process(self, data):
        raise NotImplementedError("子类必须实现 process 方法")

    def validate_input(self, data):
        if data is None:
            raise ValueError("输入数据不能为空")
        return True

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def update_config(self, **kwargs):
        self.config.update(kwargs)


# ── 数据验证器 ──────────────────────────────────────────────────

class DataValidator:
    """预处理数据验证器"""

    @staticmethod
    def validate_optical(data: np.ndarray, expected_bands: int = 6) -> List[str]:
        errors = []
        if data is None or data.size == 0:
            errors.append("光学数据为空")
            return errors
        if np.all(np.isnan(data)):
            errors.append("光学数据全为 NaN")
        if expected_bands > 0 and data.shape[1] < expected_bands:
            errors.append(f"光学通道数不足: 期望 >= {expected_bands}, 实际 {data.shape[1]}")
        neg_ratio = (data < -0.1).mean()
        if neg_ratio > 0.3:
            errors.append(f"负值过多 ({neg_ratio:.1%})，数据可能未正确校准")
        return errors

    @staticmethod
    def validate_sar(data: np.ndarray) -> List[str]:
        errors = []
        if data is None or data.size == 0:
            errors.append("SAR 数据为空")
            return errors
        if np.all(np.isnan(data)):
            errors.append("SAR 数据全为 NaN")
        if data.shape[1] < 2:
            errors.append(f"SAR 通道数不足: 期望 >= 2, 实际 {data.shape[1]}")
        return errors

    @staticmethod
    def validate_doy(doy: np.ndarray, expected_len: int = 0) -> List[str]:
        errors = []
        if doy is None or doy.size == 0:
            errors.append("DOY 数据为空")
            return errors
        if expected_len > 0 and len(doy) != expected_len:
            errors.append(f"DOY 长度不匹配: 期望 {expected_len}, 实际 {len(doy)}")
        if np.any(doy < 0) or np.any(doy > 366):
            errors.append("DOY 值越界 [0, 366]")
        return errors

    @staticmethod
    def validate_label(label: np.ndarray, num_classes: int = 7) -> List[str]:
        errors = []
        if label is None or label.size == 0:
            errors.append("标签数据为空")
            return errors
        max_cls = int(label[label < 255].max() if (label < 255).any() else 0)
        if max_cls >= num_classes:
            errors.append(f"标签类别越界: max={max_cls}, num_classes={num_classes}")
        valid_ratio = ((label > 0) & (label < 255)).mean()
        if valid_ratio < 0.01:
            errors.append(f"有效标签占比过低 ({valid_ratio:.2%})")
        return errors

    @staticmethod
    def validate_sample(sample: DataSample) -> Dict[str, List[str]]:
        results = {}
        results['optical'] = DataValidator.validate_optical(sample.opt_seq)
        results['sar'] = DataValidator.validate_sar(sample.sar_seq)
        results['doy'] = DataValidator.validate_doy(sample.doy, sample.opt_seq.shape[0])
        if sample.label is not None:
            results['label'] = DataValidator.validate_label(sample.label)
        return results

    @staticmethod
    def is_valid(sample: DataSample, verbose: bool = False) -> bool:
        results = DataValidator.validate_sample(sample)
        all_errors = [e for errors in results.values() for e in errors]
        if verbose and all_errors:
            for err in all_errors:
                print(f"  [验证失败] {err}")
        return len(all_errors) == 0


# ── 内存优化工具 ────────────────────────────────────────────────

def create_memmap(path: str, shape: Tuple[int, ...], dtype: str = 'float32') -> np.ndarray:
    """创建内存映射数组（磁盘缓存，按需加载）"""
    os.makedirs(Path(path).parent, exist_ok=True)
    return np.lib.format.open_memmap(path, mode='w+', dtype=np.dtype(dtype), shape=shape)


def load_or_compute_memmap(path: str, compute_fn, shape: Tuple[int, ...],
                           dtype: str = 'float32', force_recompute: bool = False) -> np.ndarray:
    """加载已有的 memmap，或计算并保存"""
    if os.path.exists(path) and not force_recompute:
        return np.load(path, mmap_mode='r')
    data = compute_fn()
    mmap = create_memmap(path, shape, dtype)
    mmap[:] = data[:]
    mmap.flush()
    return mmap


def batch_generator(opt_seq: np.ndarray, sar_seq: np.ndarray, dem: np.ndarray,
                    doy: np.ndarray, label: np.ndarray, batch_size: int,
                    shuffle: bool = False) -> Iterator[Dict[str, np.ndarray]]:
    """生成器模式逐批产出数据，降低内存峰值"""
    T, C_opt, H, W = opt_seq.shape
    num_pixels = H * W
    indices = np.arange(num_pixels)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, num_pixels, batch_size):
        batch_idx = indices[start:start + batch_size]
        rows = batch_idx // W
        cols = batch_idx % W
        batch_opt = opt_seq[:, :, rows, cols].transpose(2, 0, 1, 3)
        batch_sar = sar_seq[:, :, rows, cols].transpose(2, 0, 1, 3)
        batch_dem = dem[:, rows, cols].transpose(1, 0, 2)
        batch_label = label[rows, cols]
        batch_doy = doy.copy()

        yield {
            'opt': batch_opt.astype(np.float32),
            'sar': batch_sar.astype(np.float32),
            'dem': batch_dem.astype(np.float32),
            'doy': batch_doy.astype(np.float32),
            'label': batch_label.astype(np.int64)
        }


def sliding_window_generator(data: np.ndarray, window_size: int, stride: int
                             ) -> Iterator[Tuple[np.ndarray, int, int]]:
    """滑动窗口生成器，用于大图分块处理"""
    if data.ndim == 4:
        T, C, H, W = data.shape
    else:
        C, H, W = data.shape
        T = 0

    for r in range(0, H - window_size + 1, stride):
        for c in range(0, W - window_size + 1, stride):
            if T > 0:
                patch = data[:, :, r:r + window_size, c:c + window_size]
            else:
                patch = data[:, r:r + window_size, c:c + window_size]
            yield patch, r, c
