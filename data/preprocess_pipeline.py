"""
向后兼容模块 - 从 data.preprocessing 重新导出原有 API
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── 从统一模块导入 ──
from data.preprocessing.base import (PreprocessConfig, DataSample, DataValidator,
                                      load_config, save_config, batch_generator)
from data.preprocessing.optical import (GeoTIFFReader, CloudMaskProcessor,
                                        compute_spectral_indices, DataNormalizer,
                                        LS_SCALE, LS_OFFSET, OpticalProcessor)
from data.preprocessing.sar import SARProcessor, SAROpticalAligner
from data.preprocessing.temporal import TemporalInterpolator, TemporalAligner
from data.preprocessing.terrain import TerrainCorrector
from data.preprocessing.dem_features import DEMFeatureExtractor
from data.preprocessing.outlier import OutlierDetector
from data.preprocessing.spatial import SpatialDataSplitter
from data.preprocessing.label import LabelProcessor
from data.preprocessing.augment import DataAugmenter
from data.preprocessing.fusion import MultiModalFusion
from data.preprocessing.quality import QualityControl, PreprocessingQualityReport


# ── 别名（保持原有 API 名称）──

class SpatialAligner:
    """空间配准模块（向后兼容别名）"""
    def __init__(self, target_resolution: float = 10.0):
        self.target_resolution = target_resolution
        self._aligner = SAROpticalAligner()

    def align(self, data: Dict[str, np.ndarray],
              transforms: Dict[str, dict]) -> Dict[str, np.ndarray]:
        aligned = {}
        for modality, arr in data.items():
            if modality == 'dem':
                aligned[modality] = self._resample(arr, transforms.get(modality, {}))
            elif modality in ('opt', 'sar'):
                T = arr.shape[0]
                aligned_frames = [self._resample(arr[t], transforms.get(modality, {}))
                                  for t in range(T)]
                aligned[modality] = np.stack(aligned_frames)
            else:
                aligned[modality] = arr
        return aligned

    def _resample(self, arr: np.ndarray, transform: dict) -> np.ndarray:
        target_h, target_w = transform.get('target_size', arr.shape[-2:])
        return self._bilinear(arr, target_h, target_w)

    @staticmethod
    def _bilinear(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
        import torch
        if arr.ndim == 3:
            C = arr.shape[0]
            result = np.zeros((C, th, tw), dtype=arr.dtype)
            for c in range(C):
                result[c] = torch.nn.functional.interpolate(
                    torch.from_numpy(arr[c]).unsqueeze(0).unsqueeze(0),
                    size=(th, tw), mode='bilinear', align_corners=False
                ).squeeze().numpy()
            return result
        return torch.nn.functional.interpolate(
            torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
            size=(th, tw), mode='bilinear', align_corners=False
        ).squeeze().numpy()


class CloudDetector:
    """云检测模块（向后兼容别名）"""
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def detect(self, opt_seq: np.ndarray,
               sar_seq: Optional[np.ndarray] = None) -> np.ndarray:
        T, C, H, W = opt_seq.shape
        cloud_mask = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            if C >= 7:
                ndvi = (opt_seq[t, 6] - opt_seq[t, 2]) / (opt_seq[t, 6] + opt_seq[t, 2] + 1e-8)
                nir = opt_seq[t, 6]
                cloud_mask[t] = (nir > self.threshold) & (ndvi < 0.1)
            if sar_seq is not None:
                sar_mean = sar_seq[t].mean(axis=0)
                cloud_mask[t] |= (sar_mean < -15)
        return cloud_mask

    def refine_mask(self, cloud_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        from scipy.ndimage import binary_opening, binary_closing
        refined = cloud_mask.copy()
        for t in range(cloud_mask.shape[0]):
            refined[t] = binary_opening(cloud_mask[t],
                                         structure=np.ones((kernel_size, kernel_size)))
            refined[t] = binary_closing(refined[t],
                                         structure=np.ones((kernel_size, kernel_size)))
        return refined


class SpectralNormalizer:
    """光谱归一化模块（向后兼容别名 → DataNormalizer）"""
    def __init__(self, stats_path: Optional[str] = None, freeze_stats: bool = True):
        self.stats = {}
        self.freeze_stats = freeze_stats
        if stats_path:
            self.load_stats(stats_path)

    def fit(self, data: np.ndarray, modality: str):
        if self.freeze_stats and modality in self.stats:
            return
        self.stats[modality] = {
            'mean': data.mean(axis=(0, 2, 3)) if data.ndim == 4 else data.mean(),
            'std': data.std(axis=(0, 2, 3)) if data.ndim == 4 else data.std(),
            'min': float(data.min()),
            'max': float(data.max())
        }

    def normalize(self, data: np.ndarray, modality: str) -> np.ndarray:
        if modality not in self.stats:
            self.fit(data, modality)
        stats = self.stats[modality]
        mean = stats['mean']; std = stats['std']
        if data.ndim == 4:
            mean = mean.reshape(1, -1, 1, 1)
            std = std.reshape(1, -1, 1, 1)
        return (data - mean) / (std + 1e-8)

    def denormalize(self, data: np.ndarray, modality: str) -> np.ndarray:
        stats = self.stats[modality]
        mean = stats['mean']; std = stats['std']
        if data.ndim == 4:
            mean = mean.reshape(1, -1, 1, 1)
            std = std.reshape(1, -1, 1, 1)
        return data * std + mean

    def load_stats(self, path: str):
        import json
        with open(path) as f:
            self.stats = json.load(f)

    def save_stats(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)


class DataQualityChecker:
    """数据质量检查模块（向后兼容别名 → QualityControl）"""
    def __init__(self):
        self._qc = QualityControl()

    def check(self, sample: DataSample) -> Dict[str, bool]:
        results = self._qc.check_validity(sample.opt_seq)
        results['sar'] = self._qc.check_validity(sample.sar_seq)
        return {
            'nan': not bool(np.any(np.isnan(sample.opt_seq))),
            'inf': not bool(np.any(np.isinf(sample.opt_seq))),
            'range': bool(np.all(sample.opt_seq >= -5) and np.all(sample.opt_seq <= 5)),
            'valid_ratio': bool(sample.valid_count.mean() / sample.doy.shape[0] > 0.3
                                if sample.valid_count is not None else True)
        }


class DataReader:
    """数据读取器（向后兼容）"""
    @staticmethod
    def read_data(path: str, modality: str = 'opt') -> np.ndarray:
        path_str = str(path)
        if path_str.endswith('.npy'):
            return np.load(path_str)
        elif path_str.endswith(('.tif', '.tiff')):
            import rasterio
            with rasterio.open(path_str) as src:
                return src.read().astype(np.float32)
        raise ValueError(f"不支持的文件格式: {path_str}")


# ── 向后兼容的 PreprocessPipeline 包装 ──

class PreprocessPipeline:
    """完整预处理管道（向后兼容版本，委托给统一模块）"""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.aligner = SpatialAligner(config.target_resolution)
        self.cloud_detector = CloudDetector(config.cloud_threshold)
        self.interpolator = TemporalInterpolator(
            config.max_gap, config.interpolation_method,
            config.mask_long_gaps, config.long_gap_threshold)
        self.sar_processor = SARProcessor({'sar_log_transform': config.sar_log_transform,
                                           'sar_speckle_method': 'refined_lee'})
        self.normalizer = SpectralNormalizer(config.global_stats_path, config.freeze_stats)
        self.augmenter = DataAugmenter(config.augment_prob)
        self.quality_checker = DataQualityChecker()

    def process(self, raw_data: Dict[str, np.ndarray],
                transforms: Dict[str, dict],
                label: Optional[np.ndarray] = None,
                is_training: bool = False) -> Optional[DataSample]:
        # 1. 空间配准
        aligned = self.aligner.align(raw_data, transforms)
        # 2. 云检测
        cloud_mask = self.cloud_detector.detect(aligned['opt'], aligned.get('sar'))
        cloud_mask = self.cloud_detector.refine_mask(cloud_mask)
        # 3. SAR 对数变换
        sar_dict = self.sar_processor.process({'image': aligned['sar']})
        sar_processed = sar_dict['image']
        # 4. 时序插值
        opt_filled, updated_mask, is_interp, valid_count = self.interpolator.interpolate(
            aligned['opt'],
            raw_data.get('doy', np.arange(aligned['opt'].shape[0])),
            cloud_mask)
        # 5. 增强前样本
        sample = DataSample(
            opt_seq=opt_filled, sar_seq=sar_processed,
            dem=aligned['dem'],
            doy=raw_data.get('doy', np.arange(opt_filled.shape[0])),
            label=label, cloud_mask=updated_mask,
            valid_count=valid_count, is_interpolated=is_interp)
        if is_training and self.config.augment:
            sample = self.augmenter.augment(sample)
        # 6. 归一化
        if self.config.normalize:
            sample.opt_seq = self.normalizer.normalize(sample.opt_seq, 'opt')
            sample.sar_seq = self.normalizer.normalize(sample.sar_seq, 'sar')
            sample.dem = self.normalizer.normalize(sample.dem, 'dem')
        # 7. 质量检查
        quality = self.quality_checker.check(sample)
        if not all(quality.values()):
            return None
        return sample

    def save_global_stats(self, path: str):
        self.normalizer.save_stats(path)

    def set_global_stats(self, stats: Dict[str, dict]):
        self.normalizer.stats = stats
