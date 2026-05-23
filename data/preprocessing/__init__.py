"""
预处理模块
"""
from .base import (BasePreprocessor, PreprocessConfig, DataSample, DataValidator,
                   load_config, save_config, batch_generator,
                   load_or_compute_memmap, create_memmap, sliding_window_generator)
from .optical import (OpticalProcessor, GeoTIFFReader, CloudMaskProcessor,
                      compute_spectral_indices, DataNormalizer,
                      BAND_NAMES, BAND_IDX, LS_SCALE, LS_OFFSET)
from .sar import SARProcessor, SAROpticalAligner
from .temporal import TemporalInterpolator, TemporalAligner
from .terrain import TerrainCorrector
from .dem_features import DEMFeatureExtractor
from .outlier import OutlierDetector
from .spatial import SpatialDataSplitter
from .label import LabelProcessor, CROP_CLASSES
from .augment import DataAugmenter
from .fusion import MultiModalFusion
from .quality import QualityControl, PreprocessingQualityReport
from .pipeline import UnifiedPreprocessingPipeline

__all__ = [
    # base
    "BasePreprocessor", "PreprocessConfig", "DataSample", "DataValidator",
    "load_config", "save_config", "batch_generator",
    "load_or_compute_memmap", "create_memmap", "sliding_window_generator",
    # optical
    "OpticalProcessor", "GeoTIFFReader", "CloudMaskProcessor",
    "compute_spectral_indices", "DataNormalizer",
    "BAND_NAMES", "BAND_IDX", "LS_SCALE", "LS_OFFSET",
    # sar
    "SARProcessor", "SAROpticalAligner",
    # temporal
    "TemporalInterpolator", "TemporalAligner",
    # terrain
    "TerrainCorrector",
    # dem
    "DEMFeatureExtractor",
    # outlier
    "OutlierDetector",
    # spatial
    "SpatialDataSplitter",
    # label
    "LabelProcessor", "CROP_CLASSES",
    # augment
    "DataAugmenter",
    # fusion
    "MultiModalFusion",
    # quality
    "QualityControl", "PreprocessingQualityReport",
    # pipeline
    "UnifiedPreprocessingPipeline",
]
