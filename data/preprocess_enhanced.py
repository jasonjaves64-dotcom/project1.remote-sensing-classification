"""
增强版预处理入口 - YAML驱动的薄包装
委托给统一预处理模块（UnifiedPreprocessingPipeline, mode="enhanced"）
"""
import os
import numpy as np
from data.preprocessing.base import PreprocessConfig, load_config
from data.preprocessing.pipeline import UnifiedPreprocessingPipeline


class EnhancedPreprocessingPipeline:
    """增强版预处理流水线（薄包装，委托给统一模块）"""

    def __init__(self, config: dict):
        self.cfg = PreprocessConfig.from_dict(config)

    def run(self) -> dict:
        pipeline = UnifiedPreprocessingPipeline(self.cfg, mode="enhanced")
        return pipeline.run()


if __name__ == "__main__":
    config = {
        "landsat_dir": "./data/raw/landsat",
        "sar_dir": "./data/raw/sentinel1",
        "label_shp": "./data/labels/crop_parcels_2023.shp",
        "class_field": "crop_type",
        "year": 2023,
        "apply_terrain_correction": False,
        "dem_path": "./data/dem/dem_30m.tif",
        "solar_zenith": 30.0,
        "solar_azimuth": 150.0,
        "output_dir": "./data/processed",
        "max_cloud_pct": 0.5,
        "max_gap_days": 16,
        "outlier_z_thresh": 3.5,
        "temporal_diff_thresh": 0.3,
        "erosion_pixels": 2,
        "norm_method": "robust",
        "split_block_size": 64,
    }
    pipeline = EnhancedPreprocessingPipeline(config)
    results = pipeline.run()
