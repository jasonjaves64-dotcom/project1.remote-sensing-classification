"""
完整预处理入口 - YAML驱动的薄包装
委托给统一预处理模块（UnifiedPreprocessingPipeline, mode="full"）
"""
import os
import numpy as np
from data.preprocessing.base import PreprocessConfig
from data.preprocessing.pipeline import UnifiedPreprocessingPipeline


class PreprocessingPipeline:
    """完整预处理流水线（薄包装，委托给统一模块）"""

    def __init__(self, config: dict):
        self.cfg = PreprocessConfig.from_dict(config)

    def run(self) -> dict:
        pipeline = UnifiedPreprocessingPipeline(self.cfg, mode="full")
        return pipeline.run()


if __name__ == "__main__":
    config = {
        "landsat_dir": "./data/raw/landsat",
        "sar_dir": "./data/raw/sentinel1",
        "label_shp": "./data/labels/crop_parcels_2023.shp",
        "class_field": "crop_type",
        "output_dir": "./data/processed",
        "max_cloud_pct": 0.5,
        "erosion_pixels": 2,
        "norm_method": "robust",
    }
    pipeline = PreprocessingPipeline(config)
    results = pipeline.run()
