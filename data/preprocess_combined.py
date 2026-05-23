"""
融合版预处理入口 - YAML驱动的薄包装
委托给统一预处理模块（UnifiedPreprocessingPipeline, mode="combined"）
"""
import os
import numpy as np
from data.preprocessing.base import PreprocessConfig
from data.preprocessing.pipeline import UnifiedPreprocessingPipeline


class CombinedPreprocessingPipeline:
    """融合版预处理流水线（薄包装，委托给统一模块）"""

    def __init__(self, config: dict):
        self.cfg = PreprocessConfig.from_dict(config)

    def run(self) -> dict:
        pipeline = UnifiedPreprocessingPipeline(self.cfg, mode="combined")
        return pipeline.run()


if __name__ == "__main__":
    config = {
        "landsat_dir": "./data/raw/landsat",
        "sar_dir": "./data/raw/sentinel1",
        "dem_path": "./data/dem/dem_30m.tif",
        "label_shp": "./data/labels/crop_parcels_2023.shp",
        "output_dir": "./data/processed",
        "apply_terrain_correction": False,
        "max_cloud_pct": 0.5,
        "split_block_size": 64,
        "erosion_pixels": 2,
        "fusion_mode": "concat",
        "enable_supplement": True,
    }
    pipeline = CombinedPreprocessingPipeline(config)
    results = pipeline.run()
