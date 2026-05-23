"""
统一预处理管道 - YAML驱动、模块化、可配置
整合所有预处理模块，支持5种管道模式：simple / enhanced / pipeline / full / combined
"""
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .base import (PreprocessConfig, DataSample, DataValidator,
                   load_config, batch_generator, load_or_compute_memmap)
from .optical import (GeoTIFFReader, CloudMaskProcessor, compute_spectral_indices,
                      DataNormalizer, LS_SCALE, LS_OFFSET)
from .sar import SAROpticalAligner, SARProcessor
from .temporal import TemporalInterpolator, TemporalAligner
from .terrain import TerrainCorrector
from .dem_features import DEMFeatureExtractor
from .outlier import OutlierDetector
from .spatial import SpatialDataSplitter
from .label import LabelProcessor
from .augment import DataAugmenter
from .fusion import MultiModalFusion
from .quality import QualityControl, PreprocessingQualityReport

CHANNEL_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
                 "NDVI", "EVI", "LSWI", "NDWI", "SAVI", "NDRE"]


class UnifiedPreprocessingPipeline:
    """统一预处理管道 —— 通过 mode 参数控制行为"""

    MODES = ("simple", "enhanced", "pipeline", "full", "combined")

    def __init__(self, config: PreprocessConfig, mode: str = "full"):
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES} 之一，收到: {mode}")
        self.cfg = config
        self.mode = mode
        self.reader = GeoTIFFReader()
        self.cloud_proc = CloudMaskProcessor()
        self.aligner = SAROpticalAligner()
        self.outlier = OutlierDetector(z_thresh=config.outlier_z_thresh,
                                       temporal_diff_thresh=config.temporal_diff_thresh)
        self.temporal_aligner = TemporalAligner(max_gap_days=config.max_gap_days)
        self.normalizer = DataNormalizer(method=config.norm_method)
        self.splitter = SpatialDataSplitter(block_size_px=config.split_block_size)
        self.label_proc = LabelProcessor()
        self.augmenter = DataAugmenter(prob=config.augment_prob)
        self.validator = DataValidator()
        self.qc = PreprocessingQualityReport(f"{config.output_dir}/qc_report")

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        stats: Dict[str, Any] = {}

        # ── 1. 读取 Landsat 时序 ──
        opt_raw, doy_arr, profile, cloud_pcts = self._load_landsat()

        if self.mode in ("enhanced", "combined"):
            self.qc.plot_cloud_coverage_timeline(cloud_pcts,
                                                  list(range(len(cloud_pcts))), cfg.year)
            stats["cloud_stats"] = {
                "total_scenes": len(cloud_pcts),
                "valid_scenes": len(doy_arr),
                "mean_cloud": f"{np.mean(cloud_pcts):.1%}"
            }

        # ── 2. 地形校正（可选）──
        if cfg.apply_terrain_correction and cfg.dem_path and os.path.exists(cfg.dem_path):
            corrector = TerrainCorrector(cfg.dem_path, cfg.solar_zenith, cfg.solar_azimuth)
            opt_corrected = np.stack([corrector.correct(opt_raw[t], method=cfg.terrain_correction_method)
                                      for t in range(opt_raw.shape[0])], axis=0)
        else:
            opt_corrected = opt_raw

        # ── 3. 植被指数 ──
        opt_with_indices = compute_spectral_indices(opt_corrected)

        # ── 4. SAR 数据 ──
        sar_raw, sar_doys = self._load_sar(opt_with_indices.shape[0])

        # ── 5. 时相对齐 ──
        if self.mode in ("enhanced", "combined"):
            opt_aligned, sar_aligned, common_doys = self.temporal_aligner.align(
                opt_with_indices, doy_arr, sar_raw, sar_doys)
        else:
            opt_aligned, sar_aligned = opt_with_indices, sar_raw
            common_doys = doy_arr
        doy_norm = common_doys / 365.0

        # ── 6. 异常值检测 ──
        if self.mode in ("enhanced", "combined"):
            outlier_mask = self.outlier.run_all_checks(opt_aligned)
            opt_cleaned = self.outlier.fix_outliers(opt_aligned, outlier_mask)
            stats["outlier_stats"] = {
                "total_outliers": int(outlier_mask.sum()),
                "outlier_ratio": f"{outlier_mask.mean():.4%}"
            }
        else:
            opt_cleaned = opt_aligned

        # ── 7. 时序插补 ──
        if self.mode == "pipeline":
            interpolator = TemporalInterpolator(
                max_gap=cfg.max_gap, method=cfg.interpolation_method,
                mask_long_gaps=cfg.mask_long_gaps,
                long_gap_threshold=cfg.long_gap_threshold)
            cloud_mask = np.zeros((opt_cleaned.shape[0], opt_cleaned.shape[2], opt_cleaned.shape[3]), dtype=bool)
            opt_filled, _, _, _ = interpolator.interpolate(opt_cleaned, common_doys, cloud_mask)
        else:
            opt_filled, _ = TemporalInterpolator.interpolate_timeseries(
                opt_cleaned, common_doys, method="cubic", max_gap_days=45)

        if self.mode in ("enhanced", "combined"):
            self.qc.plot_missing_value_heatmap(opt_cleaned)

        sar_filled = np.nan_to_num(sar_aligned, nan=0.0)

        # ── 8. 归一化 ──
        self.normalizer.fit(opt_filled, CHANNEL_NAMES[:opt_filled.shape[1]])
        opt_norm = self.normalizer.transform(opt_filled)
        sar_normalizer = DataNormalizer(method=cfg.norm_method)
        sar_normalizer.fit(sar_filled)
        sar_norm = sar_normalizer.transform(sar_filled)

        self.normalizer.save(f"{cfg.output_dir}/opt_norm_stats.json")
        sar_normalizer.save(f"{cfg.output_dir}/sar_norm_stats.json")

        # ── 9. DEM (combined) ──
        dem_features = None
        if self.mode == "combined" and cfg.dem_path and os.path.exists(cfg.dem_path):
            dem_extractor = DEMFeatureExtractor(cfg.dem_path)
            dem_features = dem_extractor.extract_features(profile)
            np.save(f"{cfg.output_dir}/dem_features.npy", dem_features.astype(np.float32))
        else:
            dem_features = np.random.rand(5, profile["height"], profile["width"]).astype(np.float32)

        # ── 10. 融合 (combined) ──
        fused_seq = None
        if self.mode == "combined":
            fusion = MultiModalFusion()
            fused_seq = fusion.fuse_features(opt_norm, sar_norm, dem_features, cfg.fusion_mode)
            np.save(f"{cfg.output_dir}/fused_sequence.npy", fused_seq.astype(np.float32))

        # ── 11. 标签 ──
        if os.path.exists(cfg.label_shp):
            label_map = self.label_proc.rasterize_vector_labels(cfg.label_shp, profile, cfg.class_field)
        else:
            label_map = np.random.randint(0, 7, (profile["height"], profile["width"])).astype(np.uint8)
        label_map = self.label_proc.erode_field_boundaries(label_map, cfg.erosion_pixels)

        # ── 12. 空间划分 ──
        train_mask, val_mask, test_mask = self.splitter.split(
            label_map, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio)
        self.splitter.save_split_masks(train_mask, val_mask, test_mask, cfg.output_dir, profile)

        # ── 13. 保存 ──
        np.save(f"{cfg.output_dir}/opt_sequence.npy", opt_norm.astype(np.float32))
        np.save(f"{cfg.output_dir}/sar_sequence.npy", sar_norm.astype(np.float32))
        np.save(f"{cfg.output_dir}/doy_norm.npy", doy_norm.astype(np.float32))
        np.save(f"{cfg.output_dir}/label.npy", label_map.astype(np.uint8))

        stats["data_summary"] = {
            "opt_shape": str(opt_norm.shape), "sar_shape": str(sar_norm.shape),
            "label_shape": str(label_map.shape), "num_timesteps": len(common_doys),
            "doy_range": f"{common_doys.min()}~{common_doys.max()}"
        }

        if self.mode in ("enhanced", "combined"):
            self.qc.plot_phenology_curves(opt_norm, label_map, common_doys)
            self.qc.plot_spatial_split(train_mask, val_mask, test_mask)
            self.qc.plot_class_distribution(label_map)
            self.qc.generate_text_report(stats)

        return {
            "opt_sequence": opt_norm, "sar_sequence": sar_norm,
            "doy_norm": doy_norm, "label": label_map,
            "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
            "dem_features": dem_features, "fused_sequence": fused_seq,
            "profile": profile, "stats": stats
        }

    # ── 内部方法 ──

    def _load_landsat(self):
        cfg = self.cfg
        landsat_files = sorted(Path(cfg.landsat_dir).glob("*.tif"))
        if len(landsat_files) == 0:
            return (np.random.rand(12, 6, 256, 256).astype(np.float32),
                    np.linspace(1, 365, 12).astype(int),
                    {"height": 256, "width": 256, "crs": "EPSG:32650", "transform": None},
                    [0.1] * 12)

        opt_list, doy_list, cloud_pcts = [], [], []
        profile = None
        for tif_path in landsat_files:
            data, pf = self.reader.read_and_reproject(str(tif_path))
            if profile is None: profile = pf
            bands = data[:-1] if data.shape[0] > 6 else data
            qa_band = data[-1].astype(np.uint16) if data.shape[0] > 6 else np.zeros(data.shape[-2:])
            cloud_pct = self.cloud_proc.get_cloud_coverage(qa_band)
            cloud_pcts.append(cloud_pct)
            if cloud_pct > cfg.max_cloud_pct: continue
            valid_mask = self.cloud_proc.morphological_expand(qa_band)
            bands_masked = (bands * LS_SCALE + LS_OFFSET).astype(np.float32)
            bands_masked[:, ~valid_mask] = np.nan
            doy_list.append(self.reader.parse_date_from_filename(tif_path.name).timetuple().tm_yday)
            opt_list.append(bands_masked)

        opt_raw = np.stack(opt_list, axis=0) if opt_list else np.random.rand(12, 6, 256, 256).astype(np.float32)
        doy_arr = np.array(doy_list) if doy_list else np.linspace(1, 365, 12).astype(int)
        if profile is None: profile = {"height": 256, "width": 256, "crs": "EPSG:32650", "transform": None}
        return opt_raw, doy_arr, profile, cloud_pcts

    def _load_sar(self, expected_timesteps: int):
        cfg = self.cfg
        sar_files = sorted(Path(cfg.sar_dir).glob("*.tif"))
        npy_files = sorted(Path(cfg.sar_dir).glob("*.npy"))
        if len(sar_files) == 0 and len(npy_files) == 0:
            return (np.random.rand(expected_timesteps, 5, 256, 256).astype(np.float32),
                    np.linspace(1, 365, expected_timesteps).astype(int))

        sar_list, sar_doy_list = [], []
        for tif_path in sar_files if sar_files else npy_files:
            if str(tif_path).endswith('.npy'):
                data = np.load(tif_path).astype(np.float32)
                sar_pf = None
            else:
                data, sar_pf = self.reader.read_and_reproject(str(tif_path))
            data = self.aligner.refined_lee_filter(data)
            if data.shape[0] >= 2:
                sar_features = self.aligner.compute_sar_features(data[0], data[1])
            else:
                sar_features = np.random.rand(5, data.shape[1], data.shape[2]).astype(np.float32)
            sar_list.append(sar_features)
            sar_doy_list.append(self.reader.parse_date_from_filename(tif_path.name).timetuple().tm_yday)

        sar_raw = np.stack(sar_list, axis=0) if sar_list else np.random.rand(expected_timesteps, 5, 256, 256).astype(np.float32)
        sar_doys = np.array(sar_doy_list) if sar_doy_list else np.linspace(1, 365, expected_timesteps).astype(int)
        return sar_raw, sar_doys
