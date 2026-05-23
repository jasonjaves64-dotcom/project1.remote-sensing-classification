"""
预处理管道端到端测试 - 完整预处理流程验证
"""
import pytest
import numpy as np
import torch
from data.preprocessing.base import BasePreprocessor
from data.preprocessing.optical import OpticalProcessor
from data.preprocessing.sar import SARProcessor
from data.preprocessing.quality import QualityControl


class TestOpticalPipelineE2E:
    """光学预处理端到端测试"""

    def test_optical_full_flow_with_qa(self):
        processor = OpticalProcessor()
        nir = np.random.rand(32, 32).astype(np.float32) * 0.5 + 0.2
        red = np.random.rand(32, 32).astype(np.float32) * 0.3 + 0.1
        blue = np.random.rand(32, 32).astype(np.float32) * 0.4 + 0.1
        image = np.stack([blue, np.zeros_like(blue), np.zeros_like(blue),
                          red, np.zeros_like(red), np.zeros_like(red),
                          np.zeros_like(red), nir], axis=-1)
        qa_pixel = np.zeros((32, 32), dtype=np.uint16)
        data = {'image': image, 'qa_pixel': qa_pixel}
        result = processor.process(data)
        assert 'ndvi' in result
        assert 'evi' in result
        assert 'mask' in result
        assert result['mask'].shape == (32, 32)

    def test_optical_flow_without_qa(self):
        processor = OpticalProcessor()
        image = np.random.rand(32, 32, 8).astype(np.float32)
        data = {'image': image}
        result = processor.process(data)
        assert 'ndvi' in result
        assert 'evi' in result

    def test_optical_normalize_output_range(self):
        processor = OpticalProcessor()
        data = np.random.rand(100, 100).astype(np.float32) * 100 - 50
        result = processor.normalize(data, min_val=0.0, max_val=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_ndvi_range_for_valid_pixels(self):
        processor = OpticalProcessor()
        nir = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        red = np.linspace(0.1, 0.5, 100).reshape(10, 10).astype(np.float32)
        ndvi = processor.calculate_ndvi(nir, red)
        assert np.all(ndvi >= -1.0)
        assert np.all(ndvi <= 1.0)


class TestSARPipelineE2E:
    """SAR预处理端到端测试"""

    def test_sar_full_flow(self):
        processor = SARProcessor()
        image = np.abs(np.random.randn(32, 32)).astype(np.float32) * 0.3 + 0.1
        data = {'image': image}
        result = processor.process(data)
        assert result['image'].shape == (32, 32)
        assert result['image'].min() >= 0.0
        assert result['image'].max() <= 1.0

    def test_sar_db_normalization_range(self):
        processor = SARProcessor()
        data = np.abs(np.random.randn(64, 64)).astype(np.float32) * 0.5 + 0.05
        result = processor.normalize_db(data)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert np.all(np.isfinite(result))

    def test_speckle_filter_preserves_mean(self):
        processor = SARProcessor()
        np.random.seed(42)
        data = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.25
        filtered = processor.apply_speckle_filter(data, window_size=5)
        assert abs(data.mean() - filtered.mean()) < 0.15

    def test_orthorectify_no_change(self):
        processor = SARProcessor()
        data = np.random.rand(32, 32).astype(np.float32)
        result = processor.orthorectify(data)
        np.testing.assert_array_equal(data, result)


class TestQualityPipelineE2E:
    """质量控制端到端测试"""

    def test_full_qc_report(self):
        qc = QualityControl()
        data = np.random.rand(10, 64, 64).astype(np.float32)
        report = qc.generate_report(data, dataset_name="test_dataset")
        assert report['dataset_name'] == "test_dataset"
        assert list(report['shape']) == [10, 64, 64]
        assert 'validity' in report
        assert 'statistics' in report
        assert 'outlier_count' in report

    def test_qc_validity_all_valid(self):
        qc = QualityControl()
        data = np.ones((50, 50), dtype=np.float32)
        report = qc.check_validity(data)
        assert report['invalid_pixels'] == 0
        assert report['valid_percentage'] == 100.0

    def test_qc_statistics_completeness(self):
        qc = QualityControl()
        data = np.random.rand(20, 20).astype(np.float32)
        stats = qc.generate_statistics(data)
        for key in ['mean', 'std', 'min', 'max', 'median', 'percentiles']:
            assert key in stats, f"Missing key: {key}"
        for p in ['p5', 'p25', 'p50', 'p75', 'p95']:
            assert p in stats['percentiles'], f"Missing percentile: {p}"

    def test_qc_outlier_default_threshold(self):
        qc = QualityControl()
        data = np.random.randn(1000).astype(np.float32)
        data[0] = 100.0
        outliers = qc.detect_outliers(data, z_threshold=3.0)
        assert outliers[0] == True
        assert np.sum(outliers) < len(data) * 0.1

    def test_qc_report_dtype_preservation(self):
        qc = QualityControl()
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        report = qc.generate_report(data, "dtype_test")
        assert 'float32' in report['dtype']


class TestPreprocessingIntegration:
    """预处理模块集成测试"""

    def test_optical_to_qc_integration(self):
        opt = OpticalProcessor()
        qc = QualityControl()
        nir = np.random.rand(32, 32).astype(np.float32) * 0.5 + 0.2
        red = np.random.rand(32, 32).astype(np.float32) * 0.3 + 0.1
        ndvi = opt.calculate_ndvi(nir, red)
        report = qc.generate_report(ndvi, "ndvi")
        assert report['validity']['valid_percentage'] >= 99.0

    def test_sar_to_qc_integration(self):
        sar = SARProcessor()
        qc = QualityControl()
        data = np.abs(np.random.randn(64, 64)).astype(np.float32) * 0.3 + 0.1
        processed = sar.normalize_db(data)
        report = qc.check_validity(processed)
        assert report['invalid_pixels'] == 0

    def test_config_update_flow(self):
        processor = OpticalProcessor({'cloud_bit': 10})
        assert processor.get_config()['cloud_bit'] == 10
        processor.update_config(cloud_bit=12, new_param=True)
        assert processor.get_config()['cloud_bit'] == 12
        assert processor.get_config()['new_param'] == True
