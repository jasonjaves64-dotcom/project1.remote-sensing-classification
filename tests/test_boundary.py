"""
边界测试 - 极端值、无效输入、边界条件
"""
import pytest
import numpy as np
import torch
from data.preprocessing.base import BasePreprocessor
from data.preprocessing.optical import OpticalProcessor
from data.preprocessing.sar import SARProcessor
from data.preprocessing.quality import QualityControl
from utils.losses import DiceFocalLoss, WeightedDiceFocalLoss, TverskyLoss, OhemCELoss
from utils.augment import randomFlip, randomTemporalDropout, randomNoise, TemporalAugment, SpatialAugment, Compose


class TestEmptyAndNoneInputs:
    """空值和None输入测试"""

    def test_optical_none_input(self):
        processor = OpticalProcessor()
        with pytest.raises(ValueError, match="输入数据不能为空"):
            processor.process(None)

    def test_sar_none_input(self):
        processor = SARProcessor()
        with pytest.raises(ValueError, match="输入数据不能为空"):
            processor.process(None)

    def test_ndvi_zero_denominator(self):
        processor = OpticalProcessor()
        nir = np.zeros((10, 10), dtype=np.float32)
        red = np.zeros((10, 10), dtype=np.float32)
        ndvi = processor.calculate_ndvi(nir, red)
        assert np.all(np.isfinite(ndvi))
        assert ndvi.shape == (10, 10)

    def test_evi_zero_denominator(self):
        processor = OpticalProcessor()
        nir = np.zeros((10, 10), dtype=np.float32)
        red = np.zeros((10, 10), dtype=np.float32)
        blue = np.zeros((10, 10), dtype=np.float32)
        evi = processor.calculate_evi(nir, red, blue)
        assert np.all(np.isfinite(evi))

    def test_sar_db_zero_input(self):
        processor = SARProcessor()
        data = np.zeros((10, 10), dtype=np.float32)
        result = processor.normalize_db(data)
        assert result.shape == (10, 10)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_quality_empty_data(self):
        qc = QualityControl()
        data = np.array([], dtype=np.float32).reshape(0, 5)
        report = qc.check_validity(data)
        assert report["total_pixels"] == 0


class TestExtremeValues:
    """极端数值测试"""

    def test_ndvi_extreme_nir_large(self):
        processor = OpticalProcessor()
        nir = np.full((10, 10), 1e8, dtype=np.float32)
        red = np.full((10, 10), 1e-8, dtype=np.float32)
        ndvi = processor.calculate_ndvi(nir, red)
        assert np.all(ndvi >= -1) and np.all(ndvi <= 1)

    def test_ndvi_negative_inputs(self):
        processor = OpticalProcessor()
        nir = np.full((10, 10), -1.0, dtype=np.float32)
        red = np.full((10, 10), -2.0, dtype=np.float32)
        ndvi = processor.calculate_ndvi(nir, red)
        assert np.all(np.isfinite(ndvi))

    def test_normalize_uniform_data(self):
        processor = OpticalProcessor()
        data = np.full((10, 10), 0.5, dtype=np.float32)
        result = processor.normalize(data)
        assert result.shape == (10, 10)

    def test_sar_single_value(self):
        processor = SARProcessor()
        data = np.full((5, 5), 1.0, dtype=np.float32)
        result = processor.normalize_db(data)
        assert result.shape == (5, 5)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_qc_statistics_all_nan(self):
        qc = QualityControl()
        data = np.full((10, 10), np.nan, dtype=np.float32)
        stats = qc.generate_statistics(data)
        assert "mean" in stats
        assert "std" in stats

    def test_qc_outlier_detection_zero_std(self):
        qc = QualityControl()
        data = np.ones((10, 10), dtype=np.float32)
        outliers = qc.detect_outliers(data, z_threshold=3.0)
        assert np.all(outliers == 0)


class TestBoundaryShapes:
    """边界形状测试"""

    def test_ndvi_single_pixel(self):
        processor = OpticalProcessor()
        nir = np.array([[0.5]], dtype=np.float32)
        red = np.array([[0.3]], dtype=np.float32)
        ndvi = processor.calculate_ndvi(nir, red)
        assert ndvi.shape == (1, 1)
        assert -1 <= ndvi[0, 0] <= 1

    def test_speckle_filter_single_pixel(self):
        processor = SARProcessor()
        data = np.ones((1, 1), dtype=np.float32)
        result = processor.apply_speckle_filter(data, window_size=3)
        assert result.shape == (1, 1)

    def test_speckle_filter_even_window(self):
        processor = SARProcessor()
        data = np.random.rand(32, 32).astype(np.float32)
        result = processor.apply_speckle_filter(data, window_size=4)
        assert result.shape == (32, 32)

    def test_qc_validity_1d_data(self):
        qc = QualityControl()
        data = np.array([1.0, 2.0, 3.0, np.nan, 5.0], dtype=np.float32)
        report = qc.check_validity(data)
        assert report["invalid_pixels"] >= 1

    def test_normalize_extreme_range(self):
        processor = OpticalProcessor()
        data = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
        result = processor.normalize(data)
        assert np.all(result >= 0) and np.all(result <= 1)


class TestLossBoundaries:
    """损失函数边界测试"""

    def test_focal_loss_all_background(self):
        loss_fn = DiceFocalLoss(num_classes=7, ignore_index=0)
        logits = torch.randn(2, 7, 32, 32)
        targets = torch.zeros(2, 32, 32, dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_focal_loss_single_class(self):
        loss_fn = DiceFocalLoss(num_classes=2, ignore_index=0)
        logits = torch.randn(2, 2, 32, 32)
        targets = torch.ones(2, 32, 32, dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_ohem_loss_all_valid(self):
        loss_fn = OhemCELoss(num_classes=7, ignore_index=0, ohem_ratio=0.1)
        logits = torch.randn(2, 7, 32, 32)
        targets = torch.randint(1, 7, (2, 32, 32))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_ohem_loss_all_ignored(self):
        loss_fn = OhemCELoss(num_classes=7, ignore_index=0)
        logits = torch.randn(2, 7, 32, 32)
        targets = torch.zeros(2, 32, 32, dtype=torch.long)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_tversky_extreme_alpha(self):
        loss_fn = TverskyLoss(alpha=0.99, beta=0.01, num_classes=7)
        logits = torch.randn(2, 7, 32, 32)
        targets = torch.randint(1, 7, (2, 32, 32))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)

    def test_weighted_focal_with_map(self):
        loss_fn = WeightedDiceFocalLoss(num_classes=7)
        logits = torch.randn(2, 7, 32, 32)
        targets = torch.randint(1, 7, (2, 32, 32))
        weight_map = torch.rand(2, 32, 32)
        loss = loss_fn(logits, targets, weight_map)
        assert torch.isfinite(loss)

    def test_weighted_focal_without_map(self):
        loss_fn = WeightedDiceFocalLoss(num_classes=7)
        logits = torch.randn(2, 7, 32, 32)
        targets = torch.randint(1, 7, (2, 32, 32))
        loss = loss_fn(logits, targets, None)
        assert torch.isfinite(loss)

    def test_tversky_with_weight_map(self):
        loss_fn = TverskyLoss(num_classes=7)
        logits = torch.randn(2, 7, 32, 32)
        targets = torch.randint(1, 7, (2, 32, 32))
        weight_map = torch.ones(2, 32, 32)
        loss = loss_fn(logits, targets, weight_map)
        assert torch.isfinite(loss)


class TestAugmentationBoundaries:
    """数据增强边界测试"""

    def test_randomFlip_identity(self):
        x = np.ones((3, 32, 32), dtype=np.float32)
        y = np.ones((32, 32), dtype=np.int64)
        xf, yf = randomFlip(x, y, h_prob=0.0, v_prob=0.0)
        np.testing.assert_array_equal(x, xf)
        np.testing.assert_array_equal(y, yf)

    def test_randomTemporalDropout_zero(self):
        x = np.ones((12, 10, 32, 32), dtype=np.float32)
        x_out = randomTemporalDropout(x, max_drop=0)
        np.testing.assert_array_equal(x, x_out)

    def test_randomNoise_zero_sigma(self):
        x = np.ones((3, 32, 32), dtype=np.float32)
        x_out = randomNoise(x, sigma=0.0)
        np.testing.assert_array_equal(x, x_out)

    def test_temporal_augment_no_dropout(self):
        aug = TemporalAugment(p_temporal_dropout=0.0)
        x = np.ones((12, 10, 32, 32), dtype=np.float32)
        x_out = aug(x)
        np.testing.assert_array_equal(x, x_out)

    def test_spatial_augment_no_flip_no_noise(self):
        aug = SpatialAugment(p_flip=0.0, p_noise=0.0)
        x = np.ones((3, 32, 32), dtype=np.float32)
        y = np.ones((32, 32), dtype=np.int64)
        x_out, y_out = aug(x, y)
        np.testing.assert_array_equal(x, x_out)

    def test_compose_empty(self):
        aug = Compose([])
        x = np.ones((3, 32, 32), dtype=np.float32)
        x_out = aug(x)
        np.testing.assert_array_equal(x, x_out)
