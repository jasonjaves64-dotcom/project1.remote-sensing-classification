"""
模型导出往返测试 - ONNX/TorchScript/EMD/DLPK 导出与验证
"""
import pytest
import os
import json
import tempfile
import numpy as np
import torch
from pathlib import Path

from utils.export import (
    export_to_onnx, export_to_torchscript, export_to_emd,
    export_to_dlpk, export_model
)
from utils.metrics import compute_metrics


class SimpleTestModel(torch.nn.Module):
    """用于导出测试的轻量模型"""

    def __init__(self, opt_ch=10, sar_ch=5, dem_ch=5, num_classes=7):
        super().__init__()
        self.opt_conv = torch.nn.Conv2d(opt_ch, 32, 3, padding=1)
        self.sar_conv = torch.nn.Conv2d(sar_ch, 32, 3, padding=1)
        self.dem_conv = torch.nn.Conv2d(dem_ch, 16, 3, padding=1)
        self.final = torch.nn.Conv2d(80, num_classes, 1)

    def forward(self, opt, sar, dem, doy=None):
        B, T, C_opt, H, W = opt.shape
        opt_feat = self.opt_conv(opt[:, -1])
        B2, T_s, C_sar, H_s, W_s = sar.shape
        sar_feat = self.sar_conv(sar[:, -1])
        if dem.dim() == 2:
            dem_feat = self.dem_conv(torch.randn(B, 5, H, W, device=opt.device))
        else:
            dem_feat = self.dem_conv(dem)
        combined = torch.cat([opt_feat, sar_feat, dem_feat], dim=1)
        return self.final(combined)


def _make_dummy_inputs(batch_size=1, seq_len=12, H=32, W=32):
    opt = torch.randn(batch_size, seq_len, 10, H, W)
    sar = torch.randn(batch_size, seq_len, 5, H, W)
    dem = torch.randn(batch_size, 5, H, W)
    doy = torch.randn(batch_size, seq_len)
    return opt, sar, dem, doy


class TestONNXExport:
    """ONNX导出测试"""

    def test_onnx_export_creates_file(self):
        model = SimpleTestModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.onnx")
            export_to_onnx(model, path, opt_channels=10, sar_channels=5,
                           dem_channels=5, seq_len=12, patch_size=32)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_onnx_export_different_sizes(self):
        model = SimpleTestModel(opt_ch=6, sar_ch=2, dem_ch=3, num_classes=3)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_small.onnx")
            export_to_onnx(model, path, opt_channels=6, sar_channels=2,
                           dem_channels=3, seq_len=6, patch_size=16)
            assert os.path.exists(path)

    def test_onnx_reload_and_check(self):
        model = SimpleTestModel()
        model.eval()
        opt, sar, dem, doy = _make_dummy_inputs()
        with torch.no_grad():
            orig_out = model(opt, sar, dem, doy)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            export_to_onnx(model, path, opt_channels=10, sar_channels=5,
                           dem_channels=5, seq_len=12, patch_size=32)
            import onnx
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)


class TestTorchScriptExport:
    """TorchScript导出测试"""

    def test_torchscript_export_creates_file(self):
        model = SimpleTestModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            export_to_torchscript(model, path, opt_channels=10, sar_channels=5,
                                  seq_len=12, patch_size=32)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_torchscript_reload_and_infer(self):
        model = SimpleTestModel()
        model.eval()
        opt, sar, dem, doy = _make_dummy_inputs()
        with torch.no_grad():
            orig_out = model(opt, sar, dem, doy)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            export_to_torchscript(model, path, opt_channels=10,
                                  sar_channels=5, seq_len=12, patch_size=32)
            loaded = torch.jit.load(path)
            loaded.eval()
            with torch.no_grad():
                ts_opt = torch.randn(1, 12, 10, 32, 32)
                ts_sar = torch.randn(1, 12, 5, 32, 32)
                ts_doy = torch.randn(1, 12)
                ts_out = loaded(ts_opt, ts_sar, ts_doy)
            assert ts_out.shape[0] == 1


class TestEMDExport:
    """EMD导出测试"""

    def test_emd_export_creates_file(self):
        model = SimpleTestModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.emd")
            export_to_emd(model, path, opt_channels=10, sar_channels=5,
                          dem_channels=5, seq_len=12, patch_size=32, num_classes=7)
            assert os.path.exists(path)
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            assert content['Framework'] == 'PyTorch'
            assert 'ModelConfiguration' in content
            assert content['ModelConfiguration']['NumberOfClasses'] == 7

    def test_emd_custom_classes(self):
        model = SimpleTestModel(num_classes=3)
        custom = {0: "bg", 1: "crop_a", 2: "crop_b"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "custom.emd")
            export_to_emd(model, path, num_classes=3, crop_classes=custom)
            with open(path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            assert len(content['ModelConfiguration']['Classes']) == 3


class TestDLPKExport:
    """DLPK导出测试"""

    def test_dlpk_export_creates_file(self):
        model = SimpleTestModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.dlpk")
            export_to_dlpk(model, path, opt_channels=10, sar_channels=5,
                           dem_channels=5, seq_len=12, patch_size=32, num_classes=7)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_dlpk_contains_onnx_and_emd(self):
        model = SimpleTestModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.dlpk")
            export_to_dlpk(model, path, opt_channels=10, sar_channels=5,
                           dem_channels=5, seq_len=12, patch_size=32)
            import zipfile
            with zipfile.ZipFile(path, 'r') as zf:
                names = zf.namelist()
            assert 'model.onnx' in names or any('onnx' in n for n in names)
            assert 'model.emd' in names or any('emd' in n for n in names)


class TestExportModelFunction:
    """export_model 批量导出测试"""

    def test_export_model_onnx_only(self):
        model = SimpleTestModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(model, export_dir=tmpdir, formats=["onnx"],
                         opt_channels=10, sar_channels=5, dem_channels=5,
                         seq_len=12, patch_size=32)
            assert os.path.exists(os.path.join(tmpdir, "model.onnx"))

    def test_export_model_emd_only(self):
        model = SimpleTestModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(model, export_dir=tmpdir, formats=["emd"],
                         opt_channels=10, sar_channels=5, dem_channels=5,
                         seq_len=12, patch_size=32, num_classes=7)
            assert os.path.exists(os.path.join(tmpdir, "model.emd"))

    def test_export_model_multiple_formats(self):
        model = SimpleTestModel()
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_model(model, export_dir=tmpdir,
                         formats=["onnx", "torchscript", "emd"],
                         opt_channels=10, sar_channels=5, dem_channels=5,
                         seq_len=12, patch_size=32, num_classes=7)
            assert os.path.exists(os.path.join(tmpdir, "model.onnx"))
            assert os.path.exists(os.path.join(tmpdir, "model.pt"))
            assert os.path.exists(os.path.join(tmpdir, "model.emd"))


class TestMetricsRoundTrip:
    """指标计算往返验证"""

    def test_perfect_prediction_oa(self):
        preds = torch.randint(1, 7, (1, 32, 32))
        labels = preds.clone()
        m = compute_metrics(preds, labels, num_classes=7)
        assert m['OA'] == 1.0
        assert m['mIoU'] > 0.99
        assert m['Kappa'] > 0.99

    def test_all_background_iou(self):
        preds = torch.zeros(1, 32, 32, dtype=torch.long)
        labels = torch.zeros(1, 32, 32, dtype=torch.long)
        m = compute_metrics(preds, labels, num_classes=7)
        assert m['OA'] == 1.0

    def test_all_ignore_label(self):
        preds = torch.randint(0, 7, (1, 32, 32))
        labels = torch.full((1, 32, 32), 255, dtype=torch.long)
        m = compute_metrics(preds, labels, num_classes=7)
        assert "OA" in m
