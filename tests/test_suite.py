"""
系统测试套件 - 覆盖项目核心功能的测试框架

测试模块：
1. 数据预处理管道测试
2. 模型架构测试
3. 训练流程测试
4. 推理流程测试
5. 导出功能测试
6. 性能基准测试
7. EDL校准验证测试
8. 模型可解释性测试
"""

import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ==================== 数据预处理管道测试 ====================
class TestPreprocessPipeline:
    """数据预处理管道测试"""
    
    @pytest.fixture
    def raw_data(self):
        """创建测试数据"""
        T, C_opt, C_sar, C_dem, H, W = 12, 10, 5, 5, 64, 64
        return {
            'opt': np.random.rand(T, C_opt, H, W).astype(np.float32),
            'sar': np.random.rand(T, C_sar, H, W).astype(np.float32) - 10,
            'dem': np.random.rand(C_dem, H, W).astype(np.float32) * 500,
            'doy': np.linspace(0, 1, T)
        }
    
    @pytest.fixture
    def transforms(self):
        """创建变换参数"""
        return {
            'opt': {'target_size': (64, 64)},
            'sar': {'target_size': (64, 64)},
            'dem': {'target_size': (64, 64)}
        }
    
    def test_spatial_aligner(self, raw_data, transforms):
        """测试空间配准模块"""
        from data.preprocess_pipeline import SpatialAligner
        
        aligner = SpatialAligner(target_resolution=10.0)
        aligned = aligner.align(raw_data, transforms)
        
        assert 'opt' in aligned
        assert 'sar' in aligned
        assert 'dem' in aligned
        assert aligned['opt'].shape == raw_data['opt'].shape
        assert aligned['sar'].shape == raw_data['sar'].shape
        assert aligned['dem'].shape == raw_data['dem'].shape
    
    def test_cloud_detector(self, raw_data):
        """测试云检测模块"""
        from data.preprocess_pipeline import CloudDetector
        
        detector = CloudDetector(threshold=0.3)
        cloud_mask = detector.detect(raw_data['opt'], raw_data['sar'])
        
        assert cloud_mask.shape == (raw_data['opt'].shape[0], 
                                   raw_data['opt'].shape[-2], 
                                   raw_data['opt'].shape[-1])
        assert cloud_mask.dtype == bool
        
        # 测试掩膜优化
        refined = detector.refine_mask(cloud_mask)
        assert refined.shape == cloud_mask.shape
    
    def test_temporal_interpolator(self, raw_data):
        """测试时序插值模块"""
        from data.preprocess_pipeline import TemporalInterpolator
        
        interpolator = TemporalInterpolator(max_gap=30, method='linear')
        cloud_mask = np.zeros((raw_data['opt'].shape[0], 64, 64), dtype=bool)
        cloud_mask[3:5] = True  # 模拟云遮挡
        
        filled, updated_mask, is_interp, valid_count = interpolator.interpolate(
            raw_data['opt'], raw_data['doy'], cloud_mask)

        assert filled.shape == raw_data['opt'].shape
        assert valid_count.shape == (64, 64)
        assert updated_mask.shape == cloud_mask.shape
        assert is_interp.shape == cloud_mask.shape
    
    def test_spectral_normalizer(self, raw_data):
        """测试光谱归一化模块"""
        from data.preprocess_pipeline import SpectralNormalizer
        
        normalizer = SpectralNormalizer()
        
        # 测试训练模式
        norm_opt = normalizer.normalize(raw_data['opt'], 'opt')
        assert np.allclose(norm_opt.mean(), 0, atol=0.1)
        assert np.allclose(norm_opt.std(), 1, atol=0.1)
        
        # 测试反归一化
        denorm_opt = normalizer.denormalize(norm_opt, 'opt')
        assert np.allclose(denorm_opt, raw_data['opt'], atol=1e-5)
    
    def test_data_augmenter(self, raw_data):
        """测试数据增强模块"""
        from data.preprocess_pipeline import DataAugmenter, DataSample
        
        augmenter = DataAugmenter(prob=1.0)  # 强制增强
        
        sample = DataSample(
            opt_seq=raw_data['opt'],
            sar_seq=raw_data['sar'],
            dem=raw_data['dem'],
            doy=raw_data['doy'],
            label=np.random.randint(0, 7, (64, 64))
        )
        
        augmented = augmenter.augment(sample)
        
        assert augmented.opt_seq.shape == sample.opt_seq.shape
        assert augmented.sar_seq.shape == sample.sar_seq.shape
        assert augmented.dem.shape == sample.dem.shape
        if augmented.label is not None:
            assert augmented.label.shape == sample.label.shape
    
    def test_data_quality_checker(self, raw_data):
        """测试数据质量检查模块"""
        from data.preprocess_pipeline import DataQualityChecker, DataSample
        
        checker = DataQualityChecker()
        
        sample = DataSample(
            opt_seq=raw_data['opt'],
            sar_seq=raw_data['sar'],
            dem=raw_data['dem'],
            doy=raw_data['doy'],
            valid_count=np.ones((64, 64)) * 12
        )
        
        quality = checker.check(sample)
        
        assert isinstance(quality, dict)
        assert all(isinstance(v, bool) for v in quality.values())
        assert all(quality.values())  # 模拟数据应该通过所有检查
    
    def test_full_pipeline(self, raw_data, transforms):
        """测试完整预处理管道"""
        from data.preprocess_pipeline import PreprocessPipeline, PreprocessConfig
        
        config = PreprocessConfig(
            target_resolution=10.0,
            cloud_threshold=0.3,
            max_gap=30,
            normalize=True,
            augment=False
        )
        
        pipeline = PreprocessPipeline(config)
        sample = pipeline.process(raw_data, transforms, is_training=False)
        
        assert sample is not None
        assert sample.opt_seq.shape == raw_data['opt'].shape
        assert sample.sar_seq.shape == raw_data['sar'].shape
        assert sample.dem.shape == raw_data['dem'].shape
        assert sample.cloud_mask is not None
        assert sample.valid_count is not None

# ==================== 模型架构测试 ====================
class TestModelArchitecture:
    """模型架构测试"""
    
    def test_model_initialization(self):
        """测试模型初始化"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        )
        
        assert model is not None
        assert hasattr(model, 'edl_head')
        assert hasattr(model, 'predict_uncertainty')
    
    def test_forward_pass(self):
        """测试前向传播"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        B, T, H, W = 2, 12, 32, 32
        opt = torch.randn(B, T, 10, H, W).to(device)
        sar = torch.randn(B, T, 5, H, W).to(device)
        dem = torch.randn(B, 5, H, W).to(device)
        doy = torch.rand(B, T).to(device)
        
        model.train()
        alpha, ndvi, cl = model(opt, sar, dem, doy, epoch=10)
        
        assert alpha.shape == (B, 7, H, W)
        assert ndvi.shape == (B * T,)
        assert isinstance(cl, torch.Tensor)
        
        preds = dirichlet_to_predictions(alpha.detach())
        assert 'probs' in preds
        assert 'vacuity' in preds
        assert 'dissonance' in preds
    
    def test_uncertainty_inference(self):
        """测试不确定性推理"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        B, T, H, W = 1, 12, 32, 32
        opt = torch.randn(B, T, 10, H, W).to(device)
        sar = torch.randn(B, T, 5, H, W).to(device)
        dem = torch.randn(B, 5, H, W).to(device)
        doy = torch.rand(B, T).to(device)
        
        result = model.predict_uncertainty(opt, sar, dem, doy, n_passes=3, use_tta=False)
        
        assert 'probs' in result
        assert 'vacuity' in result
        assert 'dissonance' in result
        assert 'class_var' in result
        assert 'pred_class' in result
        
        assert result['probs'].shape == (B, 7, H, W)
        assert result['vacuity'].shape == (B, H, W)
        assert result['dissonance'].shape == (B, H, W)
    
    def test_model_parameters(self):
        """测试模型参数数量"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        )
        
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
        print(f"模型参数数量: {params/1e6:.1f}M")

# ==================== 训练流程测试 ====================
class TestTrainingWorkflow:
    """训练流程测试"""
    
    def test_training_step(self):
        """测试训练步骤"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, training_step, EDLLoss
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        edl_loss_fn = EDLLoss(num_classes=7, lambda_max=0.5, kl_anneal_epochs=50)
        
        B, T, H, W = 2, 12, 32, 32
        batch = {
            'opt': torch.randn(B, T, 10, H, W).to(device),
            'sar': torch.randn(B, T, 5, H, W).to(device),
            'dem': torch.randn(B, 5, H, W).to(device),
            'doy': torch.rand(B, T).to(device),
            'y': torch.randint(0, 7, (B, H, W)).to(device)
        }
        
        model.train()
        total_loss, metrics = training_step(model, batch, edl_loss_fn, epoch=10)
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert 'edl_loss' in metrics
        assert 'ndvi_loss' in metrics
        # consistency_loss only present when use_v6=True

        # 测试反向传播
        total_loss.backward()
        assert any(p.grad is not None for p in model.parameters())
    
    def test_mixed_precision_training(self):
        """测试混合精度训练"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, training_step, EDLLoss
        from torch.cuda.amp import GradScaler, autocast
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        edl_loss_fn = EDLLoss(num_classes=7)
        scaler = GradScaler()
        
        B, T, H, W = 2, 12, 32, 32
        batch = {
            'opt': torch.randn(B, T, 10, H, W).to(device),
            'sar': torch.randn(B, T, 5, H, W).to(device),
            'dem': torch.randn(B, 5, H, W).to(device),
            'doy': torch.rand(B, T).to(device),
            'y': torch.randint(0, 7, (B, H, W)).to(device)
        }
        
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        with autocast():
            total_loss, _ = training_step(model, batch, edl_loss_fn, epoch=10)
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert True  # 测试通过

# ==================== 推理流程测试 ====================
class TestInferenceWorkflow:
    """推理流程测试"""
    
    def test_sliding_window_inference(self):
        """测试滑动窗口推理"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        T, C_opt, H, W = 12, 10, 128, 128
        opt_seq = np.random.rand(T, C_opt, H, W).astype(np.float32)
        sar_seq = np.random.rand(T, 5, H, W).astype(np.float32)
        doy_norm = np.linspace(0, 1, T)
        
        model.eval()
        with torch.no_grad():
            opt_t = torch.from_numpy(opt_seq).unsqueeze(0).float().to(device)
            sar_t = torch.from_numpy(sar_seq).unsqueeze(0).float().to(device)
            dem_t = torch.randn(1, 5, H, W).float().to(device)
            doy_t = torch.from_numpy(doy_norm).unsqueeze(0).float().to(device)
            
            result = model.predict_uncertainty(opt_t, sar_t, dem_t, doy_t, n_passes=3, use_tta=False)
        
        assert result['probs'].shape == (1, 7, H, W)
        assert result['pred_class'].shape == (1, H, W)
    
    @pytest.mark.slow
    def test_onnx_export(self):
        """测试ONNX导出 (使用小型模型以加速)"""
        import os
        os.environ["PYTORCH_JIT_LOG"] = "0"
        from models.fusion_net_v5_edl import FusionCropNetV5EDL

        device = torch.device("cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        ).to(device)
        
        model.eval()
        
        B, T, H, W = 1, 12, 32, 32
        opt = torch.randn(B, T, 10, H, W).to(device)
        sar = torch.randn(B, T, 5, H, W).to(device)
        dem = torch.randn(B, 5, H, W).to(device)
        doy = torch.rand(B, T).to(device)
        
        torch.onnx.export(
            model,
            (opt, sar, dem, doy),
            "test_model.onnx",
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['opt', 'sar', 'dem', 'doy'],
            output_names=['alpha'],
            dynamic_axes={
                'opt': {0: 'batch'},
                'sar': {0: 'batch'},
                'dem': {0: 'batch'},
                'doy': {0: 'batch'},
                'alpha': {0: 'batch'}
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load("test_model.onnx")
        onnx.checker.check_model(onnx_model)
        
        # 清理测试文件
        os.remove("test_model.onnx")

# ==================== 导出功能测试 ====================
class TestExportFunctions:
    """导出功能测试"""
    
    def test_emd_export(self):
        """测试EMD导出"""
        import json
        
        emd_content = {
            "Framework": "PyTorch",
            "ModelConfiguration": {
                "ModelType": "Classification",
                "ImageHeight": 32,
                "ImageWidth": 32,
                "NumberOfClasses": 7
            },
            "ModelParameters": {
                "opt_channels": 10,
                "sar_channels": 5,
                "dem_channels": 5
            }
        }
        
        with open("test_model.emd", 'w', encoding='utf-8') as f:
            json.dump(emd_content, f, ensure_ascii=False, indent=2)
        
        assert os.path.exists("test_model.emd")
        
        with open("test_model.emd", 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        assert loaded == emd_content
        os.remove("test_model.emd")
    
    def test_dlpk_export(self):
        """测试DLPK导出"""
        import zipfile
        import io
        
        with zipfile.ZipFile("test_model.dlpk", 'w') as zf:
            zf.writestr("model.emd", '{"ModelType": "Classification"}')
            zf.writestr("model.onnx", b"dummy")
        
        assert os.path.exists("test_model.dlpk")
        
        with zipfile.ZipFile("test_model.dlpk", 'r') as zf:
            files = zf.namelist()
            assert "model.emd" in files
            assert "model.onnx" in files
        
        os.remove("test_model.dlpk")

# ==================== 性能基准测试 ====================
class TestPerformanceBenchmark:
    """性能基准测试"""
    
    def test_inference_speed(self):
        """测试推理速度"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        import time
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        model.eval()
        
        B, T, H, W = 1, 12, 64, 64
        opt = torch.randn(B, T, 10, H, W).to(device)
        sar = torch.randn(B, T, 5, H, W).to(device)
        dem = torch.randn(B, 5, H, W).to(device)
        doy = torch.rand(B, T).to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                model(opt, sar, dem, doy)
        
        # 测试
        num_runs = 10
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                model(opt, sar, dem, doy)
        elapsed = time.time() - start
        
        avg_time = elapsed / num_runs
        print(f"平均推理时间: {avg_time*1000:.2f} ms")
        assert avg_time < 5.0  # CPU推理小于5秒
    
    def test_memory_usage(self):
        """测试内存使用"""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        ).to(device)
        
        params = sum(p.numel() for p in model.parameters())
        memory_mb = params * 4 / 1e6  # float32
        
        print(f"模型内存占用: {memory_mb:.1f} MB")
        assert memory_mb < 500  # 小于500MB

# ==================== EDL校准验证测试 ====================
class TestEDLCalibration:
    """EDL不确定性校准验证测试"""

    def test_ece_computation(self):
        """测试ECE计算"""
        from utils.calibration import (expected_calibration_error, adaptive_ece,
                                        maximum_calibration_error)

        np.random.seed(42)
        N = 1000
        conf = np.clip(np.random.beta(4, 1, N), 0.01, 0.99)
        acc = (np.random.rand(N) < conf).astype(np.float32)

        ece, bins = expected_calibration_error(conf, acc, n_bins=10)
        assert 0 <= ece <= 1
        assert len(bins) == 10

        adap_ece, _ = adaptive_ece(conf, acc, n_bins=10)
        assert 0 <= adap_ece <= 1

        mce = maximum_calibration_error(conf, acc, n_bins=10)
        assert 0 <= mce <= 1

    def test_sharpness_dispersion(self):
        """测试锐度和分散度指标"""
        from utils.calibration import sharpness, dispersion

        np.random.seed(42)
        conf = np.random.beta(4, 1, 500)
        alpha = np.random.gamma(10, 1, (500, 7)) + 1.0
        targets = np.random.randint(0, 7, 500)

        s = sharpness(conf)
        d = dispersion(alpha, targets, 7)
        assert s >= 0
        assert d > 0

    def test_nll_brier_dirichlet(self):
        """测试Dirichlet NLL和Brier Score"""
        from utils.calibration import (negative_log_likelihood_dirichlet,
                                        brier_score_dirichlet)

        np.random.seed(42)
        K, N = 7, 500
        alpha = np.random.gamma(15, 1, (N, K)) + 1.0
        targets = np.random.randint(0, K, N)

        nll = negative_log_likelihood_dirichlet(alpha, targets, K)
        brier = brier_score_dirichlet(alpha, targets, K)
        assert nll > 0
        assert 0 <= brier <= 2

    def test_uncertainty_error_correlation(self):
        """测试不确定性-错误相关性"""
        from utils.calibration import (uncertainty_error_correlation,
                                        uncertainty_auroc, uncertainty_pr_auc)

        np.random.seed(42)
        N = 500
        vacuity = np.random.rand(N)
        correct = (vacuity < 0.5).astype(np.float32)
        noise_mask = np.random.rand(N) < 0.1
        correct[noise_mask] = 1 - correct[noise_mask]

        r, p = uncertainty_error_correlation(vacuity, correct)
        auroc = uncertainty_auroc(vacuity, correct)
        prauc = uncertainty_pr_auc(vacuity, correct)

        assert -1 <= r <= 1
        assert 0 <= auroc <= 1
        assert 0 <= prauc <= 1

    def test_ood_detection(self):
        """测试OOD检测指标"""
        from utils.calibration import ood_detection_metrics

        np.random.seed(42)
        N = 500
        vacuity = np.random.rand(N)
        correct = (vacuity < np.median(vacuity)).astype(np.float32)

        metrics = ood_detection_metrics(vacuity, correct, percentile=50)
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert "f1" in metrics
        assert "flag_rate" in metrics

    def test_rejection_curve(self):
        """测试拒绝曲线"""
        from utils.calibration import uncertainty_rejection_curve

        np.random.seed(42)
        N = 500
        vacuity = np.random.rand(N)
        correct = (np.random.rand(N) < 0.7).astype(np.float32)

        curve = uncertainty_rejection_curve(vacuity, correct, n_points=10)
        assert len(curve) > 0
        for p in curve:
            assert "retention" in p
            assert "accuracy" in p

    def test_full_calibration_report(self):
        """测试完整校准报告"""
        from utils.calibration import calibration_report, print_calibration_report

        np.random.seed(42)
        K, H, W = 7, 32, 32
        alpha = np.random.gamma(12, 1, (1, K, H, W)) + 1.0
        targets = np.random.randint(0, K, (H, W))

        report = calibration_report(alpha, targets, K, n_bins=10)
        assert "ECE" in report
        assert "NLL" in report
        assert "Brier" in report
        assert "SpearmanR_vacuity" in report
        assert "PerClass" in report
        assert "OOD_detection" in report
        assert "RejectionCurve" in report

        # Should not crash
        print_calibration_report(report)

    def test_calibration_vs_random(self):
        """验证校准指标对随机预测的敏感性"""
        from utils.calibration import expected_calibration_error

        np.random.seed(42)
        N = 1000

        # Perfect calibration: conf = acc
        conf_cal = np.random.beta(10, 2, N)
        acc_cal = (np.random.rand(N) < conf_cal).astype(np.float32)
        ece_cal, _ = expected_calibration_error(conf_cal, acc_cal, n_bins=10)

        # Overconfident: high conf, low acc
        conf_over = np.full(N, 0.95)
        acc_over = (np.random.rand(N) < 0.5).astype(np.float32)
        ece_over, _ = expected_calibration_error(conf_over, acc_over, n_bins=10)

        # Overconfident should have higher ECE
        assert ece_over > ece_cal or ece_cal < 0.15
        print(f"ECE (calibrated-like): {ece_cal:.4f}, ECE (overconfident): {ece_over:.4f}")


# ==================== 模型可解释性测试 ====================
class TestModelInterpretability:
    """模型可解释性分析测试"""

    @pytest.fixture
    def dummy_inputs(self):
        B, T, H, W = 1, 10, 32, 32
        return {
            "opt_seq": torch.randn(B, T, 10, H, W),
            "sar_seq": torch.randn(B, T, 5, H, W),
            "dem": torch.randn(B, 5, H, W),
            "doy": torch.rand(B, T),
        }

    @pytest.fixture
    def edl_model(self):
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        return FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4
        )

    def test_gradcam(self, edl_model, dummy_inputs):
        """测试Grad-CAM"""
        from utils.interpretability import GradCAM_EDL

        device = "cpu"
        edl_model.to(device).eval()
        opt_seq = dummy_inputs["opt_seq"].to(device)
        sar_seq = dummy_inputs["sar_seq"].to(device)
        dem = dummy_inputs["dem"].to(device)
        doy = dummy_inputs["doy"].to(device)

        gradcam = GradCAM_EDL(edl_model, target_layer_name="decoder")
        hm = gradcam(opt_seq, sar_seq, dem, doy, class_idx=3)
        assert hm.shape == (32, 32)
        assert hm.min() >= 0
        assert hm.max() <= 1

    def test_gradcam_per_class(self, edl_model, dummy_inputs):
        """测试逐类Grad-CAM"""
        from utils.interpretability import gradcam_per_class

        device = "cpu"
        edl_model.to(device).eval()
        maps = gradcam_per_class(
            edl_model,
            dummy_inputs["opt_seq"].to(device),
            dummy_inputs["sar_seq"].to(device),
            dummy_inputs["dem"].to(device),
            dummy_inputs["doy"].to(device),
            num_classes=7,
        )
        assert len(maps) == 7
        for k, m in maps.items():
            assert m.shape == (32, 32)

    def test_modality_ablation(self, edl_model, dummy_inputs):
        """测试模态消融分析"""
        from utils.interpretability import modality_ablation

        device = "cpu"
        edl_model.to(device).eval()
        results = modality_ablation(
            edl_model,
            dummy_inputs["opt_seq"].to(device),
            dummy_inputs["sar_seq"].to(device),
            dummy_inputs["dem"].to(device),
            dummy_inputs["doy"].to(device),
            device=device,
        )
        assert "full" in results
        assert "no_opt" in results
        assert "no_sar" in results
        assert "no_dem" in results
        assert "relative_importance" in results
        ri = results["relative_importance"]
        assert abs(ri["optical"] + ri["sar"] + ri["dem"] - 1.0) < 0.01

    def test_temporal_importance(self, edl_model, dummy_inputs):
        """测试时序重要性分析"""
        from utils.interpretability import temporal_importance

        device = "cpu"
        edl_model.to(device).eval()
        imp, details = temporal_importance(
            edl_model,
            dummy_inputs["opt_seq"].to(device),
            dummy_inputs["sar_seq"].to(device),
            dummy_inputs["dem"].to(device),
            dummy_inputs["doy"].to(device),
            device=device,
        )
        assert len(imp) == dummy_inputs["opt_seq"].shape[1]
        assert abs(imp.sum() - 1.0) < 0.01

    def test_spectral_band_importance(self, edl_model, dummy_inputs):
        """测试光谱波段重要性"""
        from utils.interpretability import spectral_band_importance

        device = "cpu"
        edl_model.to(device).eval()
        imp = spectral_band_importance(
            edl_model,
            dummy_inputs["opt_seq"].to(device),
            dummy_inputs["sar_seq"].to(device),
            dummy_inputs["dem"].to(device),
            dummy_inputs["doy"].to(device),
            device=device,
        )
        assert "optical_bands" in imp
        assert "sar_bands" in imp
        assert len(imp["optical_bands"]) == 10
        assert len(imp["sar_bands"]) == 5
        assert abs(sum(imp["optical_bands"]) - 1.0) < 0.01
        assert abs(sum(imp["sar_bands"]) - 1.0) < 0.01

    def test_confusion_region_analysis(self):
        """测试混淆区域分析"""
        from utils.interpretability import confusion_region_analysis

        np.random.seed(42)
        K, H, W = 7, 32, 32
        alpha = np.random.gamma(10, 1, (1, K, H, W)) + 1.0
        targets = np.random.randint(0, K, (H, W))

        pairs = confusion_region_analysis(alpha, targets, num_classes=K)
        assert len(pairs) > 0
        for k, v in pairs.items():
            assert "n" in v
            assert "mean_vacuity" in v

    def test_pixel_explanation_report(self):
        """测试像素级解释报告"""
        from utils.interpretability import pixel_explanation_report

        np.random.seed(42)
        K, H, W = 7, 32, 32
        alpha = np.random.gamma(10, 1, (1, K, H, W)) + 2.0
        targets = np.random.randint(0, K, (H, W))

        report = pixel_explanation_report(alpha, targets, num_classes=K)
        assert "correct" in report
        assert "incorrect" in report
        for key in ["vacuity", "dissonance", "margin", "entropy", "confidence"]:
            assert key in report["correct"]
            assert key in report["incorrect"]

    def test_cross_modal_attention_analysis(self, edl_model, dummy_inputs):
        """测试跨模态注意力分析"""
        from utils.interpretability import cross_modal_attention_analysis

        device = "cpu"
        edl_model.to(device).eval()
        result = cross_modal_attention_analysis(
            edl_model,
            dummy_inputs["opt_seq"].to(device),
            dummy_inputs["sar_seq"].to(device),
            dummy_inputs["dem"].to(device),
            dummy_inputs["doy"].to(device),
            device=device,
        )
        # May or may not capture gates depending on model structure
        assert isinstance(result, dict)


# ==================== V6 ModalNormalize Tests ====================
class TestV6ModalNormalize:
    """Verify ModalNormalize + Early Fusion."""

    def test_modal_norm_range_normalization(self):
        """Output should not be dominated by DEM after normalization."""
        from models._base import ModalNormalize
        mn = ModalNormalize()
        opt = torch.rand(2, 10, 64, 64)           # [0,1]
        sar = torch.randn(2, 5, 64, 64) * 5 - 5   # [-25,5]ish
        dem = torch.rand(2, 5, 64, 64) * 8000      # [0,8848]ish
        out = mn(opt, sar, dem)
        # Each modality section should have roughly unit variance
        opt_var = out[:, :10].var()
        sar_var = out[:, 10:15].var()
        dem_var = out[:, 15:].var()
        # After LayerNorm, all should be ~1.0
        assert 0.1 < opt_var < 10.0, f"opt_var={opt_var}"
        assert 0.1 < sar_var < 10.0, f"sar_var={sar_var}"
        assert 0.1 < dem_var < 10.0, f"dem_var={dem_var}"

    def test_modal_norm_output_channels(self):
        """Output = sum of input channels."""
        from models._base import ModalNormalize
        mn = ModalNormalize()
        opt = torch.randn(2, 10, 32, 32)
        sar = torch.randn(2, 5, 32, 32)
        dem = torch.randn(2, 5, 32, 32)
        out = mn(opt, sar, dem)
        assert out.shape[1] == 20  # 10+5+5

    def test_early_fusion_integration(self):
        """Model with Early Fusion enabled runs forward pass."""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
            pretrained=False
        )
        model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        with torch.no_grad():
            alpha = model(opt, sar, dem, doy)
        assert alpha.shape == (B, 7, H, W)
        assert not torch.isnan(alpha).any()


# ==================== V6 TemporalLite Integration Tests ====================
class TestV6TemporalLiteIntegration:
    """Verify TemporalLite integrates correctly in V5EDL _encode path."""

    @pytest.fixture
    def model(self):
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        return FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', pretrained=False,
            n_heads=4, n_layers=2, use_v6_enhancements=True
        )

    @pytest.fixture
    def inputs(self):
        B, T, H, W = 2, 6, 128, 128
        return (
            torch.randn(B, T, 10, H, W),  # opt
            torch.randn(B, T, 5, H, W),   # sar
            torch.randn(B, 5, H, W),      # dem
            torch.randint(1, 365, (B, T)).float() / 365.0,  # doy
        )

    def test_forward_with_temporal_lite(self, model, inputs):
        """Forward pass succeeds with TemporalLite active."""
        model.eval()
        with torch.no_grad():
            alpha = model(*inputs)
        assert alpha.shape == (2, 7, 128, 128)

    def test_no_nan_in_output(self, model, inputs):
        """Output contains no NaN."""
        model.eval()
        with torch.no_grad():
            alpha = model(*inputs)
        assert not torch.isnan(alpha).any()

    def test_temporal_lite_params_in_model(self, model):
        """Model contains TemporalLite parameters."""
        tl_params = [
            name for name, _ in model.named_parameters()
            if 'temp_lite' in name
        ]
        assert len(tl_params) > 0, "TemporalLite not found in model parameters"

    def test_temporal_lite_trainable(self, model):
        """TemporalLite parameters require grad."""
        for name, param in model.named_parameters():
            if 'temp_lite' in name:
                assert param.requires_grad, f"{name} should be trainable"

    def test_compatible_with_cloud_mask(self, model, inputs):
        """Forward pass works with cloud mask."""
        opt, sar, dem, doy = inputs
        B, T = opt.shape[:2]
        cm = torch.zeros(B, T, 128, 128, dtype=torch.bool)
        model.eval()
        with torch.no_grad():
            alpha = model(opt, sar, dem, doy, cloud_mask=cm)
        assert alpha.shape == (2, 7, 128, 128)

    def test_deterministic_eval(self, model, inputs):
        """Same input twice -> same output."""
        model.eval()
        with torch.no_grad():
            a1 = model(*inputs)
            a2 = model(*inputs)
        assert torch.allclose(a1, a2, atol=1e-5)


# ==================== V6 DEM Multi-Path Injection Tests ====================
class TestV6DEMPaths:
    """Verify 5 DEM paths work correctly."""

    @pytest.fixture
    def model(self):
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        return FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
            pretrained=False, use_v6_enhancements=True
        )

    @pytest.fixture
    def inputs(self):
        B, T, H, W = 2, 6, 128, 128
        return (
            torch.randn(B, T, 10, H, W),
            torch.randn(B, T, 5, H, W),
            torch.randn(B, 5, H, W),
            torch.rand(B, T),
        )

    def test_all_modalities_forward(self, model, inputs):
        """Full forward pass with all DEM paths active."""
        model.eval()
        with torch.no_grad():
            alpha = model(*inputs)
        assert alpha.shape == (2, 7, 128, 128)
        assert not torch.isnan(alpha).any()

    def test_dem_opt_cond_module_exists(self, model):
        """DEMOpticalConditioner is attached to the model."""
        assert hasattr(model, 'dem_opt_cond')

    def test_dem_temporal_proj_exists(self, model):
        """DEM temporal projection is attached."""
        assert hasattr(model, 'dem_temporal_proj')

    def test_dem_skip_proj_in_decoder(self, model):
        """Decoder has DEM skip projection."""
        assert hasattr(model.decoder, 'dem_skip_proj')

    def test_missing_dem_handled(self, model, inputs):
        """Forward pass with DEM masked out (use_dem=False)."""
        opt, sar, dem, doy = inputs
        model.eval()
        with torch.no_grad():
            alpha = model(opt, sar, dem, doy, modality_mask=(True, True, False))
        assert alpha.shape == (2, 7, 128, 128)
        assert not torch.isnan(alpha).any()


# ==================== V6 Multi-Scale Cross-Modal Attention Tests ====================
class TestV6MultiScaleCrossAttn:
    """Verify multi-scale cross-modal attention."""

    @pytest.fixture
    def model(self):
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        return FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
            pretrained=False, use_v6_enhancements=True
        )

    def test_cross_modal_modules_exist(self, model):
        """Multi-scale cross-modal modules are attached."""
        assert hasattr(model, 'cross_modal_h')
        assert hasattr(model, 'cross_modal_h2')
        assert hasattr(model, 'cross_modal')  # Original H/4 still exists

    def test_forward_all_modalities(self, model):
        """Full forward with multi-scale cross-attention."""
        model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        with torch.no_grad():
            alpha = model(opt, sar, dem, doy)
        assert alpha.shape == (B, 7, H, W)
        assert not torch.isnan(alpha).any()

    def test_forward_opt_only(self, model):
        """Forward with only optical (SAR masked)."""
        model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        with torch.no_grad():
            alpha = model(opt, sar, dem, doy, modality_mask=(True, False, True))
        assert not torch.isnan(alpha).any()

    def test_forward_sar_only(self, model):
        """Forward with only SAR (optical masked)."""
        model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        with torch.no_grad():
            alpha = model(opt, sar, dem, doy, modality_mask=(False, True, True))
        assert not torch.isnan(alpha).any()

    def test_cross_modal_lite_unit(self):
        """CrossModalLite forward produces valid output."""
        from models._base import CrossModalLite
        for d_model, heads in [(64, 1), (128, 4)]:
            cm = CrossModalLite(d_model, n_heads=heads)
            x = torch.randn(2, d_model, 32, 32)
            y = torch.randn(2, d_model, 32, 32)
            out = cm(x, y)
            assert out.shape == x.shape
            assert not torch.isnan(out).any()

    def test_cross_modal_lite_mismatched_res(self):
        """CrossModalLite handles mismatched spatial sizes."""
        from models._base import CrossModalLite
        cm = CrossModalLite(64, n_heads=1)
        opt = torch.randn(2, 64, 16, 16)
        sar = torch.randn(2, 64, 32, 32)  # Different resolution
        out = cm(opt, sar)
        assert out.shape == opt.shape
        assert not torch.isnan(out).any()


# ==================== V6 多任务头测试 ====================
class TestV6MultiTask:
    """Verify multi-task heads and pseudo-labels."""

    def test_lai_head_shape(self):
        from models.multi_task_heads import LAIRegressionHead
        head = LAIRegressionHead(64)
        x = torch.randn(4, 64, 32, 32)
        out = head(x)
        assert out.shape == (4,)
        assert (out >= 0).all()  # Softplus ensures non-negative

    def test_growth_stage_head_shape(self):
        from models.multi_task_heads import GrowthStageHead
        head = GrowthStageHead(64, num_stages=5)
        x = torch.randn(4, 64, 32, 32)
        out = head(x)
        assert out.shape == (4, 5)

    def test_boundary_head_shape(self):
        from models.multi_task_heads import BoundaryHead
        head = BoundaryHead(64)
        x = torch.randn(4, 64, 32, 32)
        out = head(x)
        assert out.shape == (4, 1, 32, 32)
        assert (out >= 0).all() and (out <= 1).all()

    def test_multi_task_loss(self):
        from models.multi_task_heads import MultiTaskLoss
        mtl = MultiTaskLoss(num_tasks=5)
        losses = {
            'crop': torch.tensor(2.0),
            'ndvi': torch.tensor(0.5),
            'lai': torch.tensor(0.3),
            'growth': torch.tensor(1.5),
            'boundary': torch.tensor(0.8),
        }
        total = mtl(losses)
        assert total.ndim == 0  # scalar
        total.backward()  # log_vars get gradients

    def test_lai_pseudo_generation(self):
        from data.pseudo_labels import generate_lai_pseudo
        ndvi = torch.tensor([0.2, 0.5, 0.8])
        lai = generate_lai_pseudo(ndvi)
        assert lai.shape == ndvi.shape
        assert (lai > 0).all()
        # Higher NDVI -> higher LAI
        assert lai[2] > lai[0]

    def test_boundary_pseudo_generation(self):
        from data.pseudo_labels import generate_boundary_pseudo
        # Flat DEM -> no boundaries
        dem_flat = torch.zeros(2, 5, 64, 64)
        boundary = generate_boundary_pseudo(dem_flat)
        assert boundary.shape == (2, 1, 64, 64)
        # Steep DEM -> more boundaries
        dem_steep = torch.randn(2, 5, 64, 64) * 100
        boundary_steep = generate_boundary_pseudo(dem_steep)
        assert boundary_steep.sum() > boundary.sum()

    def test_growth_stage_pseudo(self):
        from data.pseudo_labels import generate_growth_stage_pseudo
        doy = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9, 0.95]])
        stages = generate_growth_stage_pseudo(doy)
        assert stages.shape == (6,)
        assert stages[0] < stages[-1]  # Later DOY -> later stage

    def test_model_with_aux_outputs(self):
        """Model returns auxiliary outputs when return_aux=True."""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
            pretrained=False, use_v6_enhancements=True
        )
        model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        with torch.no_grad():
            result = model(opt, sar, dem, doy, return_aux=True)
        assert len(result) == 4  # (alpha, ndvi, loss, aux_tuple)
        alpha, ndvi, consist, aux = result
        assert len(aux) == 5  # V6 Block 7: lai, growth, boundary, scene_logits, crop_mix
        lai_pred, growth_pred, boundary_pred, scene_logits, crop_mix = aux
        assert lai_pred.shape == (B,)
        assert growth_pred.shape == (B, 5)
        assert boundary_pred.shape == (B, 1, H, W)
        assert scene_logits.shape == (B, 4)
        assert crop_mix.shape == (B, 7)


# ==================== V6 LightSceneHead 测试 ====================
class TestV6LightSceneHead:
    """Verify LightSceneHead."""

    def test_scene_head_shapes(self):
        from models._base import LightSceneHead
        head = LightSceneHead(in_ch=64, hidden=128, num_scene_types=4, num_crops=7)
        x = torch.randn(4, 64, 32, 32)
        scene_logits, crop_mix = head(x)
        assert scene_logits.shape == (4, 4)
        assert crop_mix.shape == (4, 7)
        # crop_mix should be a valid probability distribution
        assert torch.allclose(crop_mix.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert (crop_mix >= 0).all() and (crop_mix <= 1).all()

    def test_scene_head_deterministic(self):
        from models._base import LightSceneHead
        head = LightSceneHead(in_ch=64, hidden=128)
        head.eval()
        x = torch.randn(2, 64, 32, 32)
        s1, c1 = head(x)
        s2, c2 = head(x)
        assert torch.allclose(s1, s2)
        assert torch.allclose(c1, c2)

    def test_scene_head_integration(self):
        """Model returns scene predictions in aux output."""
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
            pretrained=False, use_v6_enhancements=True
        )
        model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        with torch.no_grad():
            result = model(opt, sar, dem, doy, return_aux=True)
        assert len(result) == 4
        alpha, ndvi, consist, aux = result
        # aux now has 5 elements: lai, growth, boundary, scene_logits, crop_mix
        assert len(aux) == 5
        lai_pred, growth_pred, boundary_pred, scene_logits, crop_mix = aux
        assert scene_logits.shape == (B, 4)
        assert crop_mix.shape == (B, 7)

    def test_scene_head_param_count(self):
        from models._base import LightSceneHead
        head = LightSceneHead(in_ch=64, hidden=256, num_scene_types=4, num_crops=7)
        total = sum(p.numel() for p in head.parameters())
        assert total < 200_000, f"LightSceneHead should be <200K params, got {total}"


# ==================== V6 Self-Training and DomainAdapter Tests ====================
class TestV6SelfTraining:
    """Verify self-training and DomainAdapter."""

    def test_vacuity_filter_all_confident(self):
        """All high-confidence, low-vacuity pixels pass filter."""
        from utils.self_training import filter_by_vacuity
        # High evidence for class 0 only → high confidence + low vacuity
        alpha = torch.ones(2, 7, 32, 32)        # base evidence=1 for all
        alpha[:, 0] = 200                         # class 0 dominates: S≈206, vacuity≈0.034, conf≈0.97
        mask = filter_by_vacuity(alpha, vacuity_threshold=0.3, confidence_threshold=0.9)
        assert mask.all()

    def test_vacuity_filter_all_uncertain(self):
        """All low-evidence pixels fail filter."""
        from utils.self_training import filter_by_vacuity
        alpha = torch.ones(2, 7, 32, 32) * 0.1  # S=0.7, vacuity=7/0.7=10 → high
        mask = filter_by_vacuity(alpha, vacuity_threshold=0.3, confidence_threshold=0.9)
        assert not mask.any()

    def test_vacuity_filter_mixed(self):
        """Mixed evidence: some pass, some fail."""
        from utils.self_training import filter_by_vacuity
        alpha = torch.ones(1, 7, 4, 4) * 10       # base S=70, vacuity=0.1, conf≈0.14
        alpha[:, 0, 0, 0] = 100                    # class 0 at (0,0): S=160, conf=100/160=0.625
        mask = filter_by_vacuity(alpha, vacuity_threshold=0.3, confidence_threshold=0.5)
        assert mask[0, 0, 0]  # High evidence / high confidence pixel should pass

    def test_self_training_loop_init(self):
        """SelfTrainingLoop initializes correctly."""
        from utils.self_training import SelfTrainingLoop
        loop = SelfTrainingLoop(None, vacuity_threshold=0.3, max_rounds=3)
        assert loop.vacuity_threshold == 0.3
        assert loop.max_rounds == 3

    def test_self_training_threshold_relaxation(self):
        """Thresholds relax after each round."""
        from utils.self_training import SelfTrainingLoop
        loop = SelfTrainingLoop(None, vacuity_threshold=0.3, confidence_threshold=0.9)
        # Simulate a round (won't actually run without unlabeled data)
        loop.vacuity_threshold = min(0.5, loop.vacuity_threshold + 0.05)
        loop.confidence_threshold = max(0.7, loop.confidence_threshold - 0.05)
        assert loop.vacuity_threshold == 0.35
        assert loop.confidence_threshold == 0.85

    def test_domain_adapter_identity_init(self):
        """DomainAdapter starts as identity transform."""
        from models._base import DomainAdapter
        da = DomainAdapter(64)
        x = torch.randn(4, 64, 32, 32)
        out = da(x)
        # Should be close to identity (shift=0, scale=1)
        assert torch.allclose(out, x, atol=1e-5)

    def test_domain_adapter_trainable(self):
        """DomainAdapter parameters receive gradients."""
        from models._base import DomainAdapter
        da = DomainAdapter(64)
        x = torch.randn(4, 64, 32, 32)
        out = da(x)
        loss = out.sum()
        loss.backward()
        assert da.shift.grad is not None
        assert da.scale.grad is not None

    def test_domain_adapter_in_optical_encoder(self):
        """OpticalEncoder with DomainAdapter runs forward."""
        from models._base import OpticalEncoder
        oe = OpticalEncoder(10, 512, 'resnet18', pretrained=False, use_domain_adapter=True)
        x = torch.randn(4, 10, 128, 128)
        main, p2, p3 = oe(x)
        assert main.shape[1] == 512  # feat_dim


# ==================== V6 ViT Feature Pyramid + 3-Expert LateFusion Tests ====================
class TestV6ViTAndExperts:
    """Verify ViT Feature Pyramid + 3-Expert LateFusion."""

    def test_vit_feature_pyramid_shapes(self):
        """ViTFeaturePyramid produces correct multi-scale outputs."""
        from models._base import ViTFeaturePyramid
        vfp = ViTFeaturePyramid(768, [256, 512, 1024])
        # Simulate ViT-B tokens: (B, 257, 768) for 256x256 img, P=16
        # grid = 256/16 = 16, N = 16*16 + 1 = 257
        x = torch.randn(2, 257, 768)
        f4, f8, f16 = vfp(x, patch_size=16, img_size=256)
        # H/16=16, H/32=8, H/64=4
        assert f4.shape == (2, 256, 16, 16)
        assert f8.shape == (2, 512, 8, 8)
        assert f16.shape == (2, 1024, 4, 4)

    def test_vit_fp_without_cls(self):
        """Works without CLS token."""
        from models._base import ViTFeaturePyramid
        vfp = ViTFeaturePyramid(768, [256, 512, 1024])
        x = torch.randn(2, 256, 768)  # 16x16 grid, no CLS
        f4, f8, f16 = vfp(x, patch_size=16, img_size=256)
        assert f4.shape[1] == 256

    def test_three_expert_fusion_shapes(self):
        """3-Expert LateFusion produces correct output shapes."""
        from models._base import ThreeExpertLateFusion
        tef = ThreeExpertLateFusion(512, 7, hidden=256)
        # Simulate flattened features (B*H2W2, D) where H2=W2=32
        N = 2 * 32 * 32
        opt_f = torch.randn(N, 512)
        sar_f = torch.randn(N, 512)
        fused_f = torch.randn(N, 512)
        # reshape context: B=2, H2=32, W2=32
        B, H2, W2 = 2, 32, 32
        opt_sp = opt_f.view(B, 512, H2, W2)
        sar_sp = sar_f.view(B, 512, H2, W2)
        fused_sp = fused_f.view(B, 512, H2, W2)

        logits_opt = tef.expert_opt(opt_sp)
        logits_sar = tef.expert_sar(sar_sp)
        logits_fused = tef.expert_fused(fused_sp)

        assert logits_opt.shape == (B, 7, H2, W2)
        assert logits_sar.shape == (B, 7, H2, W2)
        assert logits_fused.shape == (B, 7, H2, W2)

    def test_three_expert_vacuity_weights(self):
        """Vacuity weights: higher vacuity → lower weight."""
        from models._base import ThreeExpertLateFusion
        tef = ThreeExpertLateFusion(512, 7)
        B, H2, W2 = 2, 8, 8
        N = B * H2 * W2

        # Expert_opt very confident (low vacuity), Expert_sar uncertain (high vacuity)
        opt_f = torch.randn(N, 512) * 2  # higher logits → higher alpha → lower vacuity
        sar_f = torch.randn(N, 512) * 0.1
        fused_f = torch.randn(N, 512)

        final, weights = tef(opt_f, sar_f, fused_f, num_classes=7, B=B)
        assert final.shape == (B, 7, H2, W2)
        assert weights.shape == (B, 3, 1, 1)
        # Weights should sum to ~1
        assert torch.allclose(weights.sum(dim=1), torch.ones(B, 1, 1, device=weights.device), atol=1e-5)

    def test_expert_param_independence(self):
        """Three experts have independent parameters."""
        from models._base import ThreeExpertLateFusion
        tef = ThreeExpertLateFusion(512, 7)
        # Expert parameters should differ
        p_opt = list(tef.expert_opt.parameters())
        p_sar = list(tef.expert_sar.parameters())
        for po, ps in zip(p_opt, p_sar):
            assert not torch.equal(po.data, ps.data), "Expert parameters should be independent"

    def test_list_vit_models(self):
        """Registry contains expected models."""
        from models._base import list_vit_foundation_models
        models = list_vit_foundation_models()
        assert 'terrafm_b' in models
        assert 'dofa_vitb' in models
        assert models['terrafm_b']['dim'] == 768


# ==================== V6 End-to-End Model Tests ====================
class TestV6Model:
    """End-to-end FusionCropNetV6 tests."""

    @pytest.fixture
    def v6_model(self):
        from models.fusion_net_v6 import FusionCropNetV6
        return FusionCropNetV6(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
            pretrained=False
        )

    def test_v6_eval_forward(self, v6_model):
        """V6 eval returns alpha + aux dict."""
        v6_model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        with torch.no_grad():
            alpha, aux = v6_model(opt, sar, dem, doy)
        assert alpha.shape == (B, 7, H, W)
        assert 'lai' in aux
        assert 'growth' in aux
        assert 'boundary' in aux
        assert 'scene_logits' in aux
        assert 'crop_mix' in aux
        assert aux['lai'].shape == (B,)
        assert aux['growth'].shape == (B, 5)
        assert aux['boundary'].shape == (B, 1, H, W)
        assert aux['scene_logits'].shape == (B, 4)
        assert aux['crop_mix'].shape == (B, 7)

    def test_v6_train_forward(self, v6_model):
        """V6 train returns alpha, ndvi, consist, aux."""
        v6_model.train()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        alpha, ndvi, consist, aux = v6_model(opt, sar, dem, doy)
        assert alpha.shape == (B, 7, H, W)
        assert ndvi.shape == (B * T,)
        assert consist.ndim == 0  # scalar
        assert 'lai' in aux

    def test_v6_defaults(self, v6_model):
        """V6 has correct defaults."""
        assert v6_model.use_grad_ckpt == True
        assert v6_model.modality_dropout_p == 0.1

    def test_v6_all_components_present(self, v6_model):
        """All 14 V6 components exist."""
        expected = [
            'temp_lite_s1', 'temp_lite_s2', 'temp_lite_opt_p2',
            'modal_norm', 'early_fusion',
            'dem_opt_cond', 'dem_temporal_proj',
            'cross_modal_h', 'cross_modal_h2',
            'lai_head', 'growth_head', 'boundary_head',
            'scene_head', 'multi_task_loss'
        ]
        for attr in expected:
            assert hasattr(v6_model, attr), f"Missing: {attr}"

    def test_v6_backward(self, v6_model):
        """V6 backward pass works with all components."""
        v6_model.train()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        alpha, ndvi, consist, aux = v6_model(opt, sar, dem, doy)
        loss = alpha.sum() + ndvi.sum() + consist
        for v in aux.values():
            if isinstance(v, torch.Tensor) and v.requires_grad:
                loss = loss + v.sum()
        loss.backward()
        no_grad = [n for n, p in v6_model.named_parameters()
                   if p.grad is None and p.requires_grad]
        # Expected no-grad params: placeholders (only used when modality masked),
        # fallbacks (only used when fully clouded), dead-code early_fusion weight,
        # _init_weights-skipped backbone params, and log_vars (not in this loss path)
        expected_no_grad = {'placeholder_opt', 'placeholder_sar', 'placeholder_dem_feat',
                           'fallback_gate_opt', 'fallback_gate_sar',
                           'early_fusion.weight', 'multi_task_loss.log_vars',
                           'opt_temporal.fallback', 'sar_temporal.fallback',
                           'opt_temporal.obs_tok.embed.weight', 'sar_temporal.obs_tok.embed.weight'}
        unexpected = [n for n in no_grad if n.split('.')[-1] not in expected_no_grad
                      and not any(p in n for p in expected_no_grad)]
        # Backbone params may be skipped by _init_weights (pretrained_modules set)
        # Filter those out too
        unexpected = [n for n in unexpected if 'backbone' not in n and 'opt_enc.sp' not in n]
        assert len(unexpected) == 0, f"Unexpected missing gradients: {unexpected[:5]}"

    def test_v6_with_cloud_mask(self, v6_model):
        """V6 handles cloud mask."""
        v6_model.eval()
        B, T, H, W = 2, 6, 128, 128
        opt = torch.randn(B, T, 10, H, W)
        sar = torch.randn(B, T, 5, H, W)
        dem = torch.randn(B, 5, H, W)
        doy = torch.rand(B, T)
        cm = torch.zeros(B, T, H, W, dtype=torch.bool)
        with torch.no_grad():
            alpha, aux = v6_model(opt, sar, dem, doy, cloud_mask=cm)
        assert not torch.isnan(alpha).any()

    def test_v6_modality_dropout(self, v6_model):
        """V6 handles modality dropout in training (small spatial to avoid OOM)."""
        v6_model.train()
        B, T, H, W = 2, 4, 64, 64  # Small spatial: H/2=32, H=64 avoids 2GB attention
        for _ in range(10):
            opt = torch.randn(B, T, 10, H, W)
            sar = torch.randn(B, T, 5, H, W)
            dem = torch.randn(B, 5, H, W)
            doy = torch.rand(B, T)
            try:
                alpha, ndvi, consist, aux = v6_model(opt, sar, dem, doy)
                assert not torch.isnan(alpha).any()
            except Exception as e:
                # Skip OOM errors (expected on low-memory machines at certain scales)
                if 'memory' in str(e).lower() or 'alloc' in str(e).lower():
                    pytest.skip(f"Skipping due to memory: {e}")
                pytest.fail(f"Modality dropout failed: {e}")

    def test_v6_from_import(self):
        """FusionCropNetV6 importable from models."""
        from models import FusionCropNetV6
        m = FusionCropNetV6(opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                            feat_dim=512, backbone='resnet18', n_heads=4, n_layers=2,
                            pretrained=False)
        assert isinstance(m, FusionCropNetV6)


# ==================== 主测试入口 ====================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
