"""
训练器模块测试
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.trainer import FusionTrainer, TwoPhaseTrainer
from models.fusion_net_v5 import FusionCropNetV5

class SimpleDictDataset(Dataset):
    """简化的字典格式数据集"""
    def __init__(self, opt, sar, doy, labels):
        self.opt = opt
        self.sar = sar
        self.doy = doy
        self.labels = labels
    
    def __len__(self):
        return len(self.opt)
    
    def __getitem__(self, idx):
        return {
            "opt": self.opt[idx],
            "sar": self.sar[idx],
            "doy": self.doy[idx],
            "y": self.labels[idx]
        }

class TestFusionTrainer:
    """测试融合训练器"""
    
    def test_trainer_creation(self):
        """测试训练器创建"""
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        device = torch.device("cpu")
        
        trainer = FusionTrainer(model, optimizer, criterion, device)
        
        assert trainer is not None
        assert trainer.model is model
        assert trainer.device == device
    
    def test_train_epoch(self):
        """测试训练轮次"""
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        device = torch.device("cpu")
        
        trainer = FusionTrainer(model, optimizer, criterion, device)
        
        opt = torch.randn(2, 12, 10, 32, 32)
        sar = torch.randn(2, 12, 5, 32, 32)
        doy = torch.randn(2, 12)
        labels = torch.randint(1, 7, (2, 32, 32))
        
        dataset = SimpleDictDataset(opt, sar, doy, labels)
        dataloader = DataLoader(dataset, batch_size=2)
        
        try:
            metrics = trainer.train_epoch(dataloader)
            assert "loss" in metrics
            assert isinstance(metrics["loss"], float)
        except Exception as e:
            pytest.fail(f"训练轮次失败: {e}")
    
    def test_validate(self):
        """测试验证功能"""
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        device = torch.device("cpu")
        
        trainer = FusionTrainer(model, optimizer, criterion, device)
        
        opt = torch.randn(2, 12, 10, 32, 32)
        sar = torch.randn(2, 12, 5, 32, 32)
        doy = torch.randn(2, 12)
        labels = torch.randint(1, 7, (2, 32, 32))
        
        dataset = SimpleDictDataset(opt, sar, doy, labels)
        dataloader = DataLoader(dataset, batch_size=2)
        
        try:
            metrics = trainer.validate(dataloader)
            assert "OA" in metrics
            assert "mIoU" in metrics
            assert 0 <= metrics["OA"] <= 1
            assert 0 <= metrics["mIoU"] <= 1
        except Exception as e:
            pytest.fail(f"验证失败: {e}")

class TestTwoPhaseTrainer:
    """测试两阶段训练器"""
    
    def test_trainer_creation(self):
        """测试两阶段训练器创建"""
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        device = torch.device("cpu")
        
        trainer = TwoPhaseTrainer(model, device)
        
        assert trainer is not None
        assert trainer.model is model
        assert trainer.device == device
    
    def test_freeze_backbone(self):
        """测试冻结骨干网络"""
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        device = torch.device("cpu")
        
        trainer = TwoPhaseTrainer(model, device)
        trainer.freeze_backbone(freeze_layers=6)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        assert trainable < total
    
    def test_unfreeze_all(self):
        """测试解冻所有参数"""
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        device = torch.device("cpu")
        
        trainer = TwoPhaseTrainer(model, device)
        trainer.freeze_backbone(freeze_layers=6)
        trainer.unfreeze_all()
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        assert trainable == total
    
    def test_get_layerwise_params(self):
        """测试获取分层学习率参数"""
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=128, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
        device = torch.device("cpu")
        
        trainer = TwoPhaseTrainer(model, device)
        params = trainer.get_layerwise_params(lr=1e-3, backbone_lr_ratio=0.1)
        
        assert isinstance(params, list)
        assert len(params) >= 1