from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import contextlib
import torch
import numpy as np
from .logger import TrainingLogger
from .losses import DiceFocalLoss

class BaseTrainer(ABC):
    """
    训练器基类，定义训练流程的基本接口
    """
    
    def __init__(self, model, optimizer, criterion, device):
        """
        Args:
            model: 待训练的模型
            optimizer: 优化器
            criterion: 损失函数
            device: 训练设备
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = TrainingLogger()
    
    @abstractmethod
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            dataloader: 训练数据加载器
        
        Returns:
            训练指标字典
        """
        pass
    
    @abstractmethod
    def validate(self, dataloader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            dataloader: 验证数据加载器
        
        Returns:
            验证指标字典
        """
        pass
    
    def fit(self, train_loader, val_loader, epochs: int, 
            phase: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        训练主循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮次
            phase: 训练阶段（用于日志记录）
            verbose: 是否打印详细信息
        
        Returns:
            训练历史记录
        """
        history = {
            "train_loss": [],
            "val_miou": [],
            "val_oa": [],
            "best_miou": 0.0,
            "best_epoch": 0
        }
        
        for epoch in range(1, epochs + 1):
            if hasattr(self, '_current_epoch'):
                self._current_epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            history["train_loss"].append(train_metrics["loss"])
            history["val_miou"].append(val_metrics["mIoU"])
            history["val_oa"].append(val_metrics["OA"])
            
            if val_metrics["mIoU"] > history["best_miou"]:
                history["best_miou"] = val_metrics["mIoU"]
                history["best_epoch"] = epoch
                self.logger.log_best_model(epoch, val_metrics["mIoU"], val_metrics["OA"])
            
            if verbose:
                self.logger.log_epoch(
                    epoch, phase,
                    train_metrics["loss"],
                    val_metrics["mIoU"],
                    val_metrics["OA"],
                    val_metrics.get("IoU_per_class")
                )
        
        return history

class FusionTrainer(BaseTrainer):
    """
    Unified trainer for FusionCropNet V1/V4/V5/V5EDL/V5Pro.
    Auto-detects model type and uses correct forward signature.
    """

    def __init__(self, model, optimizer, criterion, device, scaler=None, total_epochs=80, use_amp: bool = None):
        super().__init__(model, optimizer, criterion, device)
        # V6: AMP now supports CPU (PyTorch 2.0+) and is explicitly configurable
        if use_amp is None:
            use_amp = (device.type == "cuda")
        self.use_amp = use_amp
        self._amp_device = device.type if hasattr(device, 'type') else str(device)

        if self.use_amp:
            self.amp_autocast = torch.amp.autocast(self._amp_device)
            if self._amp_device == "cuda":
                self.scaler = scaler if scaler else torch.amp.GradScaler("cuda")
            else:
                self.scaler = None  # CPU AMP doesn't use GradScaler
        else:
            self.scaler = None
            self.amp_autocast = contextlib.nullcontext()
        self.num_classes = 7
        self.total_epochs = total_epochs
        self._current_epoch = 0

    def _model_forward(self, opt, sar, dem, doy, cloud_mask=None, valid_count=None):
        """Dispatch to model forward with correct signature based on model type."""
        model_name = self.model.__class__.__name__
        if 'V5Pro' in model_name:
            return self.model(opt, sar, dem, doy, cloud_mask, valid_count,
                             epoch=self._current_epoch, total_epochs=self.total_epochs)
        elif 'V5EDL' in model_name or 'V5' in model_name or 'V6' in model_name:
            return self.model(opt, sar, dem, doy, cloud_mask, valid_count)
        else:
            return self.model(opt, sar, dem, doy)

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            opt = batch["opt"].to(self.device)
            sar = batch["sar"].to(self.device)
            doy = batch["doy"].to(self.device)
            y = batch["y"].to(self.device)
            dem = batch.get("dem", torch.zeros(
                opt.shape[0], 5, opt.shape[-2], opt.shape[-1])).to(self.device)
            cm = batch.get("cloud_mask", None)
            vc = batch.get("valid_count", None)
            if cm is not None: cm = cm.to(self.device)
            if vc is not None: vc = vc.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with self.amp_autocast:
                    out = self._model_forward(opt, sar, dem, doy, cm, vc)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = self.criterion(out, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self._model_forward(opt, sar, dem, doy, cm, vc)
                if isinstance(out, tuple):
                    out = out[0]
                loss = self.criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            total_loss += loss.item()

        return {"loss": total_loss / num_batches}

    def validate(self, dataloader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                opt = batch["opt"].to(self.device)
                sar = batch["sar"].to(self.device)
                doy = batch["doy"].to(self.device)
                y = batch["y"]
                dem = batch.get("dem", torch.zeros(
                    opt.shape[0], 5, opt.shape[-2], opt.shape[-1])).to(self.device)
                cm = batch.get("cloud_mask", None)
                vc = batch.get("valid_count", None)
                if cm is not None: cm = cm.to(self.device)
                if vc is not None: vc = vc.to(self.device)

                if self.use_amp:
                    with self.amp_autocast:
                        out = self._model_forward(opt, sar, dem, doy, cm, vc)
                        if isinstance(out, tuple):
                            out = out[0]
                        preds = out.argmax(dim=1).cpu() if out.shape[1] > 1 else out.cpu()
                else:
                    out = self._model_forward(opt, sar, dem, doy, cm, vc)
                    if isinstance(out, tuple):
                        out = out[0]
                    preds = out.argmax(dim=1).cpu() if out.shape[1] > 1 else out.cpu()

                all_preds.append(preds)
                all_labels.append(y)

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        return self._compute_metrics(preds, labels)
    
    def _compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算评估指标"""
        valid = labels != 0
        p = preds[valid]
        l = labels[valid]
        
        oa = (p == l).float().mean().item()
        
        iou_list = []
        for cls in range(1, self.num_classes):
            tp = ((p == cls) & (l == cls)).sum().float()
            fp = ((p == cls) & (l != cls)).sum().float()
            fn = ((p != cls) & (l == cls)).sum().float()
            iou = (tp / (tp + fp + fn + 1e-6)).item()
            iou_list.append(iou)
        
        return {
            "OA": oa,
            "mIoU": sum(iou_list) / len(iou_list) if iou_list else 0,
            "IoU_per_class": iou_list
        }

class TwoPhaseTrainer:
    """
    两阶段训练器
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.logger = TrainingLogger()
    
    def freeze_backbone(self, freeze_layers: int = 6):
        """冻结骨干网络"""
        from models.fusion_net import PretrainedWeightManager
        manager = PretrainedWeightManager(self.model)
        manager.freeze_backbone(freeze_layers)
    
    def unfreeze_all(self):
        """解冻所有参数"""
        from models.fusion_net import PretrainedWeightManager
        manager = PretrainedWeightManager(self.model)
        manager.unfreeze_all()
    
    def get_layerwise_params(self, lr: float, backbone_lr_ratio: float = 0.1):
        """获取分层学习率参数"""
        from models.fusion_net import PretrainedWeightManager
        manager = PretrainedWeightManager(self.model)
        return manager.get_layerwise_lr_params(lr, backbone_lr_ratio)
    
    def train_phase1(self, train_loader, val_loader, epochs: int = 20, lr: float = 1e-3):
        """阶段1：冻结骨干训练"""
        self.logger.info("="*60)
        self.logger.info(f"训练阶段1：冻结光学骨干（{epochs} epochs）")
        self.logger.info("="*60)
        
        self.freeze_backbone(freeze_layers=6)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        
        criterion = self._get_criterion()
        trainer = FusionTrainer(self.model, optimizer, criterion, self.device)
        
        return trainer.fit(train_loader, val_loader, epochs, phase=1)
    
    def train_phase2(self, train_loader, val_loader, epochs: int = 60, lr: float = 3e-4,
                     best_ckpt_path: Optional[str] = None):
        """阶段2：全量Fine-tune"""
        self.logger.info("="*60)
        self.logger.info(f"训练阶段2：全量Fine-tune（{epochs} epochs）")
        self.logger.info("="*60)
        
        if best_ckpt_path and self._check_ckpt_exists(best_ckpt_path):
            from models.fusion_net import PretrainedWeightManager
            manager = PretrainedWeightManager(self.model)
            manager.load_checkpoint(best_ckpt_path, strict=True)
            self.logger.info(f"已加载阶段1最优模型: {best_ckpt_path}")
        
        self.unfreeze_all()
        
        param_groups = self.get_layerwise_params(lr, backbone_lr_ratio=0.1)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        
        criterion = self._get_criterion()
        trainer = FusionTrainer(self.model, optimizer, criterion, self.device)
        
        return trainer.fit(train_loader, val_loader, epochs, phase=2)
    
    def _get_criterion(self):
        """获取损失函数 — EDL 模型用 EDLLoss，其他用 DiceFocalLoss"""
        model_name = self.model.__class__.__name__
        if 'EDL' in model_name or 'V6' in model_name:
            from models.fusion_net_v5_edl import EDLLoss
            return EDLLoss(num_classes=self.model.num_classes)
        return DiceFocalLoss(num_classes=7)
    
    def _check_ckpt_exists(self, path: str) -> bool:
        """检查检查点文件是否存在"""
        return __import__('os').path.exists(path)