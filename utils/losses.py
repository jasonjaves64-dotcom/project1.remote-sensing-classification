"""
损失函数模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler

class DiceFocalLoss(nn.Module):
    """
    Dice-Focal混合损失函数
    
    结合Focal Loss和Dice Loss，用于处理类别不平衡问题
    """
    
    def __init__(self, num_classes=7, gamma=2.0, alpha=0.5, ignore_index=0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def focal_loss(self, logits, targets):
        """计算Focal Loss"""
        ce = F.cross_entropy(logits, targets,
                            reduction="none",
                            ignore_index=self.ignore_index)
        p_t = torch.exp(-ce)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        return loss.mean()
    
    def dice_loss(self, logits, targets):
        """计算Dice Loss"""
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        dice = 0.0
        count = 0
        for cls in range(1, self.num_classes):
            pred = probs[:, cls][valid]
            true = (targets[valid] == cls).float()
            intersection = (pred * true).sum()
            dice += 1 - (2 * intersection + 1) / \
                    (pred.sum() + true.sum() + 1)
            count += 1
        return dice / count if count > 0 else 0.0
    
    def forward(self, logits, targets, weight_map=None):
        """前向传播"""
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return focal + 0.5 * dice

class WeightedDiceFocalLoss(nn.Module):
    """
    加权Dice-Focal混合损失函数
    
    支持传入weight_map对不同像素进行加权
    """
    
    def __init__(self, num_classes=7, gamma=2.0, alpha=0.5, ignore_index=0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def focal_loss(self, logits, targets, weight_map):
        """计算加权Focal Loss"""
        ce = F.cross_entropy(logits, targets,
                            reduction="none",
                            ignore_index=self.ignore_index)
        p_t = torch.exp(-ce)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        
        if weight_map is not None:
            valid = targets != self.ignore_index
            loss = loss * weight_map
            return loss[valid].mean()
        return loss.mean()
    
    def dice_loss(self, logits, targets, weight_map):
        """计算加权Dice Loss"""
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        
        if weight_map is not None:
            weights = weight_map[valid]
        else:
            weights = torch.ones_like(targets[valid], dtype=torch.float)
        
        dice = 0.0
        count = 0
        for cls in range(1, self.num_classes):
            pred = probs[:, cls][valid]
            true = (targets[valid] == cls).float()
            
            weighted_intersection = (pred * true * weights).sum()
            weighted_pred = (pred * weights).sum()
            weighted_true = (true * weights).sum()
            
            dice += 1 - (2 * weighted_intersection + 1) / \
                    (weighted_pred + weighted_true + 1)
            count += 1
        return dice / count if count > 0 else 0.0
    
    def forward(self, logits, targets, weight_map=None):
        """前向传播"""
        focal = self.focal_loss(logits, targets, weight_map)
        dice = self.dice_loss(logits, targets, weight_map)
        return focal + 0.5 * dice

class TverskyLoss(nn.Module):
    """
    Tversky损失函数 - 对Dice损失的改进，通过alpha和beta参数调整对假阳性和假阴性的惩罚
    """
    
    def __init__(self, alpha=0.7, beta=0.3, num_classes=7, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets, weight_map=None):
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        
        if weight_map is not None:
            weights = weight_map[valid]
        else:
            weights = torch.ones_like(targets[valid], dtype=torch.float)
        
        tversky = 0.0
        count = 0
        for cls in range(1, self.num_classes):
            pred = probs[:, cls][valid]
            true = (targets[valid] == cls).float()
            
            tp = (pred * true * weights).sum()
            fp = ((1 - true) * pred * weights).sum()
            fn = (true * (1 - pred) * weights).sum()
            
            tversky += 1 - (tp + 1) / (tp + self.alpha * fp + self.beta * fn + 1)
            count += 1
        return tversky / count if count > 0 else 0.0

class OhemCELoss(nn.Module):
    """
    Online Hard Example Mining Cross Entropy Loss
    """
    
    def __init__(self, num_classes=7, ignore_index=0, ohem_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ohem_ratio = ohem_ratio
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                            reduction="none",
                            ignore_index=self.ignore_index)
        
        valid = targets != self.ignore_index
        valid_ce = ce[valid]
        
        if valid_ce.numel() == 0:
            return ce.mean()
        
        n_min = int(valid_ce.numel() * self.ohem_ratio)
        _, topk_idx = torch.topk(valid_ce, n_min)
        hard_ce = valid_ce[topk_idx]
        
        return hard_ce.mean()

class StratifiedPatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int = 16,
                 min_per_class: int = 2, num_classes: int = 7):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.num_classes = num_classes

        print("StratifiedPatchSampler: scanning patch class membership...")
        self.class_to_patches = self._build_class_index()
        self.n_batches = len(dataset) // batch_size

    def _build_class_index(self):
        class_index = {c: [] for c in range(1, self.num_classes)}
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            y = sample['y']
            if isinstance(y, torch.Tensor):
                y = y.numpy()
            present = np.unique(y)
            present = present[(present > 0) & (present < 255)]
            for c in present:
                class_index[int(c)].append(idx)
        return class_index

    def __iter__(self):
        all_indices = list(range(len(self.dataset)))

        for _ in range(self.n_batches):
            batch = []

            for cls, patches in self.class_to_patches.items():
                if len(patches) == 0:
                    continue
                n = min(self.min_per_class, len(patches))
                sampled = np.random.choice(patches, n, replace=False)
                batch.extend(sampled.tolist())

            remaining = self.batch_size - len(batch)
            if remaining > 0:
                extra = np.random.choice(all_indices, remaining, replace=False)
                batch.extend(extra.tolist())

            np.random.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self):
        return self.n_batches