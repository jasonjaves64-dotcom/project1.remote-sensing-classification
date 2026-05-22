"""Lightweight multi-task heads for V6.

Heads take pre_head features (B, 64, H, W) and produce auxiliary predictions.
All heads are < 0.1M params each.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import ConvBNGELU


class LAIRegressionHead(nn.Module):
    """Predict Leaf Area Index from shared features."""
    def __init__(self, in_ch: int = 64, hidden: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNGELU(in_ch, hidden),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Softplus()  # LAI >= 0
        )

    def forward(self, x):
        return self.mlp(self.conv(x)).squeeze(-1)  # (B,)


class GrowthStageHead(nn.Module):
    """Predict growth stage (emergence/vegetative/reproductive/grainfill/maturity)."""
    def __init__(self, in_ch: int = 64, hidden: int = 32, num_stages: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNGELU(in_ch, hidden),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, num_stages)
        )

    def forward(self, x):
        return self.mlp(self.conv(x))  # (B, num_stages)


class BoundaryHead(nn.Module):
    """Predict field boundaries from shared features."""
    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNGELU(in_ch, 32),
            nn.Conv2d(32, 1, 1),  # Don't use BatchNorm on single channel
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)  # (B, 1, H, W)


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss.

    Uses homoscedastic task uncertainty (Kendall et al. 2018):
    L_total = sum (1/(2*sigma_i^2) * L_i + log(sigma_i)) for regression tasks
    L_total = sum (1/sigma_i^2 * L_i + log(sigma_i)) for classification tasks

    With 5 tasks: crop(EDL), ndvi(MSE), lai(Huber), growth(CE), boundary(Dice+BCE)
    """
    def __init__(self, num_tasks: int = 5):
        super().__init__()
        # Learnable log variances (one per task).
        # IMPORTANT: log_vars[i] corresponds to loss dict key in insertion order:
        #   [0]='crop', [1]='ndvi', [2]='lai', [3]='growth', [4]='boundary'
        # If dict order changes, the mapping WILL be silently wrong.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: dict) -> torch.Tensor:
        """Compute weighted total loss.

        Args:
            losses: dict with keys 'crop', 'ndvi', 'lai', 'growth', 'boundary'
        Returns:
            total_loss: scalar
        """
        total = 0.0
        for i, (name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            if name in ('ndvi', 'lai'):  # regression tasks
                total += 0.5 * precision * loss + 0.5 * self.log_vars[i]
            else:  # classification/segmentation tasks
                total += precision * loss + self.log_vars[i]
        return total
