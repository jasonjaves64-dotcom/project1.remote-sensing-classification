"""TemporalLite — lightweight temporal encoder for high-resolution features.

Replaces time_average() on SAR s1(64ch) and s2(128ch) with a learnable
1D depthwise convolution + gated temporal pooling. ~0.1M params each.

See: V6-时序编码瓶颈-方案评审.md Section 6.4
"""
import torch
import torch.nn as nn


class TemporalLite(nn.Module):
    """Extremely lightweight temporal encoder.

    Complexity: O(T x D) via depthwise conv — vs O(T x D^2) for Transformer FFN.

    Args:
        d_model: feature dimension
        k: convolution kernel size (temporal window), default 3
    """
    def __init__(self, d_model: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, k,
            padding=k // 2,
            groups=d_model,     # depthwise
            bias=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequence into a single feature vector.

        Args:
            x: (N, T, D) — N sequences (e.g. BxHxW pixels),
               T timesteps, D channels

        Returns:
            (N, D) — temporally pooled features
        """
        # Conv1d expects (N, D, T)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.norm(x)
        # Weighted temporal mean (learned gate vs fixed time_average)
        return x.mean(dim=1) * self.gate
