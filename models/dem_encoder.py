"""
DEM Encoder and terrain feature utilities.
Basic blocks imported from ._base; unique FiLMLayer + ThreeWayFusion kept here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import ConvBNGELU, SEBlock, DEMEncoder


class FiLMLayer(nn.Module):
    """FiLM variant using sigmoid-gamma + tanh-beta (different from _base.FiLM)."""
    def __init__(self, cond_ch, feat_ch):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Conv2d(cond_ch, feat_ch, 1, bias=False), nn.Sigmoid())
        self.beta = nn.Sequential(
            nn.Conv2d(cond_ch, feat_ch, 1, bias=True), nn.Tanh())

    def forward(self, feat, cond):
        if cond.shape[-2:] != feat.shape[-2:]:
            cond = F.interpolate(cond, feat.shape[-2:], mode='bilinear', align_corners=False)
        return feat * (1 + self.gamma(cond)) + self.beta(cond)


class ThreeWayFusion(nn.Module):
    """DEM-conditioned fusion gate (V4-compatible, uses FiLMLayer)."""
    def __init__(self, feat_ch, dem_ch=128):
        super().__init__()
        self.film = FiLMLayer(dem_ch, feat_ch)
        self.gate = nn.Sequential(nn.Conv2d(feat_ch + dem_ch, feat_ch, 1), nn.Sigmoid())
        self.mix = nn.Sequential(ConvBNGELU(feat_ch + dem_ch, feat_ch), SEBlock(feat_ch))
        self.norm = nn.GroupNorm(32, feat_ch)

    def forward(self, fused, dem_feat):
        if dem_feat.shape[-2:] != fused.shape[-2:]:
            dem_feat = F.interpolate(dem_feat, fused.shape[-2:], mode='bilinear', align_corners=False)
        filmed = self.film(fused, dem_feat)
        cat = torch.cat([filmed, dem_feat], dim=1)
        gate = self.gate(cat)
        out = gate * self.mix(cat) + (1 - gate) * fused
        return self.norm(out)


def compute_dem_bands(elevation, pixel_size_m=30.0):
    """Compute 5-band DEM features: elevation, slope, cos(aspect), sin(aspect), TWI."""
    import numpy as np
    dy, dx = np.gradient(elevation, pixel_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dx, dy) % (2 * np.pi)
    twi = np.log(1.0 / np.maximum(np.tan(slope_rad), 0.001))

    def norm(a):
        return (a - a.min()) / (a.max() - a.min() + 1e-6)

    return np.stack([
        norm(elevation),
        slope_rad / (np.pi / 2),
        np.cos(aspect_rad),
        np.sin(aspect_rad),
        norm(twi)
    ], axis=0).astype(np.float32)
