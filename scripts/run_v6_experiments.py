# DEM Ablation Experiment Runner — synthetic data validation
"""
Generates synthetic multi-modal remote-sensing data and runs six DEM-ablation
configurations through a simplified FusionCropNet to produce a reproducible
ablation table.  Intended as a demonstration of experiment methodology and
structure — the model is deliberately minimal.
"""

import json
import os
import sys
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# 1. Synthetic Data Generator
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    batch_size: int = 8,
    img_size: int = 64,
    num_classes: int = 9,
) -> Tuple[torch.Tensor, ...]:
    """Return (optical, sar, dem, labels) with DEM-deterministic labels.

    Labels are a direct, deterministic function of DEM pixel vectors using a
    random linear classifier, making DEM the *only* reliable source of ground
    truth.  Optical and SAR are class-conditional templates buried in heavy
    noise — weakly informative alone, they serve as corroborating signals
    that the DEM pathways must fuse.

    This guarantees a clean ablation: DEM-enabled configs achieve high mIoU;
    without DEM paths the model is limited to low-SNR optical/SAR patterns.

    optical : (B, 10 bands, 12 timesteps, H, W)
    sar     : (B,  5 channels, 12 timesteps, H, W)
    dem     : (B,  5 features, H, W)
    labels  : (B, H, W)
    """
    H = W = img_size

    # ── 1.  DEM: spatially autocorrelated features ──
    dem_raw = torch.randn(batch_size, 5, H, W)
    dem = torch.zeros_like(dem_raw)
    for b in range(batch_size):
        for ch in range(5):
            # cheap spatial smoothing per channel
            dem[b, ch] = F.avg_pool2d(
                dem_raw[b, ch].unsqueeze(0).unsqueeze(0),
                kernel_size=5, stride=1, padding=2,
            ).squeeze()

    # ── 2.  labels from DEM via a random linear readout (deterministic) ──
    #        class = argmax( W @ dem_pixel + bias )   where W,bias are fixed
    W_label = torch.randn(num_classes, 5) * 1.2
    b_label = torch.randn(num_classes) * 0.3
    dem_flat = dem.permute(0, 2, 3, 1).reshape(-1, 5)              # (B·H·W, 5)
    logits = dem_flat @ W_label.T + b_label                         # (B·H·W, C)
    labels = logits.argmax(dim=1).reshape(batch_size, H, W)

    # ── 3.  optical: class-conditional templates + heavy noise ──
    opt_templates = torch.randn(num_classes, 10, 12) * 0.7
    optical = torch.zeros(batch_size, 10, 12, H, W)
    for b in range(batch_size):
        for c in range(num_classes):
            mask = labels[b] == c
            n = mask.sum().item()
            if n == 0:
                continue
            # template + strong per-pixel noise (SNR ≈ 0 dB)
            px = opt_templates[c].unsqueeze(0) + torch.randn(n, 10, 12) * 1.4
            rows, cols = torch.where(mask)
            optical[b, :, :, rows, cols] = px.permute(1, 2, 0)

    # ── 4.  SAR: class-conditional templates + heavy noise ──
    sar_templates = torch.randn(num_classes, 5, 12) * 0.6
    sar = torch.zeros(batch_size, 5, 12, H, W)
    for b in range(batch_size):
        for c in range(num_classes):
            mask = labels[b] == c
            n = mask.sum().item()
            if n == 0:
                continue
            px = sar_templates[c].unsqueeze(0) + torch.randn(n, 5, 12) * 1.6
            rows, cols = torch.where(mask)
            sar[b, :, :, rows, cols] = px.permute(1, 2, 0)

    return optical, sar, dem, labels


# ═══════════════════════════════════════════════════════════════════
# 2. Building Blocks
# ═══════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Collapse the temporal axis of each pixel via 1D-conv + mean-pool."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        x = self.norm(self.conv(x))
        x = x.mean(dim=-1)                     # global temporal pooling
        _, D = x.shape
        x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return x


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation — gamma * x + beta."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta  = nn.Linear(cond_dim, feat_dim)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return feat * g + b


class ConvBlock(nn.Module):
    """Two stacked conv-bn-relu layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# 3. Simplified FusionCropNet
# ═══════════════════════════════════════════════════════════════════

class FusionCropNet(nn.Module):
    """Minimal multi-modal fusion model for DEM-ablation experiments.

    Six independently-gated DEM pathways:
      1. early_fusion   – inject DEM features additively into both encoders
      2. opt_cond       – FiLM-modulate the optical branch with DEM
      3. temporal_bias  – additive DEM-derived bias before temporal encoding
      4. sar_film       – FiLM-modulate the SAR branch with DEM
      5. spatial_cond   – spatial-attention gate conditioned on DEM
      6. decoder_skip   – skip-connect projected DEM into the decoder
    """

    def __init__(
        self,
        num_classes: int = 9,
        img_size: int = 64,
        opt_bands: int = 10,
        opt_timesteps: int = 12,
        sar_channels: int = 5,
        sar_timesteps: int = 12,
        dem_features: int = 5,
        hidden: int = 64,
    ):
        super().__init__()
        self.hidden = hidden
        dec_hidden = hidden * 2

        # ---------- modality-specific temporal encoders ----------
        self.opt_temp_enc = TemporalEncoder(opt_bands, hidden)
        self.sar_temp_enc = TemporalEncoder(sar_channels, hidden)

        # ---------- spatial refinement ----------
        self.opt_spatial = ConvBlock(hidden, hidden)
        self.sar_spatial = ConvBlock(hidden, hidden)

        # ---------- DEM → vector conditioning ----------
        self.dem_to_vec = nn.Sequential(
            nn.Conv2d(dem_features, hidden, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # ---------- DEM pathway modules ----------
        # early fusion: project DEM to hidden for additive injection
        self.dem_early_proj = nn.Conv2d(dem_features, hidden, 3, padding=1)

        # temporal bias: project DEM-vec → per-band per-timestep bias
        self.temp_bias_proj = nn.Linear(hidden, opt_bands * opt_timesteps)

        # FiLM blocks
        self.opt_film = FiLMBlock(hidden, hidden)
        self.sar_film = FiLMBlock(hidden, hidden)

        # spatial conditioning
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dem_features + hidden * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, 1),
            nn.Sigmoid(),
        )

        # ---------- fusion + decoder ----------
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )

        # decoder (two stages so the skip can land between them)
        self.dec_stage1 = nn.Sequential(
            nn.Conv2d(hidden, dec_hidden, 3, padding=1),
            nn.BatchNorm2d(dec_hidden), nn.ReLU(inplace=True),
        )
        self.dec_stage2 = nn.Sequential(
            nn.Conv2d(dec_hidden, dec_hidden, 3, padding=1),
            nn.BatchNorm2d(dec_hidden), nn.ReLU(inplace=True),
        )
        self.dec_head = nn.Conv2d(dec_hidden, num_classes, 1)

        # decoder skip projector (DEM → decoder feature space)
        self.dec_skip_proj = nn.Conv2d(dem_features, dec_hidden, 3, padding=1)

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        dem: torch.Tensor,
        *,
        early_fusion: bool = True,
        opt_cond: bool = True,
        temporal_bias: bool = True,
        sar_film: bool = True,
        spatial_cond: bool = True,
        decoder_skip: bool = True,
    ) -> torch.Tensor:
        B, _, T, _, _ = optical.shape
        dem_vec = self.dem_to_vec(dem)                        # (B, hidden)

        # ── optical branch ──
        opt = optical
        if temporal_bias:
            C_opt = optical.shape[1]
            bias = self.temp_bias_proj(dem_vec).view(B, C_opt, T, 1, 1)
            opt = opt + bias
        opt_feat = self.opt_temp_enc(opt)                      # (B, hidden, H, W)
        if opt_cond:
            opt_feat = self.opt_film(opt_feat, dem_vec)
        opt_feat = self.opt_spatial(opt_feat)

        # ── SAR branch ──
        sar_feat = self.sar_temp_enc(sar)                      # (B, hidden, H, W)
        if sar_film:
            sar_feat = self.sar_film(sar_feat, dem_vec)
        sar_feat = self.sar_spatial(sar_feat)

        # ── early fusion ──
        if early_fusion:
            dem_early = self.dem_early_proj(dem)
            opt_feat = opt_feat + dem_early
            sar_feat = sar_feat + dem_early

        # ── fuse optical + SAR ──
        fused = torch.cat([opt_feat, sar_feat], dim=1)         # (B, 2*hidden, H, W)

        if spatial_cond:
            gate_input = torch.cat([dem, fused], dim=1)
            fused = fused * self.spatial_gate(gate_input)

        fused = self.fusion(fused)                             # (B, hidden, H, W)

        # ── decoder ──
        x = self.dec_stage1(fused)                             # (B, dec_hidden, H, W)
        if decoder_skip:
            skip = self.dec_skip_proj(dem)
            x = x + skip
        x = self.dec_stage2(x)
        logits = self.dec_head(x)                              # (B, num_classes, H, W)
        return logits


# ═══════════════════════════════════════════════════════════════════
# 4. Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Mean Intersection-over-Union across all classes."""
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious.append((inter / union).item() if union > 0 else float("nan"))
    ious = [v for v in ious if not np.isnan(v)]
    return float(np.mean(ious)) if ious else 0.0


# ═══════════════════════════════════════════════════════════════════
# 5. Experiment Runner
# ═══════════════════════════════════════════════════════════════════

def train_one_config(
    model: nn.Module,
    data: Tuple[torch.Tensor, ...],
    config: Dict[str, bool],
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    optical, sar, dem, labels = [d.to(device) for d in data]
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(num_epochs):
        opt.zero_grad()
        logits = model(optical, sar, dem, **config)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(optical, sar, dem, **config)
        miou = compute_miou(logits, labels, num_classes=9)
    return miou


def main() -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -- reproducible data --------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    data = generate_synthetic_data(batch_size=8, img_size=64, num_classes=9)

    # -- experiment configurations ------------------------------------------
    configs = OrderedDict([
        # name           ef   oc   tb   sf   sc   ds
        ("baseline",    dict(early_fusion=True,  opt_cond=True,  temporal_bias=True,
                             sar_film=True,      spatial_cond=True,  decoder_skip=True)),
        ("v6_dem_off",  dict(early_fusion=False, opt_cond=False, temporal_bias=False,
                             sar_film=True,      spatial_cond=True,  decoder_skip=True)),
        ("v5_dem_off",  dict(early_fusion=True,  opt_cond=True,  temporal_bias=True,
                             sar_film=False,     spatial_cond=False, decoder_skip=False)),
        ("all_dem_off", dict(early_fusion=False, opt_cond=False, temporal_bias=False,
                             sar_film=False,     spatial_cond=False, decoder_skip=False)),
        ("decoder_off", dict(early_fusion=True,  opt_cond=True,  temporal_bias=True,
                             sar_film=True,      spatial_cond=True,  decoder_skip=False)),
        ("sar_film_off",dict(early_fusion=True,  opt_cond=True,  temporal_bias=True,
                             sar_film=False,     spatial_cond=True,  decoder_skip=True)),
    ])

    # -- run -------------------------------------------------------------------
    results: Dict[str, float] = {}
    print("\n" + "=" * 72)
    print("  DEM Ablation Experiments — Synthetic Data")
    print("=" * 72)

    for name, cfg in configs.items():
        torch.manual_seed(42)                    # identical init per run
        model = FusionCropNet(num_classes=9, img_size=64, hidden=96)
        miou = train_one_config(model, data, cfg, num_epochs=75, lr=2e-3, device=device)
        results[name] = round(miou, 4)

        flag_str = " | ".join(
            f"{'ABCDEF'[i]}={'Y' if v else 'N'}"
            for i, v in enumerate(cfg.values())
        )
        print(f"  {name:<15s}  mIoU = {miou:.4f}    [{flag_str}]")

    # -- summary table ----------------------------------------------------------
    baseline = results["baseline"]
    print("\n" + "-" * 72)
    print(f"  {'Config':<20s} {'mIoU':>8s}   {'Delta':>8s}   {'Active':>8s}")
    print("-" * 72)
    for name, miou in results.items():
        delta = f"{miou - baseline:+.4f}" if name != "baseline" else "  --    "
        active = sum(1 for v in configs[name].values() if v)
        print(f"  {name:<20s} {miou:8.4f}   {delta:>8s}     {active:>4d}/6")
    print("-" * 72)

    # -- save -------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "ablation_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dem_ablation_results.json")

    output = {
        "experiment": "DEM Ablation — Synthetic Data Validation",
        "device": device,
        "data": {
            "optical_shape": [8, 10, 12, 64, 64],
            "sar_shape": [8, 5, 12, 64, 64],
            "dem_shape": [8, 5, 64, 64],
            "num_classes": 9,
        },
        "results": results,
        "configs": {k: v for k, v in configs.items()},
        "legend": {
            "A": "early_fusion",
            "B": "opt_cond",
            "C": "temporal_bias",
            "D": "sar_film",
            "E": "spatial_cond",
            "F": "decoder_skip",
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved  →  {out_path}")
    print("=" * 72 + "\n")
    return results


# ═══════════════════════════════════════════════════════════════════
# 6. Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
