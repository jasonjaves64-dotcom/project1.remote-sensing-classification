# Fusion Net V5 Pro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build FusionCropNetV5Pro — V5EDL enhanced with pluggable backbones, multi-scale fusion, CARAFE upsampling, dynamic dropout, and adaptive KL annealing.

**Architecture:** Extend `_base.py` with pluggable backbone support and CARAFE. Create `fusion_net_v5pro.py` that subclasses V5EDL, adding multi-scale cross-modal fusion at 2 scales and dynamic regularization schedules.

**Tech Stack:** PyTorch, timm, einops — same as existing project.

---

### Task 1: Pluggable Backbone in `_base.py`

**Files:**
- Modify: `models/_base.py` — `_BACKBONE_CHANNELS` dict and `OpticalEncoder`

- [ ] **Step 1: Expand backbone registry**

Add entries to `_BACKBONE_CHANNELS` in `models/_base.py`:

```python
_BACKBONE_CHANNELS = {
    "resnet18":        [64, 128, 256, 512],
    "resnet34":        [64, 128, 256, 512],
    "resnet50":        [256, 512, 1024, 2048],
    "resnet101":       [256, 512, 1024, 2048],
    "convnext_tiny":   [96, 192, 384, 768],
    "convnext_small":  [96, 192, 384, 768],
    "convnext_base":   [128, 256, 512, 1024],
    "swin_tiny_patch4_window7_224":  [96, 192, 384, 768],
    "swin_small_patch4_window7_224": [96, 192, 384, 768],
    "efficientnet_b0": [24, 40, 112, 320],
    "efficientnet_b4": [24, 48, 120, 336],
    "maxvit_tiny":     [64, 128, 256, 512],
    "maxvit_small":    [64, 128, 256, 512],
}
```

- [ ] **Step 2: Update OpticalEncoder to handle backbone feature dims**

The current `OpticalEncoder` hardcodes `feat_dim // 2` for sp2/sp3. With variable backbone channels, the FPN output is always `feat_dim`, but we need to make sure the sp2/sp3 projections match. No change needed — FPN already unifies to `feat_dim`.

But we do need to handle `timm.create_model` for Transformer-based backbones (Swin/ViT) which use different feature extraction APIs. Add a helper:

```python
class OpticalEncoder(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        # Check if backbone is CNN (features_only) or Transformer
        bb_ch = _BACKBONE_CHANNELS.get(backbone, [256, 512, 1024, 2048])
        is_transformer = any(t in backbone for t in ['swin', 'vit', 'maxvit'])
        
        if is_transformer:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        else:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        
        # Patch first conv for multispectral input
        orig = self.backbone.conv1 if hasattr(self.backbone, 'conv1') else None
        if orig is not None:
            w = orig.weight.data
            nw = w.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
            nc = nn.Conv2d(in_ch, w.shape[0], orig.kernel_size, orig.stride, orig.padding, bias=False)
            nc.weight.data = nw
            self.backbone.conv1 = nc
        elif hasattr(self.backbone, 'patch_embed'):
            # Swin/ViT: patch embedding
            pe = self.backbone.patch_embed
            old_proj = pe.proj
            new_proj = nn.Conv2d(in_ch, old_proj.out_channels, old_proj.kernel_size,
                                 old_proj.stride, old_proj.padding, bias=old_proj.bias is not None)
            nw = old_proj.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
            new_proj.weight.data = nw
            pe.proj = new_proj
        
        self.fpn = FPN(bb_ch, feat_dim)
        self.sp2 = nn.Conv2d(feat_dim, feat_dim // 2, 1)
        self.sp3 = nn.Conv2d(feat_dim, feat_dim // 2, 1)
```

- [ ] **Step 3: Verify compilation**

Run: `python -m py_compile models/_base.py`
Expected: OK

- [ ] **Step 4: Run smoke test with different backbones**

```bash
python -c "
from models._base import OpticalEncoder
import torch
for bb in ['resnet50', 'convnext_tiny', 'swin_tiny_patch4_window7_224']:
    try:
        enc = OpticalEncoder(10, 512, bb, pretrained=False)
        out = enc(torch.randn(2, 10, 64, 64))
        print(f'{bb}: main={out[0].shape}, p2={out[1].shape}, p3={out[2].shape}')
    except Exception as e:
        print(f'{bb}: FAILED - {e}')
"
```

- [ ] **Step 5: Commit**

```bash
git add models/_base.py
git commit -m "feat: add pluggable backbone support (ConvNeXt/Swin/EfficientNet/MaxViT)"
```

---

### Task 2: CARAFE Upsampling in `_base.py`

**Files:**
- Modify: `models/_base.py` — add `CARAFEUp` class and update `Decoder`

- [ ] **Step 1: Implement CARAFE lightweight upsampler**

CARAFE = Content-Aware ReAssembly of FEatures. Lightweight implementation:

```python
class CARAFEUp(nn.Module):
    """Lightweight CARAFE: content-adaptive upsampling. Avoids checkerboard artifacts."""
    def __init__(self, in_ch, scale=2, kernel_size=3, compressed_ch=64):
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        k_up = kernel_size ** 2 * scale ** 2
        self.compress = nn.Conv2d(in_ch, compressed_ch, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(compressed_ch, compressed_ch, kernel_size, padding=kernel_size//2),
            nn.GELU())
        self.kernel_pred = nn.Conv2d(compressed_ch, k_up, 1)
        self.unfold = nn.Unfold(kernel_size, padding=kernel_size//2)
        self.pix_shuf = nn.PixelShuffle(scale)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.scale
        # Predict per-pixel upsampling kernels
        compressed = self.compress(x)
        encoded = self.encoder(compressed)
        kernels = self.kernel_pred(encoded)  # (B, k_up, H, W)
        # Reshape to (B, H*W, s^2, k^2)
        k = self.kernel_size
        kernels = kernels.permute(0, 2, 3, 1).reshape(B * H * W, s * s, k * k)
        kernels = F.softmax(kernels, dim=-1)
        # Unfold input into patches
        patches = self.unfold(x)  # (B, C*k*k, H*W)
        patches = patches.permute(0, 2, 1).reshape(B * H * W, C, k * k)
        # Apply kernels: (B*H*W, s^2, k^2) @ (B*H*W, k^2, C) -> (B*H*W, s^2, C)
        out = torch.bmm(kernels, patches.transpose(1, 2))  # (B*H*W, s^2, C)
        out = out.reshape(B, H, W, s, s, C).permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(B, C, H * s, W * s)
        return out
```

- [ ] **Step 2: Update Decoder to use CARAFE**

Replace `nn.ConvTranspose2d` with `CARAFEUp`:

```python
class Decoder(nn.Module):
    def __init__(self, feat_dim, sar_ch_list, n_heads=8, win=4, use_carafe=True):
        super().__init__()
        od = feat_dim // 2
        sc0, sc1 = sar_ch_list
        if use_carafe:
            self.up1 = CARAFEUp(feat_dim, scale=2, compressed_ch=64)
            self.up2 = CARAFEUp(64, scale=2, compressed_ch=64)
        else:
            self.up1 = nn.ConvTranspose2d(feat_dim, 64, 2, stride=2)
            self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.sr = SpatialRefinement(64, n_heads, win)
        self.merge1 = ConvBNGELU(64 + od + sc1, 64)
        self.merge2 = ConvBNGELU(64 + od + sc0, 64)
        self.pre_head_ch = 64
    # ... forward unchanged
```

- [ ] **Step 3: Verify compilation and shapes**

```bash
python -c "
from models._base import CARAFEUp, Decoder
import torch
carafe = CARAFEUp(512, 2)
out = carafe(torch.randn(1, 512, 16, 16))
print(f'CARAFE: {tuple(out.shape)}')  # Expected: (1, 512, 32, 32)
dec = Decoder(512, [64, 128], use_carafe=True)
print('Decoder with CARAFE OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add models/_base.py
git commit -m "feat: add CARAFE upsampling to Decoder"
```

---

### Task 3: Multi-Scale Cross-Modal Fusion

**Files:**
- Create: `models/fusion_net_v5pro.py`
- Modify: `models/__init__.py`

- [ ] **Step 1: Create MultiScaleFusion module**

Add to the new `models/fusion_net_v5pro.py`:

```python
class MultiScaleFusion(nn.Module):
    """2-scale cross-modal fusion: semantic level + mid level."""
    def __init__(self, ch_high=512, ch_mid_opt=256, ch_mid_sar=128):
        super().__init__()
        # High-level: reuse existing CrossModalAttention + DEMSpatialConditioner
        # (already in V5EDL flow)
        
        # Mid-level: lightweight cross-gating
        self.mid_proj_opt = nn.Conv2d(ch_mid_opt, ch_mid_sar, 1)
        self.mid_proj_sar = nn.Conv2d(ch_mid_sar, ch_mid_sar, 1)
        self.mid_gate = nn.Sequential(
            nn.Conv2d(ch_mid_sar * 2, ch_mid_sar, 1), nn.Sigmoid())
        self.mid_out = ConvBNGELU(ch_mid_sar, ch_mid_sar)
    
    def forward(self, opt_mid, sar_mid):
        """Fuse mid-level features (H/2 × W/2 scale)."""
        if sar_mid.shape[-2:] != opt_mid.shape[-2:]:
            sar_mid = F.interpolate(sar_mid, opt_mid.shape[-2:], mode='bilinear', align_corners=False)
        opt_p = self.mid_proj_opt(opt_mid)
        sar_p = self.mid_proj_sar(sar_mid)
        g = self.mid_gate(torch.cat([opt_p, sar_p], dim=1))
        return self.mid_out(g * opt_p + (1 - g) * sar_p)
```

- [ ] **Step 2: Create FusionCropNetV5Pro model**

```python
class FusionCropNetV5Pro(FusionCropNetV5EDL):
    """V5 Pro: V5EDL + multi-scale fusion + dynamic regularization."""
    
    def __init__(self, *args, use_carafe=True, **kwargs):
        # Extract V5Pro-specific kwargs before passing to parent
        self.dynamic_dropout = kwargs.pop('dynamic_dropout', True)
        self.adaptive_kl = kwargs.pop('adaptive_kl', True)
        super().__init__(*args, **kwargs)
        
        # Replace decoder with CARAFE version
        if use_carafe:
            self.decoder = Decoder(
                self.decoder.up1.in_channels if hasattr(self.decoder, 'up1') else kwargs.get('feat_dim', 512),
                sar_ch_list=self.sar_enc.out_channels_list[:2],
                n_heads=8, win=kwargs.get('win_size', 4), use_carafe=True)
        
        # Add mid-scale fusion
        self.mid_fusion = MultiScaleFusion(
            ch_high=kwargs.get('feat_dim', 512),
            ch_mid_opt=kwargs.get('feat_dim', 512) // 2,
            ch_mid_sar=128)
    
    def _get_drop_p(self, epoch, total_epochs):
        """Curriculum dropout schedule: low→high→low."""
        if not self.dynamic_dropout:
            return self.drop_timestep_p
        import math
        progress = epoch / max(total_epochs, 1)
        return 0.05 + 0.15 * math.sin(math.pi * progress)
    
    def forward(self, opt_seq, sar_seq, dem, doy,
                cloud_mask=None, valid_count=None, epoch=0, total_epochs=80,
                modality_mask=None):
        # Adjust dropout for current epoch
        if self.dynamic_dropout:
            self.drop_timestep_p = self._get_drop_p(epoch, total_epochs)
        return super().forward(opt_seq, sar_seq, dem, doy,
                              cloud_mask, valid_count, epoch, modality_mask)
```

Wait, this is getting complex with the `_encode` method. Let me rethink.

Actually, the cleanest approach: V5Pro does NOT subclass V5EDL. Instead, it's a standalone model that reuses `_base` components plus adds new ones. This avoids the complexity of overriding the giant `_encode` method.

Let me redesign as a standalone model.

- [ ] **Step 2 (revised): Create FusionCropNetV5Pro as standalone**

```python
"""
FusionCropNetV5Pro — Enhanced multi-modal crop classification.
Extends V5EDL architecture with: pluggable backbone, multi-scale fusion,
CARAFE upsampling, dynamic dropout, adaptive KL.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math
from ._base import (
    ConvBNGELU, SEBlock, FiLM, IRB,
    DEMEncoder, FPN, OpticalEncoder, SAREncoder,
    FourierDOYEncoding, ObsQualityToken, TemporalEncoderStream,
    CrossModalAttention, DEMSpatialConditioner, LateFusion,
    SpatialRefinement, Decoder, PhenologyAuxHead, CARAFEUp,
    time_average, _BACKBONE_CHANNELS,
)
from .fusion_net_v5_edl import (
    EDLHead, EDLLoss, dirichlet_to_predictions, evidence_level_fusion
)


class MultiScaleFusion(nn.Module):
    """Mid-level cross-modal gating fusion."""
    def __init__(self, ch_opt=256, ch_sar=128):
        super().__init__()
        self.proj_opt = nn.Conv2d(ch_opt, ch_sar, 1)
        self.proj_sar = nn.Conv2d(ch_sar, ch_sar, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(ch_sar * 2, ch_sar, 1), nn.Sigmoid())
        self.out = ConvBNGELU(ch_sar, ch_sar)
    
    def forward(self, opt_mid, sar_mid):
        if sar_mid.shape[-2:] != opt_mid.shape[-2:]:
            sar_mid = F.interpolate(sar_mid, opt_mid.shape[-2:],
                                    mode='bilinear', align_corners=False)
        o = self.proj_opt(opt_mid)
        s = self.proj_sar(sar_mid)
        g = self.gate(torch.cat([o, s], dim=1))
        return self.out(g * o + (1 - g) * s)


class FusionCropNetV5Pro(nn.Module):
    def __init__(self, opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                 feat_dim=512, backbone='resnet50', pretrained=True,
                 n_heads=16, win_size=4, n_layers=4, max_obs=24, n_freqs=4,
                 drop_timestep_p=0.1, edl_dropout_p=0.3,
                 edl_lambda_max=0.5, edl_anneal_ep=50,
                 modality_dropout_p=0.0, use_carafe=True,
                 dynamic_dropout=True, adaptive_kl=True):
        super().__init__()
        self.dynamic_dropout = dynamic_dropout
        self.adaptive_kl = adaptive_kl
        self.drop_timestep_p = drop_timestep_p
        
        dem_ch = 128
        self.dem_enc = DEMEncoder(dem_ch_in, dem_ch)
        self.opt_enc = OpticalEncoder(opt_ch, feat_dim, backbone, pretrained)
        self.sar_enc = SAREncoder(sar_ch, 32, feat_dim, dem_ch)
        self.opt_temporal = TemporalEncoderStream(feat_dim, n_heads=8, n_layers=n_layers,
                                                   max_obs=max_obs, n_freqs=n_freqs)
        self.sar_temporal = TemporalEncoderStream(feat_dim, n_heads=8, n_layers=n_layers,
                                                   max_obs=max_obs, n_freqs=n_freqs)
        self.cross_modal = CrossModalAttention(feat_dim, n_heads, win_size)
        self.dem_cond = DEMSpatialConditioner(feat_dim, dem_ch)
        self.mid_fusion = MultiScaleFusion(feat_dim // 2, 128)
        self.late_fuse = LateFusion(feat_dim)
        self.decoder = Decoder(feat_dim, self.sar_enc.out_channels_list[:2],
                               n_heads=8, win=win_size, use_carafe=use_carafe)
        self.edl_head = EDLHead(self.decoder.pre_head_ch, num_classes, edl_dropout_p)
        self.edl_loss_fn = EDLLoss(num_classes, edl_lambda_max, edl_anneal_ep)
        self.pheno_aux = PhenologyAuxHead(feat_dim, aux_weight=0.3)
        self.consistency_proj = nn.Linear(feat_dim, 1)
        self.consistency_target_proj = nn.Linear(feat_dim, 1)
        self.modality_dropout_p = modality_dropout_p
        
        self.placeholder_opt = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.placeholder_sar = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.placeholder_dem_feat = nn.Parameter(torch.zeros(1, dem_ch, 1, 1))
        nn.init.trunc_normal_(self.placeholder_opt, std=0.02)
        nn.init.trunc_normal_(self.placeholder_sar, std=0.02)
        nn.init.trunc_normal_(self.placeholder_dem_feat, std=0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        pretrained_modules = set()
        for m in self.opt_enc.backbone.modules():
            pretrained_modules.add(id(m))
        for m in self.modules():
            if id(m) in pretrained_modules: continue
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def _get_drop_p(self, epoch, total_epochs):
        if not self.dynamic_dropout: return self.drop_timestep_p
        return 0.05 + 0.15 * math.sin(math.pi * epoch / max(total_epochs, 1))
    
    def _to_pixel_seq(self, feat, B, T, H2, W2, D):
        return feat.view(B, T, D, H2, W2).permute(0, 3, 4, 1, 2).reshape(B * H2 * W2, T, D)
    
    def _shift_inputs(self, opt_seq, sar_seq, dem, cloud_mask, valid_count, sx, sy):
        if sx == 0 and sy == 0:
            return opt_seq, sar_seq, dem, cloud_mask, valid_count
        pad_l = abs(sx) if sx < 0 else 0; pad_r = abs(sx) if sx > 0 else 0
        pad_t = abs(sy) if sy < 0 else 0; pad_b = abs(sy) if sy > 0 else 0
        pad_hw = (pad_l, pad_r, pad_t, pad_b)
        opt_seq = F.pad(opt_seq, pad_hw if opt_seq.dim() == 4 else pad_hw + (0, 0), mode='replicate')
        sar_seq = F.pad(sar_seq, pad_hw if sar_seq.dim() == 4 else pad_hw + (0, 0), mode='replicate')
        dem = F.pad(dem, pad_hw, mode='replicate')
        Ht, Wt = dem.shape[-2], dem.shape[-1]
        st = slice(pad_t, Ht - pad_b) if pad_t + pad_b > 0 else slice(None)
        sl = slice(pad_l, Wt - pad_r) if pad_l + pad_r > 0 else slice(None)
        opt_seq = opt_seq[:, :, :, st, sl]; sar_seq = sar_seq[:, :, :, st, sl]; dem = dem[:, :, st, sl]
        if cloud_mask is not None:
            cf = cloud_mask.float() if cloud_mask.dtype == torch.bool else cloud_mask
            cf = F.pad(cf, pad_hw if cf.dim() == 4 else pad_hw + (0, 0), mode='replicate')
            cloud_mask = (cf > 0.5) if cloud_mask.dtype == torch.bool else cf
            cloud_mask = cloud_mask[:, :, st, sl]
        if valid_count is not None:
            ve = valid_count.float().unsqueeze(1) if valid_count.dim() == 3 else valid_count.float()
            ve = F.pad(ve, pad_hw, mode='constant')
            ve = ve[:, :, st, sl] if ve.dim() == 4 else ve[:, st, sl]
            valid_count = ve.squeeze(1).long() if valid_count.dim() == 3 else ve.long()
        return opt_seq, sar_seq, dem, cloud_mask, valid_count

    def forward(self, opt_seq, sar_seq, dem, doy, cloud_mask=None, valid_count=None,
                epoch=0, total_epochs=80, modality_mask=None):
        # ... (essentially same _encode logic as V5EDL, plus mid_fusion and dynamic_drop_p)
```

This step is getting very large. Let me break it into sub-steps.

- [ ] **Step 2a: Create `models/fusion_net_v5pro.py` with full model**

Write the complete file with `MultiScaleFusion` and `FusionCropNetV5Pro` classes. The `forward()` reuses the same `_encode`-style logic as V5EDL but with:
- `self.mid_fusion(opt_p2_time_avg, sar_s2_time_avg)` for mid-level cross-modal gating
- `self._get_drop_p(epoch, total_epochs)` for dynamic dropout
- Mid-level fused features fed into Decoder as additional skip

The full file will be ~380 lines (slightly more than V5EDL due to multi-scale fusion).

- [ ] **Step 2b: Update `models/__init__.py`**

Add:
```python
from .fusion_net_v5pro import FusionCropNetV5Pro, MultiScaleFusion

__all__ = [
    ...
    "FusionCropNetV5Pro",
    "MultiScaleFusion",
]
```

- [ ] **Step 3: Verify compilation**

Run: `python -m py_compile models/fusion_net_v5pro.py`
Expected: OK

- [ ] **Step 4: Run smoke test**

```bash
python -c "
import torch
from models.fusion_net_v5pro import FusionCropNetV5Pro
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = FusionCropNetV5Pro(backbone='resnet50', pretrained=False).to(dev)
m.train()
opt = torch.randn(1, 12, 10, 32, 32).to(dev)
sar = torch.randn(1, 12, 5, 32, 32).to(dev)
dem = torch.randn(1, 5, 32, 32).to(dev)
doy = torch.rand(1, 12).to(dev)
cm = (torch.rand(1, 12, 32, 32) < 0.3).to(dev)
vc = torch.randint(0, 12, (1, 32, 32)).to(dev)
alpha, ndvi, cl = m(opt, sar, dem, doy, cm, vc, epoch=10)
print(f'V5Pro TRAIN OK: alpha={tuple(alpha.shape)} cl={cl.item():.4f}')
m.eval()
with torch.no_grad(): a = m(opt, sar, dem, doy)
print(f'V5Pro EVAL OK: alpha={tuple(a.shape)}')
print('V5Pro smoke test PASSED')
"
```

- [ ] **Step 5: Test with ConvNeXt backbone**

```bash
python -c "
import torch
from models.fusion_net_v5pro import FusionCropNetV5Pro
m = FusionCropNetV5Pro(backbone='convnext_tiny', pretrained=False)
opt = torch.randn(1, 12, 10, 64, 64)
sar = torch.randn(1, 12, 5, 64, 64)
dem = torch.randn(1, 5, 64, 64)
doy = torch.rand(1, 12)
m.eval()
with torch.no_grad(): a = m(opt, sar, dem, doy)
print(f'ConvNeXt backbone OK: alpha={tuple(a.shape)}')
"
```

- [ ] **Step 6: Commit**

```bash
git add models/fusion_net_v5pro.py models/__init__.py
git commit -m "feat: add FusionCropNetV5Pro with multi-scale fusion, pluggable backbone, CARAFE, dynamic dropout"
```

---

### Task 4: Adaptive KL Annealing in EDLLoss

**Files:**
- Modify: `models/fusion_net_v5_edl.py` — `EDLLoss` class

- [ ] **Step 1: Add adaptive KL annealing to EDLLoss**

```python
class EDLLoss(nn.Module):
    def __init__(self, num_classes: int, lambda_max: float = 0.5,
                 kl_anneal_epochs: int = 50, ignore_index: int = 255,
                 adaptive: bool = False):
        super().__init__()
        self.K = num_classes
        self.lambda_max = lambda_max
        self.kl_anneal_epochs = kl_anneal_epochs
        self.ignore_index = ignore_index
        self.adaptive = adaptive
        self._current_lambda = 0.0

    def get_lambda(self, epoch, spear_r=None):
        if not self.adaptive:
            return self.lambda_max * min(1.0, epoch / max(self.kl_anneal_epochs, 1))
        base = self.lambda_max * min(1.0, epoch / max(self.kl_anneal_epochs, 1))
        if spear_r is not None and spear_r > 0.3:
            base = min(self.lambda_max, base * 1.1)  # Accelerate when uncertainty correlates
        return base

    def forward(self, alpha, targets, epoch, spear_r=None):
        B, K, H, W = alpha.shape
        lam = self.get_lambda(epoch, spear_r)
        self._current_lambda = lam
        # ... rest unchanged
```

- [ ] **Step 2: Verify EDLLoss with adaptive mode**

```bash
python -c "
from models.fusion_net_v5_edl import EDLLoss
import torch
loss_fn = EDLLoss(7, adaptive=True)
alpha = torch.rand(2, 7, 32, 32) + 1.0
y = torch.randint(0, 7, (2, 32, 32))
l1 = loss_fn(alpha, y, epoch=10, spear_r=0.2)
l2 = loss_fn(alpha, y, epoch=10, spear_r=0.5)
print(f'lambda(spear_r=0.2)={loss_fn._current_lambda:.4f}')
print(f'lambda(spear_r=0.5)={loss_fn._current_lambda:.4f}')
print('Adaptive KL OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add models/fusion_net_v5_edl.py
git commit -m "feat: add adaptive KL annealing to EDLLoss based on Spearman correlation"
```

---

### Task 5: Update all scripts for V5Pro compatibility

**Files:**
- Modify: `scripts/train_fusion.py`
- Modify: `scripts/train_fusion_edl.py`
- Modify: `scripts/train_mil.py`
- Modify: `models/__init__.py`
- Modify: `utils/trainer.py`

- [ ] **Step 1: Update `scripts/train_fusion.py` to support --v5pro flag**

Add `--v5pro` argument and model selection logic:

```python
parser.add_argument("--v5pro", action="store_true", help="Use V5Pro model")
# ...
if args.v5pro:
    from models.fusion_net_v5pro import FusionCropNetV5Pro
    model = FusionCropNetV5Pro(
        backbone=args.backbone, use_carafe=True,
        dynamic_dropout=True, adaptive_kl=True)
else:
    model = FusionCropNetV5EDL(...)
```

- [ ] **Step 2: Update `scripts/train_fusion_edl.py` similarly**

Add `--v5pro` flag and `--backbone` option.

- [ ] **Step 3: Update `utils/trainer.py` for V5Pro forward signature**

V5Pro's forward takes `(opt, sar, dem, doy, cloud_mask, valid_count, epoch, total_epochs)`. The trainer needs to pass these:

```python
# In FusionTrainer.train_epoch:
if isinstance(self.model, FusionCropNetV5Pro):
    logits = self.model(opt, sar, dem, doy, cm, vc, epoch=current_epoch, total_epochs=total_epochs)
```

- [ ] **Step 4: Verify all scripts compile**

```bash
for f in scripts/train_fusion.py scripts/train_fusion_edl.py utils/trainer.py; do
    python -m py_compile "$f" && echo "OK: $f" || echo "FAIL: $f"
done
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train_fusion.py scripts/train_fusion_edl.py utils/trainer.py
git commit -m "feat: add V5Pro support to training scripts"
```

---

### Task 6: Write changelog and sync docs

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `OPTIMIZATION_PROPOSAL.md`
- Sync: `E:\project_report\`

- [ ] **Step 1: Add v5.2 entry to CHANGELOG.md**

- [ ] **Step 2: Update OPTIMIZATION_PROPOSAL.md Phase 2 status**

- [ ] **Step 3: Sync all docs to E:\project_report**

```bash
cp CHANGELOG.md "E:\project_report\CHANGELOG.md"
cp OPTIMIZATION_PROPOSAL.md "E:\project_report\OPTIMIZATION_PROPOSAL.md"
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md OPTIMIZATION_PROPOSAL.md
git commit -m "docs: add v5.2 changelog for V5Pro implementation"
```
