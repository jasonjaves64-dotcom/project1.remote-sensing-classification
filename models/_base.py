"""
Shared base components for FusionCropNet model family.
Single canonical source — no duplication across model files.

All components use the bug-fixed versions from V5EDL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from einops import rearrange

# ═══════════════════════════════════════════════════════════════
# Basic blocks
# ═══════════════════════════════════════════════════════════════

class ConvBNGELU(nn.Module):
    def __init__(self, i, o, k=3, s=1, p=1, g=1):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(o), nn.GELU())

    def forward(self, x):
        return self.b(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 4)), nn.ReLU(),
            nn.Linear(max(ch // r, 4), ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation with clipping."""
    def __init__(self, cond_ch: int, feat_ch: int, gamma_clip: float = 2.0, beta_clip: float = 1.0):
        super().__init__()
        self.gamma = nn.Conv2d(cond_ch, feat_ch, 1, bias=False)
        self.beta = nn.Conv2d(cond_ch, feat_ch, 1, bias=True)
        self.gamma_clip = gamma_clip
        self.beta_clip = beta_clip
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[-2:] != feat.shape[-2:]:
            cond = F.interpolate(cond, feat.shape[-2:], mode='bilinear', align_corners=False)
        g = torch.clamp(self.gamma(cond), -self.gamma_clip, self.gamma_clip)
        b = torch.clamp(self.beta(cond), -self.beta_clip, self.beta_clip)
        return feat * (1 + torch.tanh(g)) + b


class IRB(nn.Module):
    """Inverted Residual Block with SE attention."""
    def __init__(self, i, o, e=4):
        super().__init__()
        m = i * e
        self.conv = nn.Sequential(
            nn.Conv2d(i, m, 1, bias=False), nn.BatchNorm2d(m), nn.GELU(),
            nn.Conv2d(m, m, 3, 1, 1, groups=m, bias=False), nn.BatchNorm2d(m), nn.GELU(),
            SEBlock(m),
            nn.Conv2d(m, o, 1, bias=False), nn.BatchNorm2d(o))
        self.skip = nn.Conv2d(i, o, 1, bias=False) if i != o else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv(x) + self.skip(x))


# ═══════════════════════════════════════════════════════════════
# DEM Encoder
# ═══════════════════════════════════════════════════════════════

class DEMEncoder(nn.Module):
    def __init__(self, in_ch: int = 5, out_ch: int = 128):
        super().__init__()
        self.out_ch = out_ch
        self.local = nn.Sequential(
            ConvBNGELU(in_ch, 32), ConvBNGELU(32, 64), SEBlock(64),
            nn.Conv2d(64, 128, 5, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            ConvBNGELU(128, out_ch), SEBlock(out_ch))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(in_ch, 64), nn.GELU(),
            nn.Linear(64, out_ch), nn.Sigmoid())
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, dem: torch.Tensor) -> torch.Tensor:
        g = self.global_mlp(self.global_pool(dem).flatten(1)).view(-1, self.out_ch, 1, 1)
        return self.local(dem) * g + self.skip(dem)


# ═══════════════════════════════════════════════════════════════
# Optical Encoder
# ═══════════════════════════════════════════════════════════════

_BACKBONE_CHANNELS = {
    "resnet18":        [64, 128, 256, 512],
    "resnet34":        [64, 128, 256, 512],
    "resnet50":        [256, 512, 1024, 2048],
    "resnet101":       [256, 512, 1024, 2048],
    "convnext_tiny":   [96, 192, 384, 768],
    "convnext_small":  [96, 192, 384, 768],
    "swin_tiny_patch4_window7_224":  [96, 192, 384, 768],
    "efficientnet_b0": [24, 40, 112, 320],
    "efficientnet_b4": [24, 48, 120, 336],
    "maxvit_tiny":     [64, 128, 256, 512],
}


# Registry of available remote sensing ViT foundation models
_VIT_FOUNDATION_MODELS = {
    "terrafm_b": {"dim": 768, "patch_size": 16, "url": "huggingface.co/MBZUAI/TerraFM"},
    "terrafm_l": {"dim": 1024, "patch_size": 16, "url": "huggingface.co/MBZUAI/TerraFM"},
    "dofa_vitb": {"dim": 768, "patch_size": 16, "url": "huggingface.co/earthflow/DOFA"},
    "dofa_vitl": {"dim": 1024, "patch_size": 16, "url": "huggingface.co/earthflow/DOFA"},
}


def list_vit_foundation_models():
    """Return available ViT foundation model names and their specs."""
    return dict(_VIT_FOUNDATION_MODELS)


def load_vit_foundation_weights(model_name: str, checkpoint_path: str):
    """Load ViT foundation model weights.

    Args:
        model_name: key in _VIT_FOUNDATION_MODELS
        checkpoint_path: path to downloaded .pth file

    Returns:
        state_dict or None if model not recognized
    """
    if model_name not in _VIT_FOUNDATION_MODELS:
        raise ValueError(f"Unknown ViT model '{model_name}'. "
                         f"Available: {list(_VIT_FOUNDATION_MODELS.keys())}")

    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    return state


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_ch):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels_list])
        self.outs = nn.ModuleList([ConvBNGELU(out_ch, out_ch) for _ in in_channels_list])

    def forward(self, feats):
        target = feats[0].shape[-2:]
        merged = None
        fps = []
        for i, f in enumerate(feats):
            lat = self.lats[i](f)
            if merged is not None:
                lat = lat + F.interpolate(merged, lat.shape[-2:], mode='bilinear', align_corners=False)
            merged = self.outs[i](lat)
            fps.append(merged)
        main = sum(F.interpolate(f, target, mode='bilinear', align_corners=False) for f in fps)
        p2 = F.interpolate(fps[0], scale_factor=2, mode='bilinear', align_corners=False)
        p3 = fps[0]
        return main, p2, p3


class OpticalEncoder(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int, backbone: str = "resnet50",
                 pretrained: bool = True, rs_weights_path: str = None,
                 use_domain_adapter: bool = False):
        super().__init__()
        bb_ch = _BACKBONE_CHANNELS.get(backbone, [256, 512, 1024, 2048])
        # ConvNeXt/Swin use (0,1,2,3); ResNet/EfficientNet/MaxViT use (1,2,3,4)
        if any(k in backbone for k in ['convnext', 'swin']):
            out_idx = (0, 1, 2, 3)
        else:
            out_idx = (1, 2, 3, 4)

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=out_idx)

        # V6 Block 6: Remote sensing pre-trained weights (SeCo, GASSL, etc.)
        if rs_weights_path is not None:
            self._load_rs_weights(rs_weights_path)

        # Adapt first layer for multispectral input (10ch)
        self._adapt_first_conv(in_ch)

        self.fpn = FPN(bb_ch, feat_dim)
        self.domain_adapter = DomainAdapter(feat_dim) if use_domain_adapter else None
        self.sp2 = nn.Conv2d(feat_dim, feat_dim // 2, 1)
        self.sp3 = nn.Conv2d(feat_dim, feat_dim // 2, 1)

    def _adapt_first_conv(self, in_ch: int):
        """Patch the first Conv2d with 3 input channels to accept `in_ch` channels.
        Walks the full module tree via named_modules() so it works for any backbone
        (ResNet conv1, ConvNeXt stem[0], Swin patch_embed.proj, EfficientNet conv_stem, etc.)
        regardless of how timm's FeatureListNet wraps them."""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                parts = name.split('.')
                parent = self.backbone
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                nc = nn.Conv2d(in_ch, module.out_channels, module.kernel_size,
                               module.stride, module.padding,
                               groups=module.groups, bias=module.bias is not None)
                nw = module.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1)
                nc.weight.data = nw
                if module.bias is not None:
                    nc.bias.data = module.bias.data
                if isinstance(parent, nn.Sequential):
                    parent[int(parts[-1])] = nc
                elif isinstance(parent, nn.ModuleList):
                    parent[int(parts[-1])] = nc
                else:
                    setattr(parent, parts[-1], nc)
                return  # only the first in_channels=3 Conv2d is adapted

    def _load_rs_weights(self, path: str):
        """Load remote sensing pre-trained weights with key mapping.

        Handles:
          - Standard torch.save state_dict
          - PyTorch Lightning checkpoints (state_dict key)
          - SeCo MoCo-v2 format (encoder_q.N.* keys)
          - conv1 3ch→10ch shape mismatch (copy first 3 input channels)
        """
        state = torch.load(path, map_location='cpu', weights_only=False)

        # Handle common save formats
        if isinstance(state, dict):
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'online_backbone' in state:
                state = state['online_backbone']
            elif 'target_backbone' in state:
                state = state['target_backbone']
            elif 'model_state' in state:
                state = state['model_state']

        # Detect SeCo MoCo format: encoder_q.* keys
        is_seco = any(k.startswith('encoder_q.') for k in state.keys())
        if is_seco:
            state = self._map_seco_keys(state)
            if state is None:
                return

        timm_state = self.backbone.state_dict()

        def normalize_key(k):
            for prefix in ['module.', 'encoder.', 'backbone.', 'online_backbone.', 'target_backbone.']:
                if k.startswith(prefix):
                    k = k[len(prefix):]
            return k

        seco_keys = {normalize_key(k): k for k in state.keys()}
        timm_keys = list(timm_state.keys())

        loaded = 0
        skipped = 0
        new_state = {}

        for tk in timm_keys:
            nk = normalize_key(tk)
            if nk in seco_keys:
                w = state[seco_keys[nk]]
                if w.shape == timm_state[tk].shape:
                    new_state[tk] = w
                    loaded += 1
                elif tk == 'conv1.weight' and w.dim() == 4 and w.shape[1] == 3:
                    # SeCo conv1 is (64, 3, 7, 7) but timm adapted to (64, 10, 7, 7)
                    # Copy the 3 pre-trained channels, leave rest as-is
                    tw = timm_state[tk].clone()
                    tw[:, :3] = w
                    new_state[tk] = tw
                    loaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        if loaded > 0:
            self.backbone.load_state_dict(new_state, strict=False)

        total = loaded + skipped
        pct = 100 * loaded / max(total, 1)
        print(f"[RS] Loaded {loaded}/{total} keys ({pct:.0f}%) from {path}")
        if pct < 50:
            print(f"[RS] WARNING: Low match rate — check key format compatibility")

    def _map_seco_keys(self, state):
        """Map SeCo MoCo-v2 encoder_q.N.* keys to standard torchvision ResNet keys.

        SeCo:  encoder_q.0.* = conv1, encoder_q.1.* = bn1,
               encoder_q.4.* = layer1, encoder_q.5.* = layer2,
               encoder_q.6.* = layer3, encoder_q.7.* = layer4
        """
        mapped = {}
        for k, v in state.items():
            if not k.startswith('encoder_q.'):
                continue
            parts = k.split('.', 2)  # ['encoder_q', 'N', 'rest']
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            rest = parts[2]

            if idx == 0:
                new_k = 'conv1.' + rest if rest else 'conv1.weight'
            elif idx == 1:
                new_k = 'bn1.' + rest
            elif 4 <= idx <= 7:
                layer_num = idx - 3  # 4→1, 5→2, 6→3, 7→4
                new_k = f'layer{layer_num}.{rest}'
            else:
                continue

            mapped[new_k] = v

        if len(mapped) == 0:
            print(f"[RS] Could not map SeCo keys — found 0 encoder_q.* keys")
            return None
        return mapped

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        # Some transformer backbones (Swin/ViT) emit NHWC (B,H,W,C);
        # detect by checking if the expected channel count sits in the last dim.
        feats_nchw = []
        for i, f in enumerate(feats):
            if (f.ndim == 4 and f.shape[-1] == self.fpn.lats[i].in_channels
                    and f.shape[1] != self.fpn.lats[i].in_channels):
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        main, p2, p3 = self.fpn(feats_nchw)
        if self.domain_adapter is not None:
            main = self.domain_adapter(main)
        return main, self.sp2(p2), self.sp3(p3)


# ═══════════════════════════════════════════════════════════════
# SAR Encoder
# ═══════════════════════════════════════════════════════════════

class SAREncoder(nn.Module):
    def __init__(self, in_ch: int = 5, base_ch: int = 32, out_ch: int = 512, dem_ch: int = 128):
        super().__init__()
        self.stem = ConvBNGELU(in_ch, base_ch)
        self.stage1 = nn.Sequential(IRB(base_ch, base_ch * 2), IRB(base_ch * 2, base_ch * 2))
        self.film1 = FiLM(dem_ch, base_ch * 2)
        self.down1 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(IRB(base_ch * 2, base_ch * 4), IRB(base_ch * 4, base_ch * 4))
        self.film2 = FiLM(dem_ch, base_ch * 4)
        self.down2 = nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(IRB(base_ch * 4, out_ch), IRB(out_ch, out_ch))
        self.out_channels_list = [base_ch * 2, base_ch * 4, out_ch]

    def forward(self, x: torch.Tensor, dem_feat: torch.Tensor = None):
        x = self.stem(x)
        s1 = self.stage1(x)
        if dem_feat is not None:
            s1 = self.film1(s1, dem_feat)
        s2 = self.stage2(self.down1(s1))
        if dem_feat is not None:
            d2 = F.interpolate(dem_feat, s2.shape[-2:], mode='bilinear', align_corners=False)
            s2 = self.film2(s2, d2)
        s3 = self.stage3(self.down2(s2))
        return s1, s2, s3


# ═══════════════════════════════════════════════════════════════
# Temporal Encoder
# ═══════════════════════════════════════════════════════════════

class FourierDOYEncoding(nn.Module):
    def __init__(self, d_model: int, n_freqs: int = 4):
        super().__init__()
        freqs = torch.arange(1, n_freqs + 1).float()
        self.register_buffer('freqs', freqs)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * doy.unsqueeze(-1) * self.freqs
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return x + self.proj(fourier)


class ObsQualityToken(nn.Module):
    def __init__(self, d_model: int, max_obs: int = 24):
        super().__init__()
        self.embed = nn.Embedding(max_obs + 1, d_model)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, x: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
        tok = self.embed(vc.clamp(0, self.embed.num_embeddings - 1)).unsqueeze(1)
        return torch.cat([tok, x], dim=1)


class TemporalEncoderStream(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4,
                 dropout: float = 0.1, max_obs: int = 24, n_freqs: int = 4):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = FourierDOYEncoding(d_model, n_freqs)
        self.obs_tok = ObsQualityToken(d_model, max_obs)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fallback = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.fallback, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, doy, cloud_mask=None, valid_count=None, fallback_feat=None):
        N, T, D = x.shape
        x = self.pos_enc(x, doy)
        obs_added = False
        if valid_count is not None:
            x = self.obs_tok(x, valid_count)
            obs_added = True
        if cloud_mask is not None:
            fully = cloud_mask.all(dim=1)
            if fully.any():
                fb = (fallback_feat[fully] if fallback_feat is not None
                      else self.fallback.expand(fully.sum(), D))
                x[fully] = fb.unsqueeze(1).expand(-1, x.shape[1], -1)
        cls = self.cls.expand(N, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if cloud_mask is not None:
            cls_m = torch.zeros(N, 1, dtype=torch.bool, device=x.device)
            if obs_added:
                obs_m = torch.zeros(N, 1, dtype=torch.bool, device=x.device)
                pad_mask = torch.cat([cls_m, obs_m, cloud_mask], dim=1)
            else:
                pad_mask = torch.cat([cls_m, cloud_mask], dim=1)
            if fully.any():
                pad_mask[fully] = False
        else:
            pad_mask = None
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        cls_out = self.norm(x[:, 0])
        seq_out = self.norm(x[:, 2:] if obs_added else x[:, 1:])
        return cls_out, seq_out


# ═══════════════════════════════════════════════════════════════
# Multi-scale Cross-Modal Attention
# ═══════════════════════════════════════════════════════════════

class CrossModalLite(nn.Module):
    """Lightweight cross-modal attention for high-resolution features (H, H/2).

    Uses single-head attention + depthwise projections to minimize compute.
    For H/2 (128x128 at 256px input): 16K tokens — full multi-head would be expensive.
    """
    def __init__(self, d_model: int, n_heads: int = 1):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        C = d_model

        # Light projections: use depthwise + pointwise instead of full conv
        self.qo = nn.Sequential(
            nn.Conv2d(C, C, 1, groups=C // 8 if C >= 8 else 1, bias=False),
            nn.Conv2d(C, C, 1, bias=False)
        )
        self.ks = nn.Sequential(
            nn.Conv2d(C, C, 1, groups=C // 8 if C >= 8 else 1, bias=False),
            nn.Conv2d(C, C, 1, bias=False)
        )
        self.vs = nn.Sequential(
            nn.Conv2d(C, C, 1, groups=C // 8 if C >= 8 else 1, bias=False),
            nn.Conv2d(C, C, 1, bias=False)
        )

        self.gate = nn.Sequential(nn.Conv2d(C * 2, C, 1), nn.Sigmoid())
        self.proj = ConvBNGELU(C * 2, C)
        self.norm = nn.GroupNorm(min(32, C // 2) if C >= 4 else 1, C)

    def _xattn(self, qf, kvf, qp, kp, vp):
        B, C, H, W = qf.shape
        h, d = self.nh, C // self.nh
        Q = qp(qf).view(B, h, d, -1).permute(0, 1, 3, 2)
        K = kp(kvf).view(B, h, d, -1).permute(0, 1, 3, 2)
        V = vp(kvf).view(B, h, d, -1).permute(0, 1, 3, 2)
        a = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        return (a @ V).permute(0, 1, 3, 2).reshape(B, C, H, W)

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if sar.shape[-2:] != opt.shape[-2:]:
            sar = F.interpolate(sar, opt.shape[-2:], mode='bilinear', align_corners=False)

        o2s = self._xattn(opt, sar, self.qo, self.ks, self.vs)
        s2o = self._xattn(sar, opt, self.qo, self.ks, self.vs)

        g = self.gate(torch.cat([o2s, s2o], dim=1))
        out = self.proj(torch.cat([g * o2s + (1 - g) * s2o, opt], dim=1))
        return self.norm(out) + opt


class CrossModalAttention(nn.Module):
    def __init__(self, d_model, n_heads=16, win_size=None):
        super().__init__()
        self.nh = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        C = d_model
        self.qo = nn.Conv2d(C, C, 1, bias=False)
        self.ks = nn.Conv2d(C, C, 1, bias=False)
        self.vs = nn.Conv2d(C, C, 1, bias=False)
        self.qs = nn.Conv2d(C, C, 1, bias=False)
        self.ko = nn.Conv2d(C, C, 1, bias=False)
        self.vo = nn.Conv2d(C, C, 1, bias=False)
        self.sw_o2s = nn.Sequential(nn.Conv2d(C, C, 1), nn.GELU(), nn.Conv2d(C, C, 1))
        self.sw_s2o = nn.Sequential(nn.Conv2d(C, C, 1), nn.GELU(), nn.Conv2d(C, C, 1))
        self.gate = nn.Sequential(nn.Conv2d(C * 2, C, 1), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Conv2d(C * 2, C, 1), nn.GELU(), nn.Conv2d(C, C, 1))
        self.norm = nn.GroupNorm(32, C)

    def _xattn(self, qf, kvf, qp, kp, vp):
        B, C, H, W = qf.shape
        h, d = self.nh, C // self.nh
        Q = qp(qf).view(B, h, d, -1).permute(0, 1, 3, 2)
        K = kp(kvf).view(B, h, d, -1).permute(0, 1, 3, 2)
        V = vp(kvf).view(B, h, d, -1).permute(0, 1, 3, 2)
        a = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        return (a @ V).permute(0, 1, 3, 2).reshape(B, C, H, W)

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if sar.shape[-2:] != opt.shape[-2:]:
            sar = F.interpolate(sar, opt.shape[-2:], mode='bilinear', align_corners=False)
        o2s = self.sw_o2s(self._xattn(opt, sar, self.qo, self.ks, self.vs))
        s2o = self.sw_s2o(self._xattn(sar, opt, self.qs, self.ko, self.vo))
        g = self.gate(torch.cat([o2s, s2o], dim=1))
        out = self.proj(torch.cat([g * o2s + (1 - g) * s2o, opt], dim=1))
        return self.norm(out) + opt


# ═══════════════════════════════════════════════════════════════
# DEM Spatial Conditioner
# ═══════════════════════════════════════════════════════════════

class DEMSpatialConditioner(nn.Module):
    def __init__(self, feat_ch: int, dem_ch: int):
        super().__init__()
        self.film = FiLM(dem_ch, feat_ch)
        self.gate = nn.Sequential(nn.Conv2d(feat_ch + dem_ch, feat_ch, 1), nn.Sigmoid())
        self.mix = nn.Sequential(ConvBNGELU(feat_ch + dem_ch, feat_ch), SEBlock(feat_ch))
        self.norm = nn.GroupNorm(32, feat_ch)

    def forward(self, fused: torch.Tensor, dem: torch.Tensor) -> torch.Tensor:
        if dem.shape[-2:] != fused.shape[-2:]:
            dem = F.interpolate(dem, fused.shape[-2:], mode='bilinear', align_corners=False)
        filmed = self.film(fused, dem)
        cat = torch.cat([filmed, dem], dim=1)
        gate = self.gate(cat)
        return self.norm(gate * self.mix(cat) + (1 - gate) * fused)


# ═══════════════════════════════════════════════════════════════
# DEM Optical Conditioner — brings terrain awareness to optical stream
# ═══════════════════════════════════════════════════════════════

class DEMOpticalConditioner(nn.Module):
    """FiLM-modulate optical features with DEM at multiple scales."""

    def __init__(self, opt_ch: int, dem_ch: int = 128):
        super().__init__()
        self.film_high = FiLM(dem_ch, opt_ch)       # H/4 x W/4
        self.film_mid = FiLM(dem_ch, opt_ch // 2)   # H/2 x W/2
        self.gate = nn.Sequential(
            nn.Conv2d(opt_ch + dem_ch, opt_ch, 1), nn.Sigmoid())
        self.norm = nn.GroupNorm(min(32, opt_ch), opt_ch)

    def forward(self, opt_main, opt_p2, dem_feat):
        """Condition optical features on DEM at high and mid scales."""
        if dem_feat.shape[-2:] != opt_main.shape[-2:]:
            dem_high = F.interpolate(dem_feat, opt_main.shape[-2:],
                                     mode='bilinear', align_corners=False)
        else:
            dem_high = dem_feat
        if dem_feat.shape[-2:] != opt_p2.shape[-2:]:
            dem_mid = F.interpolate(dem_feat, opt_p2.shape[-2:],
                                    mode='bilinear', align_corners=False)
        else:
            dem_mid = dem_feat

        opt_main_c = self.film_high(opt_main, dem_high)
        opt_p2_c = self.film_mid(opt_p2, dem_mid)
        # Gate: learn how much DEM to mix in at high level
        cat = torch.cat([opt_main_c, dem_high], dim=1)
        g = self.gate(cat)
        opt_main_out = self.norm(g * opt_main_c + (1 - g) * opt_main)
        return opt_main_out, opt_p2_c


# ═══════════════════════════════════════════════════════════════
# ModalNormalize — per-modality LayerNorm for early fusion
# ═══════════════════════════════════════════════════════════════

class ModalNormalize(nn.Module):
    """Per-modality LayerNorm — solves numerical range mismatch.

    opt: [0,1] reflectances   sar: [-25,5] dB   dem: [0,8848] meters
    Without normalization, DEM dominates gradients.
    """
    def forward(self, opt: torch.Tensor, sar: torch.Tensor, dem: torch.Tensor) -> torch.Tensor:
        # LayerNorm over (C,H,W) for each modality independently
        opt_n = F.layer_norm(opt, opt.shape[1:])
        sar_n = F.layer_norm(sar, sar.shape[1:])
        dem_n = F.layer_norm(dem, dem.shape[1:])
        return torch.cat([opt_n, sar_n, dem_n], dim=1)


# ═══════════════════════════════════════════════════════════════
# Lightweight Cross-Modal Attention — for mid-level fusion (H/2 x W/2)
# ═══════════════════════════════════════════════════════════════

class CrossModalAttentionLight(nn.Module):
    """Lightweight cross-modal attention for mid-resolution features.
    Uses single-head depthwise attention instead of full multi-head attention."""

    def __init__(self, ch_opt: int, ch_sar: int, out_ch: int = None):
        super().__init__()
        if out_ch is None:
            out_ch = ch_sar
        self.q_opt = nn.Conv2d(ch_opt, out_ch, 1, bias=False)
        self.k_sar = nn.Conv2d(ch_sar, out_ch, 1, bias=False)
        self.v_sar = nn.Conv2d(ch_sar, out_ch, 1, bias=False)
        self.q_sar = nn.Conv2d(ch_sar, out_ch, 1, bias=False)
        self.k_opt = nn.Conv2d(ch_opt, out_ch, 1, bias=False)
        self.v_opt = nn.Conv2d(ch_opt, out_ch, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1), nn.Sigmoid())
        self.proj = ConvBNGELU(out_ch + ch_opt, ch_opt)
        self.norm = nn.GroupNorm(min(32, ch_opt), ch_opt)
        self.scale = out_ch ** -0.5

    def forward(self, opt: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if sar.shape[-2:] != opt.shape[-2:]:
            sar = F.interpolate(sar, opt.shape[-2:], mode='bilinear', align_corners=False)
        B, C_opt, H, W = opt.shape
        C_out = self.gate[0].out_channels  # out_ch

        # Optical queries SAR
        Qo = self.q_opt(opt).view(B, -1, H * W)
        Ks = self.k_sar(sar).view(B, -1, H * W)
        Vs = self.v_sar(sar).view(B, -1, H * W)
        attn_o2s = F.softmax(Qo.transpose(1, 2) @ Ks * self.scale, dim=-1)
        o2s = (attn_o2s @ Vs.transpose(1, 2)).transpose(1, 2).view(B, -1, H, W)

        # SAR queries optical
        Qs = self.q_sar(sar).view(B, -1, H * W)
        Ko = self.k_opt(opt).view(B, -1, H * W)
        Vo = self.v_opt(opt).view(B, -1, H * W)
        attn_s2o = F.softmax(Qs.transpose(1, 2) @ Ko * self.scale, dim=-1)
        s2o = (attn_s2o @ Vo.transpose(1, 2)).transpose(1, 2).view(B, -1, H, W)

        g = self.gate(torch.cat([o2s, s2o], dim=1))
        fused = g * o2s + (1 - g) * s2o
        out = self.proj(torch.cat([fused, opt], dim=1))
        return self.norm(out) + opt


# ═══════════════════════════════════════════════════════════════
# Late Fusion (3-input version: fused, opt, sar)
# ═══════════════════════════════════════════════════════════════

class LateFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))

    def forward(self, fused, opt, sar):
        g = self.gate(torch.cat([fused, opt, sar], dim=-1))
        blend = g * fused + (1 - g) * sar
        return self.proj(torch.cat([blend, opt], dim=-1))


# ═══════════════════════════════════════════════════════════════
# Spatial Refinement (BUG3 fixed: windowed attention, no O(n²))
# ═══════════════════════════════════════════════════════════════

class SpatialRefinement(nn.Module):
    def __init__(self, ch: int, n_heads: int = 8, win: int = 4):
        super().__init__()
        self.win = win
        self.norm1 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 1), nn.GELU(), nn.Conv2d(ch * 4, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.win
        ph = (w - H % w) % w
        pw = (w - W % w) % w
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // w, Wp // w
        xw = rearrange(x, 'b c (nh wh)(nw ww) -> (b nh nw)(wh ww) c', wh=w, ww=w)
        normed = self.norm1(xw)
        attn_out, _ = self.attn(normed, normed, normed)
        xw = xw + attn_out
        out = rearrange(xw, '(b nh nw)(wh ww) c -> b c (nh wh)(nw ww)',
                        b=B, nh=nH, nw=nW, wh=w, ww=w)
        if ph > 0 or pw > 0:
            out = out[:, :, :H, :W]
        out = out + self.mlp(self.norm2(out))
        return out


# ═══════════════════════════════════════════════════════════════
# CARAFE Up-sampler
# ═══════════════════════════════════════════════════════════════

class CARAFEUp(nn.Module):
    """Content-Aware ReAssembly of FEatures — lightweight upsampling without checkerboard artifacts."""
    def __init__(self, in_ch, scale=2, compressed_ch=64):
        super().__init__()
        self.scale = scale
        self.compressed_ch = compressed_ch
        self.compress = nn.Conv2d(in_ch, compressed_ch, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(compressed_ch, compressed_ch, 3, padding=1), nn.GELU())
        self.kernel_pred = nn.Conv2d(compressed_ch, scale ** 2, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.scale
        compressed = self.compress(x)
        kernels = self.kernel_pred(self.encoder(compressed))  # (B, s^2, H, W)
        kernels = F.softmax(kernels, dim=1)
        kernels = kernels.view(B, 1, s * s, H, W)
        # Reshape x for pixel shuffle reconstruction
        x_exp = x.unsqueeze(2).expand(-1, -1, s * s, -1, -1)
        out = (x_exp * kernels).view(B, C, s * s, H, W)
        out = out.permute(0, 1, 3, 4, 2).reshape(B, C, H, s, W, s)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H * s, W * s)
        return out


# ═══════════════════════════════════════════════════════════════
# ViT Feature Pyramid
# ═══════════════════════════════════════════════════════════════

class ViTFeaturePyramid(nn.Module):
    """Convert ViT single-scale patch tokens to multi-scale feature pyramid.

    ViT outputs (B, N, D) patch tokens at a single resolution.
    This module reshapes to spatial format and generates 3 scales (H/4, H/8, H/16)
    via learned upsampling/downsampling, compatible with Decoder skip connections.

    Reference: ViTDet Simple Feature Pyramid (Li et al. 2022)
    """
    def __init__(self, in_dim: int, out_dims: list = None):
        super().__init__()
        if out_dims is None:
            out_dims = [256, 512, 1024]
        self.out_dims = out_dims

        # Lateral convolutions for each output scale
        self.lat_h4 = nn.Conv2d(in_dim, out_dims[0], 1)
        self.lat_h8 = nn.Conv2d(in_dim, out_dims[1], 1)
        self.lat_h16 = nn.Conv2d(in_dim, out_dims[2], 1)

        # Output convolutions (smoothing)
        self.out_h4 = ConvBNGELU(out_dims[0], out_dims[0])
        self.out_h8 = ConvBNGELU(out_dims[1], out_dims[1])
        self.out_h16 = ConvBNGELU(out_dims[2], out_dims[2])

    def forward(self, x: torch.Tensor, patch_size: int = 16, img_size: int = 256):
        """Convert ViT tokens to feature pyramid.

        Args:
            x: (B, N, D) ViT patch tokens (including CLS token if present)
            patch_size: ViT patch size (e.g., 16 for ViT-B/16)
            img_size: input image spatial size

        Returns:
            f_h4: (B, out_dims[0], H/4, W/4)
            f_h8: (B, out_dims[1], H/8, W/8)
            f_h16: (B, out_dims[2], H/16, W/16)
        """
        B, N, D = x.shape
        # Remove CLS token if N = (H/P)*(W/P) + 1
        if N == (img_size // patch_size) ** 2 + 1:
            x = x[:, 1:]  # Remove CLS
            N = N - 1

        grid = int(N ** 0.5)
        # Reshape to spatial: (B, D, H/P, W/P)
        x = x.permute(0, 2, 1).reshape(B, D, grid, grid)

        # H/4, H/8, H/16 scales
        f_h4 = self.out_h4(self.lat_h4(x))   # H/P scale (near H/4 for P=16, H=256: H/16)
        f_h8 = self.out_h8(self.lat_h8(F.avg_pool2d(x, 2)))  # H/(2P)
        f_h16 = self.out_h16(self.lat_h16(F.avg_pool2d(x, 4)))  # H/(4P)

        return f_h4, f_h8, f_h16


# ═══════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """Decoder that outputs pre-head features (64ch). Caller applies final head."""
    def __init__(self, feat_dim, sar_ch_list, n_heads=8, win=4, use_carafe=True, dem_ch=128):
        super().__init__()
        od = feat_dim // 2
        sc0, sc1 = sar_ch_list
        if use_carafe:
            self.up1 = nn.Sequential(
                CARAFEUp(feat_dim, scale=2),
                nn.Conv2d(feat_dim, 64, 1))
            self.up2 = CARAFEUp(64, scale=2)
        else:
            self.up1 = nn.ConvTranspose2d(feat_dim, 64, 2, stride=2)
            self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.sr = SpatialRefinement(64, n_heads, win)
        self.merge1 = ConvBNGELU(64 + od + sc1, 64)
        self.merge2 = ConvBNGELU(64 + od + sc0, 64)
        self.pre_head_ch = 64
        self.dem_skip_proj = nn.Conv2d(dem_ch, self.pre_head_ch, 1, bias=False)
        self.early_skip_proj = nn.Conv2d(128, self.pre_head_ch, 1, bias=False)

    def _a(self, f, hw):
        return (F.interpolate(f, hw, mode='bilinear', align_corners=False)
                if f.shape[-2:] != hw else f)

    def forward(self, x, opt_skips, sar_skips, target_size, dem_skip=None, early_skip=None):
        opt_p2, = opt_skips
        sar_s1, sar_s2 = sar_skips
        x = self.up1(x)
        x = self.merge1(torch.cat([x,
                                   self._a(opt_p2, x.shape[-2:]),
                                   self._a(sar_s2, x.shape[-2:])], dim=1))
        # V6 Block 3: DEM → Decoder skip (terrain prior)
        if dem_skip is not None:
            if dem_skip.shape[-2:] != x.shape[-2:]:
                dem_skip = F.interpolate(dem_skip, x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + self.dem_skip_proj(dem_skip)
        x = self.sr(x)
        x = self.up2(x)
        x = self.merge2(torch.cat([x,
                                   self._a(opt_p2, x.shape[-2:]),
                                   self._a(sar_s1, x.shape[-2:])], dim=1))
        # V6: Early fusion skip — tri-modal joint representation at full resolution
        if early_skip is not None:
            if early_skip.shape[-2:] != x.shape[-2:]:
                early_skip = F.interpolate(early_skip, x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + self.early_skip_proj(early_skip)
        return self._a(x, target_size)


# ═══════════════════════════════════════════════════════════════
# Phenology Auxiliary Head
# ═══════════════════════════════════════════════════════════════

class PhenologyAuxHead(nn.Module):
    def __init__(self, feat_dim: int, aux_weight: float = 0.3):
        super().__init__()
        self.aux_weight = aux_weight
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feat_dim, feat_dim // 4), nn.GELU(),
            nn.Linear(feat_dim // 4, 1))

    def forward(self, opt_feat_BT: torch.Tensor) -> torch.Tensor:
        return self.head(opt_feat_BT).squeeze(-1)

    @staticmethod
    def compute_loss(pred, target, cloud_mask_BT=None) -> torch.Tensor:
        err = F.smooth_l1_loss(pred, target, reduction='none', beta=0.1)
        if cloud_mask_BT is not None:
            valid = ~cloud_mask_BT
            return err[valid].mean() * 0.3 if valid.any() else err.mean() * 0.0
        return err.mean() * 0.3

    @staticmethod
    def schedule_weight(epoch: int) -> float:
        if epoch < 5:   return 0.1 * epoch / 5
        if epoch <= 40: return 0.1
        return max(0.0, 0.1 * (80 - epoch) / 40)


# ═══════════════════════════════════════════════════════════════
# 3-Expert LateFusion (V6 Block 9 P3)
# ═══════════════════════════════════════════════════════════════

class ThreeExpertLateFusion(nn.Module):
    """3-Expert decision-level fusion with EDL vacuity weighting.

    Three parallel expert heads predict crop types from different views:
      - Expert_opt: optical-only path features
      - Expert_sar: SAR-only path features
      - Expert_fused: fused cross-modal features

    Final prediction = vacuity-weighted ensemble of the three.
    Missing modality → vacuity high → weight low → automatic fallback.
    """
    def __init__(self, in_ch: int, num_classes: int, hidden: int = 256):
        super().__init__()
        # Shared architecture, separate weights per expert
        self.expert_opt = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1)
        )
        self.expert_sar = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1)
        )
        self.expert_fused = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1)
        )

        # Weight MLP: vacuity_opt, vacuity_sar, vacuity_fused → 3 softmax weights
        self.weight_net = nn.Sequential(
            nn.Linear(3, 8), nn.GELU(),
            nn.Linear(8, 3)
        )

    def forward(self, opt_feat: torch.Tensor, sar_feat: torch.Tensor,
                fused_feat: torch.Tensor, num_classes: int = 7, B: int = None):
        """Fuse three expert predictions via vacuity-weighted ensemble.

        Args:
            opt_feat: (B*H2W2, D) optical path features (spatial dims flattened)
            sar_feat: (B*H2W2, D) SAR path features
            fused_feat: (B*H2W2, D) cross-modal fused features
            num_classes: K
            B: batch size (required when features are flattened as B*H2W2).

        Returns:
            final_logits: (B, K, H2, W2) — ensembled logits
            expert_weights: (B, 3, 1, 1) — per-image expert weights
        """
        # Reshape to spatial and apply experts
        N = opt_feat.shape[0]
        C = opt_feat.shape[1]

        if B is not None:
            # Explicit batch: N = B * H2 * W2
            HW = N // B
            hw = int(HW ** 0.5)
            if hw * hw != HW:
                raise ValueError(f"Cannot infer spatial dims: N={N} B={B} HW={HW} (not a perfect square)")
            opt_sp = opt_feat.view(B, C, hw, hw)
            sar_sp = sar_feat.view(B, C, hw, hw)
            fused_sp = fused_feat.view(B, C, hw, hw)
        else:
            # Auto-detect: works only when N is a perfect square (single-sample or square batch)
            hw = int(N ** 0.5)
            if hw * hw == N:
                opt_sp = opt_feat.view(-1, C, hw, hw)
                sar_sp = sar_feat.view(-1, C, hw, hw)
                fused_sp = fused_feat.view(-1, C, hw, hw)
            else:
                opt_sp = opt_feat.unsqueeze(-1).unsqueeze(-1)
                sar_sp = sar_feat.unsqueeze(-1).unsqueeze(-1)
                fused_sp = fused_feat.unsqueeze(-1).unsqueeze(-1)

        logits_opt = self.expert_opt(opt_sp)
        logits_sar = self.expert_sar(sar_sp)
        logits_fused = self.expert_fused(fused_sp)

        # EDL vacuity: K/S for each expert
        alpha_opt = F.softplus(logits_opt) + 1.0
        alpha_sar = F.softplus(logits_sar) + 1.0
        alpha_fused = F.softplus(logits_fused) + 1.0

        K = num_classes
        vacuity_opt = K / alpha_opt.sum(dim=1, keepdim=True)
        vacuity_sar = K / alpha_sar.sum(dim=1, keepdim=True)
        vacuity_fused = K / alpha_fused.sum(dim=1, keepdim=True)

        # Compute per-image weights from average vacuity
        # Each vacuity_opt.mean(dim=(2,3)) is (B, 1) → squeeze to (B,) then stack to (B, 3)
        vac_stack = torch.stack([
            vacuity_opt.mean(dim=(2, 3)).squeeze(1),
            vacuity_sar.mean(dim=(2, 3)).squeeze(1),
            vacuity_fused.mean(dim=(2, 3)).squeeze(1),
        ], dim=1)  # (B, 3)

        # Lower vacuity → higher weight (invert via negative before softmax)
        weights = F.softmax(-vac_stack, dim=-1)  # (B, 3)
        weights = weights.view(-1, 3, 1, 1)  # (B, 3, 1, 1)

        # Weighted ensemble
        final = (weights[:, 0:1] * logits_opt +
                 weights[:, 1:2] * logits_sar +
                 weights[:, 2:3] * logits_fused)

        return final, weights


# ═══════════════════════════════════════════════════════════════
# Lightweight Scene Head (V6 Block 7)
# ═══════════════════════════════════════════════════════════════

class LightSceneHead(nn.Module):
    """Lightweight scene-level understanding from pooled features.

    Predicts scene type and crop distribution from globally pooled features.
    Used as an interim solution until GeoFM Scene Encoder is available (P3).
    """
    def __init__(self, in_ch: int = 64, hidden: int = 256,
                 num_scene_types: int = 4, num_crops: int = 7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.scene_type_head = nn.Linear(hidden, num_scene_types)
        self.crop_mix_head = nn.Linear(hidden, num_crops)

    def forward(self, x: torch.Tensor) -> tuple:
        """Predict scene type logits and crop distribution.

        Args:
            x: (B, in_ch, H, W) — shared pre_head features from decoder
        Returns:
            scene_logits: (B, num_scene_types)
            crop_mix: (B, num_crops) — softmax-normalized crop distribution
        """
        g = self.shared(x)  # (B, hidden)
        scene_logits = self.scene_type_head(g)
        crop_mix = F.softmax(self.crop_mix_head(g), dim=-1)
        return scene_logits, crop_mix


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════

def compute_ndvi_loss(opt, ndvi_pred, cloud_mask, ndvi_channel: int = 6):
    """Compute NDVI auxiliary task loss from optical sequence.

    Args:
        opt: (B, T, C, H, W) optical sequence
        ndvi_pred: (B*T,) predicted NDVI values
        cloud_mask: (B, T, H, W) or None
        ndvi_channel: index of NDVI band in optical channels

    Returns:
        ndvi_loss: scalar, weighted NDVI prediction loss
        aux_weight: PhenologyAuxHead schedule weight for current epoch
    """
    import torch.nn.functional as F
    B, T = opt.shape[:2]
    ndvi_tgt = opt[:, :, ndvi_channel].mean(dim=(-2, -1)).reshape(B * T)
    cm_bt = cloud_mask.view(B * T, -1).any(-1) if cloud_mask is not None else None
    return PhenologyAuxHead.compute_loss(ndvi_pred, ndvi_tgt, cm_bt)


def time_average(feat: torch.Tensor, B: int, T: int) -> torch.Tensor:
    _, C, h, w = feat.shape
    return feat.view(B, T, C, h, w).mean(dim=1)


# ═══════════════════════════════════════════════════════════════
# DomainAdapter — AdaIN-style domain adaptation for remote sensing
# ═══════════════════════════════════════════════════════════════

class DomainAdapter(nn.Module):
    """Lightweight AdaIN-style domain adaptation for remote sensing.

    Shifts feature statistics from source domain (ImageNet/SeCo) toward
    target domain (specific crop regions). Inserted between backbone and FPN.

    Uses learnable channel-wise shift + scale, initialized to identity.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned affine transformation per channel.

        Args:
            x: (B, C, H, W) feature map
        Returns:
            (B, C, H, W) adapted features
        """
        return x * self.scale + self.shift
