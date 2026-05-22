"""
Classification / uncertainty heads and spatial refinement.
Basic blocks imported from ._base; unique SWBlock + UncertaintyHead kept here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ._base import ConvBNGELU, SEBlock


class SWBlock(nn.Module):
    """Shifted-window attention block (Swin-style)."""
    def __init__(self, dim, win=4, nh=8, shift=False):
        super().__init__()
        self.win = win
        self.sh = win // 2 if shift else 0
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nh, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 4, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.win
        if self.sh:
            x = torch.roll(x, (-self.sh, -self.sh), (2, 3))
        ph = (w - H % w) % w
        pw = (w - W % w) % w
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, Hp, Wp = x.shape
        nH, nW = Hp // w, Wp // w
        xw = rearrange(x, 'b c (nh wh)(nw ww)->(b nh nw)(wh ww) c', wh=w, ww=w)
        a, _ = self.attn(self.norm(xw), self.norm(xw), self.norm(xw))
        xw = xw + a + self.ffn(xw)
        out = rearrange(xw, '(b nh nw)(wh ww) c->b c (nh wh)(nw ww)',
                        b=B, nh=nH, nw=nW, wh=w, ww=w)
        if ph or pw:
            out = out[:, :, :H, :W]
        if self.sh:
            out = torch.roll(out, (self.sh, self.sh), (2, 3))
        return out


class SpatialRefinement(nn.Module):
    """V4-style spatial refinement: spatial gate + SWBlocks + depthwise conv."""
    def __init__(self, channels, n_heads=8, win_size=4):
        super().__init__()
        self.sg = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1), nn.GELU(),
            nn.Conv2d(channels // 8, 1, 7, padding=3), nn.Sigmoid())
        self.sw1 = SWBlock(channels, win_size, n_heads, shift=False)
        self.sw2 = SWBlock(channels, win_size, n_heads, shift=True)
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.norm = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x):
        x = x * self.sg(x) + x
        x = self.sw1(x)
        x = self.sw2(x)
        return self.norm(x + self.dw(x))


class UncertaintyHead(nn.Module):
    """MC-Dropout uncertainty head for V4."""
    def __init__(self, in_channels: int, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNGELU(in_channels, 64), nn.Dropout2d(dropout_p),
            ConvBNGELU(64, 64), nn.Dropout2d(dropout_p),
            nn.Conv2d(64, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def predict_with_uncertainty(self, x: torch.Tensor, mc_samples: int = 20) -> tuple:
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.head(x)
                preds.append(torch.softmax(logits, dim=1))
        preds = torch.stack(preds, dim=0)
        mean_pred = preds.mean(dim=0)
        eps = 1e-6
        entropy = -(mean_pred * (mean_pred + eps).log()).sum(dim=1)
        self.eval()
        return mean_pred, entropy


class PhenologyAuxHead(nn.Module):
    """V4 phenology head: operates on sequence features (B*H2*W2, T, D)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))

    def forward(self, seq_out: torch.Tensor) -> torch.Tensor:
        return self.head(seq_out).squeeze(-1)

    @staticmethod
    def loss(pred_ndvi: torch.Tensor, true_ndvi: torch.Tensor, cloud_mask=None) -> torch.Tensor:
        err = F.mse_loss(pred_ndvi, true_ndvi, reduction='none')
        if cloud_mask is not None:
            err = err * (~cloud_mask).float()
            return err.sum() / (~cloud_mask).float().sum().clamp(min=1)
        return err.mean()

    @staticmethod
    def aux_weight(epoch: int, warmup: int = 5, decay_end: int = 40) -> float:
        if epoch < warmup:
            return 0.1 * epoch / warmup
        if epoch <= decay_end:
            return 0.1
        remaining = max(0, 80 - epoch)
        return 0.1 * remaining / (80 - decay_end)
