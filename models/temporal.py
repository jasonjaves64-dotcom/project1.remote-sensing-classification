"""
Temporal encoder modules — V4 variant with DEM-conditioned FiLM and checkpointing.
Basic blocks imported from ._base; unique extended TemporalEncoderStream kept here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ._base import FourierDOYEncoding, ObsQualityToken


class FiLMModulation(nn.Module):
    """Sequence-level FiLM modulation (linear projection from condition)."""
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.gamma_proj = nn.Sequential(nn.Linear(cond_dim, feat_dim), nn.Sigmoid())
        self.beta_proj = nn.Sequential(nn.Linear(cond_dim, feat_dim), nn.Tanh())

    def forward(self, feat, cond):
        gamma = self.gamma_proj(cond).unsqueeze(1)
        beta = self.beta_proj(cond).unsqueeze(1)
        return feat * (1 + gamma) + beta


class TemporalEncoderStream(nn.Module):
    """Extended temporal encoder with DEM FiLM conditioning + gradient checkpointing.
    V4 uses this variant; V5/V5EDL use _base.TemporalEncoderStream (simpler, no dem_cond)."""
    def __init__(self, d_model: int, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.1,
                 max_obs: int = 24, n_freqs: int = 4,
                 use_checkpointing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        self.pos_enc = FourierDOYEncoding(d_model, n_freqs)
        self.obs_token = ObsQualityToken(d_model, max_obs)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.fallback = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.fallback, std=0.02)

        self.dem_film = FiLMModulation(d_model, d_model)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout, activation='gelu',
                batch_first=True, norm_first=True)
            for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d_model)

    def _checkpointed_transformer_layer(self, layer, x, pad_mask, dem_cond, is_even_layer):
        def _forward(x):
            out = layer(x, src_key_padding_mask=pad_mask)
            if dem_cond is not None and is_even_layer:
                cls_token = out[:, 0:1, :]
                seq_tokens = out[:, 1:, :]
                seq_tokens = self.dem_film(seq_tokens, dem_cond)
                out = torch.cat([cls_token, seq_tokens], dim=1)
            return out
        return torch.utils.checkpoint.checkpoint(_forward, x)

    def forward(self, x: torch.Tensor, doy: torch.Tensor,
                cloud_mask=None, valid_count=None,
                fallback_feat=None, dem_cond=None):
        N, T, D = x.shape
        x = self.pos_enc(x, doy)
        if valid_count is not None:
            x = self.obs_token(x, valid_count)
            T_ext = T + 1
        else:
            T_ext = T
        if cloud_mask is not None:
            fully_masked = cloud_mask.all(dim=1)
            if fully_masked.any():
                if fallback_feat is not None:
                    fb = fallback_feat[fully_masked].unsqueeze(1).expand(-1, T_ext, -1)
                else:
                    fb = self.fallback.expand(fully_masked.sum(), T_ext, D)
                x[fully_masked] = fb
        cls = self.cls.expand(N, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if cloud_mask is not None:
            extra_false = torch.zeros(
                N, 1 + (1 if valid_count is not None else 0),
                dtype=torch.bool, device=cloud_mask.device)
            pad_mask = torch.cat([extra_false, cloud_mask], dim=1)
            if fully_masked.any():
                pad_mask[fully_masked] = False
        else:
            pad_mask = None
        out = x
        for i, layer in enumerate(self.transformer_layers):
            if self.use_checkpointing and self.training:
                out = self._checkpointed_transformer_layer(layer, out, pad_mask, dem_cond, i % 2 == 0)
            else:
                out = layer(out, src_key_padding_mask=pad_mask)
                if dem_cond is not None and i % 2 == 0:
                    cls_token = out[:, 0:1, :]
                    seq_tokens = out[:, 1:, :]
                    seq_tokens = self.dem_film(seq_tokens, dem_cond)
                    out = torch.cat([cls_token, seq_tokens], dim=1)
        out = self.norm(out)
        cls_out = out[:, 0, :]
        seq_out = out[:, 1:T_ext + 1, :]
        return cls_out, seq_out


class LateFusion(nn.Module):
    """2-input late fusion (opt, sar) — used by V4. Different from _base.LateFusion (3-input)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))

    def forward(self, opt: torch.Tensor, sar: torch.Tensor):
        gate = self.gate(torch.cat([opt, sar], dim=-1))
        blend = gate * opt + (1 - gate) * sar
        return self.proj(torch.cat([blend, opt], dim=-1))
