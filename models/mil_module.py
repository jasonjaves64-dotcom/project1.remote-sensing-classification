"""
MIL (Multi-Instance Learning) module for FusionCropNetV5EDL.

Transforms pixel-level supervised classification into bag-level (image-level)
weakly supervised learning. An image = bag, patches = instances.

Architecture:
  Patches → Shared Encoder → Instance Features → MIL Pooling → Bag Feature
                                                                    ↓
                                              Bag Classifier → Class + Uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from models.fusion_net_v5_edl import (
    FusionCropNetV5EDL, EDLHead, EDLLoss,
    dirichlet_to_predictions, evidence_level_fusion
)


# =============================================================================
# MIL Pooling Mechanisms
# =============================================================================

class MILAttentionPooling(nn.Module):
    """Multi-head self-attention pooling for instance aggregation.

    Learns instance importance weights via a trainable query vector.
    Supports variable-length bags through padding masks.

    Input:  instance features (B, N, D), mask (B, N) optional
    Output: bag feature (B, D), attention weights (B, N)
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape

        q = self.q_proj(self.query).view(1, 1, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # (1, H, 1, hd)

        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, H, N, hd)

        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, H, N, hd)

        scale = math.sqrt(self.head_dim)
        attn = (q * k).sum(dim=-1) / scale  # (B, H, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        if mask is not None:
            attn = attn * mask.unsqueeze(1).float()

        weighted = (attn.unsqueeze(-1) * v).sum(dim=2)  # (B, H, hd)
        weighted = weighted.transpose(1, 2).reshape(B, D)
        bag_feat = self.out_proj(weighted)

        attn_weights = attn.mean(dim=1)  # average over heads (B, N)
        return bag_feat, attn_weights


class GatedMILPooling(nn.Module):
    """Gated attention pooling (Ilse et al. 2018).

    Uses a learnable context vector with sigmoid-gated tanh activation,
    proven effective for MIL in medical imaging and remote sensing.
    """

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape

        a_v = self.attention_v(x)  # (B, N, H)
        a_u = self.attention_u(x)  # (B, N, H)
        a = self.attention_w(a_v * a_u).squeeze(-1)  # (B, N)

        if mask is not None:
            a = a.masked_fill(~mask, float('-inf'))

        weights = a.softmax(dim=-1)  # (B, N)
        bag_feat = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        return bag_feat, weights


# =============================================================================
# Bag-Level Classifier with EDL
# =============================================================================

class BagEDLClassifier(nn.Module):
    """Bag-level classifier with Evidential Deep Learning uncertainty.

    Takes the aggregated bag feature and outputs Dirichlet alpha parameters
    for bag-level classification with uncertainty estimation.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256,
                 dropout_p: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, bag_feat: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(bag_feat)
        alpha = F.softplus(logits) + 1.0  # Dirichlet evidence (B, K)
        return alpha


def bag_dirichlet_to_predictions(alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert bag-level Dirichlet alpha to predictions and uncertainty.

    Args:
        alpha: (B, K) Dirichlet concentration parameters

    Returns:
        dict with probs, pred_class, vacuity, dissonance, class_var
    """
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)  # (B, 1)
    probs = alpha / S  # (B, K)
    pred = probs.argmax(dim=1)  # (B,)
    vacuity = K / S.squeeze(1)  # (B,) — data uncertainty
    dissonance = 1.0 - (probs * probs).sum(dim=1)  # (B,) — epistemic uncertainty
    class_var = probs * (1.0 - probs) / (S + 1)  # (B, K)

    return {
        "probs": probs,
        "pred_class": pred,
        "vacuity": vacuity,
        "dissonance": dissonance,
        "class_var": class_var,
    }


# =============================================================================
# Instance Feature Extractor (from existing encoder)
# =============================================================================

class InstanceFeatureExtractor(nn.Module):
    """Extracts fixed-dimension feature vectors from image patches using
    the FusionCropNetV5EDL encoder.

    Each patch goes through the full encoder → global average pool → D-dim vector.
    """

    def __init__(self, base_model: FusionCropNetV5EDL, feature_dim: int = 512):
        super().__init__()
        self.encoder = base_model
        self.feature_dim = feature_dim
        # Pre-allocate feature projection (avoids creating nn.Linear in forward)
        self.feat_proj = nn.Linear(64, feature_dim)

    def forward(self, opt_patch: torch.Tensor, sar_patch: torch.Tensor,
                dem_patch: torch.Tensor, doy_patch: torch.Tensor,
                cloud_mask: Optional[torch.Tensor] = None,
                valid_count: Optional[torch.Tensor] = None,
                modality_mask=None) -> torch.Tensor:
        """Extract features from patches.

        Args:
            opt_patch:  (B, T, 10, P, P) optical patches
            sar_patch:  (B, T, 5,  P, P) SAR patches
            dem_patch:  (B, 5, P, P)     DEM patches
            doy_patch:  (B, T)           DOY values
            cloud_mask: (B, T, P, P) or None
            valid_count:(B, P, P) or None
            modality_mask: None or (use_opt, use_sar, use_dem)

        Returns:
            instance_feat: (B, D) pooled instance features
        """
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            pre_head, *_ = self.encoder._encode(
                opt_patch, sar_patch, dem_patch, doy_patch, cloud_mask, valid_count,
                modality_mask=modality_mask)
        if was_training:
            self.encoder.train()

        instance_feat = pre_head.mean(dim=[2, 3])  # GAP: (B, 64, H, W) → (B, 64)
        instance_feat = self.feat_proj(instance_feat)  # (B, 64) → (B, feature_dim)
        return instance_feat


# =============================================================================
# Full MIL Model
# =============================================================================

class FusionCropNetV5MIL(nn.Module):
    """MIL wrapper for FusionCropNetV5EDL.

    Architecture:
      Image → [Patch₁, Patch₂, ..., Patchₙ]  (sliding window)
                ↓         ↓             ↓
            Encoder   Encoder   ...  Encoder   (shared weights)
                ↓         ↓             ↓
            feat₁     feat₂         featₙ     (D-dim vectors)
                └─────────┬──────────┘
                          ↓
                  MIL Pooling  (attention-based aggregation)
                          ↓
                   Bag Feature  (D-dim)
                          ↓
             Bag EDL Classifier → α (Dirichlet params)
                          ↓
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
          probs       vacuity      dissonance
    """

    def __init__(self, base_model: FusionCropNetV5EDL,
                 num_classes: int = 7,
                 pool_method: str = "attention",
                 feature_dim: int = 512,
                 freeze_encoder: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.pool_method = pool_method.lower()
        self.freeze_encoder = freeze_encoder

        self.extractor = InstanceFeatureExtractor(base_model, feature_dim)
        if freeze_encoder:
            for p in self.extractor.encoder.parameters():
                p.requires_grad = False

        if self.pool_method == "attention":
            self.pool = MILAttentionPooling(feature_dim)
        elif self.pool_method == "gated":
            self.pool = GatedMILPooling(feature_dim)
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")

        self.classifier = BagEDLClassifier(feature_dim, num_classes)
        self.edl_loss_fn = EDLLoss(num_classes, lambda_max=0.3, kl_anneal_epochs=50)

    def forward(self, opt_bag: torch.Tensor, sar_bag: torch.Tensor,
                dem_bag: torch.Tensor, doy_bag: torch.Tensor,
                instance_mask: Optional[torch.Tensor] = None,
                cloud_mask: Optional[torch.Tensor] = None,
                valid_count: Optional[torch.Tensor] = None,
                epoch: int = 0,
                modality_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a bag of instances.

        Args:
            opt_bag:  (B, N, T, 10, P, P) — bag of N optical patches
            sar_bag:  (B, N, T, 5,  P, P) — bag of N SAR patches
            dem_bag:  (B, N, 5, P, P)     — bag of N DEM patches
            doy_bag:  (B, N, T)           — bag of N DOY vectors
            instance_mask: (B, N) — True for valid instances, False for padding
            cloud_mask: (B, N, T, P, P) or None
            valid_count:(B, N, P, P) or None
            epoch:     current epoch for KL annealing
            modality_mask: None or (use_opt, use_sar, use_dem) — propagates to encoder

        Returns:
            alpha:       (B, K) Dirichlet evidence for bag-level classification
            attn_weights:(B, N) instance attention weights
        """
        B, N = opt_bag.shape[0], opt_bag.shape[1]

        # Flatten bag dimension into batch for parallel encoding
        opt_flat = opt_bag.view(B * N, *opt_bag.shape[2:])
        sar_flat = sar_bag.view(B * N, *sar_bag.shape[2:])
        dem_flat = dem_bag.view(B * N, *dem_bag.shape[2:])
        doy_flat = doy_bag.view(B * N, *doy_bag.shape[2:])

        cm_flat = None
        if cloud_mask is not None:
            cm_flat = cloud_mask.view(B * N, *cloud_mask.shape[2:])

        vc_flat = None
        if valid_count is not None:
            vc_flat = valid_count.view(B * N, *valid_count.shape[2:])

        instance_feats = self.extractor(
            opt_flat, sar_flat, dem_flat, doy_flat, cm_flat, vc_flat,
            modality_mask=modality_mask)
        instance_feats = instance_feats.view(B, N, -1)  # (B, N, D)

        bag_feat, attn_weights = self.pool(instance_feats, instance_mask)
        alpha = self.classifier(bag_feat)  # (B, K)

        return alpha, attn_weights

    def compute_loss(self, alpha: torch.Tensor, bag_labels: torch.Tensor,
                     epoch: int = 0) -> torch.Tensor:
        """Compute EDL loss at bag level."""
        return self.edl_loss_fn(alpha, bag_labels, epoch)

    def predict_bag(self, opt_bag: torch.Tensor, sar_bag: torch.Tensor,
                    dem_bag: torch.Tensor, doy_bag: torch.Tensor,
                    instance_mask: Optional[torch.Tensor] = None,
                    n_passes: int = 5) -> dict:
        """Bag-level prediction with uncertainty via MC-Dropout ensemble.

        Returns dict with: probs, pred_class, vacuity, dissonance,
                          class_var, attn_weights, attn_heatmap
        """
        self.eval()
        for m in self.classifier.modules():
            if isinstance(m, nn.Dropout):
                m.train()  # Enable dropout for MC sampling

        alpha_list = []
        attn_list = []

        with torch.no_grad():
            for _ in range(n_passes):
                alpha, attn = self.forward(
                    opt_bag, sar_bag, dem_bag, doy_bag,
                    instance_mask=instance_mask)
                alpha_list.append(alpha)
                attn_list.append(attn)

        # Evidence-level fusion
        fused_alpha = evidence_level_fusion(alpha_list)
        preds = bag_dirichlet_to_predictions(fused_alpha)

        # Average attention across passes
        attn_avg = torch.stack(attn_list, dim=0).mean(dim=0)
        preds["attn_weights"] = attn_avg
        preds["alpha"] = fused_alpha

        return preds


def _build_default_base_model(config: dict = None) -> FusionCropNetV5EDL:
    """Build a default FusionCropNetV5EDL from config."""
    if config is None:
        config = {
            "opt_channels": 10, "sar_channels": 5, "dem_channels": 5,
            "num_classes": 7, "feat_dim": 512, "backbone": "resnet50",
            "pretrained": True, "n_heads": 16, "win_size": 4,
            "n_layers": 4, "drop_timestep_p": 0.1,
            "edl_dropout_p": 0.3,
        }

    return FusionCropNetV5EDL(
        opt_ch=config["opt_channels"],
        sar_ch=config["sar_channels"],
        dem_ch_in=config["dem_channels"],
        num_classes=config["num_classes"],
        feat_dim=config["feat_dim"],
        backbone=config["backbone"],
        pretrained=config["pretrained"],
        n_heads=config["n_heads"],
        win_size=config["win_size"],
        n_layers=config["n_layers"],
        drop_timestep_p=config["drop_timestep_p"],
        edl_dropout_p=config["edl_dropout_p"],
    )


def create_mil_model(base_model: FusionCropNetV5EDL = None,
                     config: dict = None,
                     pool_method: str = "attention",
                     freeze_encoder: bool = True) -> FusionCropNetV5MIL:
    """Factory function to create a MIL model.

    Args:
        base_model: Pre-trained FusionCropNetV5EDL. If None, built from config.
        config:     Model configuration dict.
        pool_method:"attention" or "gated".
        freeze_encoder: Whether to freeze the encoder during MIL training.

    Returns:
        FusionCropNetV5MIL instance
    """
    if base_model is None:
        base_model = _build_default_base_model(config)

    num_classes = config.get("num_classes", 7) if config else 7
    return FusionCropNetV5MIL(
        base_model=base_model,
        num_classes=num_classes,
        pool_method=pool_method,
        freeze_encoder=freeze_encoder,
    )
