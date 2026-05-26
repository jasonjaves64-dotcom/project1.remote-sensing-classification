"""FusionCropNetV6 — the V6 architecture with all enhancements enabled by default.

This is a thin subclass of FusionCropNetV5EDL that activates V6 enhancements
via use_v6_enhancements=True. All V6 components live in V5EDL behind this flag.

V6 differs from V5EDL in:
  - use_v6_enhancements: ON (activates TemporalLite, Multi-Scale CrossAttn, etc.)
  - Gradient Checkpointing: ON by default (30% memory reduction)
  - Modality Dropout: ON (p=0.1) for robustness
  - RS Weights: SeCo path expected (falls back to ImageNet if not provided)
  - forward(): always returns aux outputs (multi-task heads)
  - training_step(): uses V6 multi-task loss
"""
import torch
import torch.nn.functional as F
from .fusion_net_v5_edl import FusionCropNetV5EDL, EDLLoss, training_step as v5edl_training_step
from ._base import PhenologyAuxHead, compute_ndvi_loss


class FusionCropNetV6(FusionCropNetV5EDL):
    """FusionCropNet V6 — Hierarchical Multi-Scale Multi-Task Architecture.

    Builds on V5EDL with all V6 enhancements enabled by default:
    - TemporalLite for s1/s2/opt_p2 temporal encoding
    - ModalNormalize + Early Fusion
    - DEM 5-path injection (Optical, Temporal, Decoder)
    - Multi-scale cross-modal attention (H, H/2, H/4)
    - Multi-task heads (LAI, GrowthStage, FieldBoundary)
    - LightSceneHead
    - Gradient Checkpointing (on)
    - Modality Dropout (on, p=0.1)

    Args:
        rs_weights: path to SeCo or other remote sensing pre-trained weights
        All other args same as FusionCropNetV5EDL
    """
    def __init__(self,
                 opt_ch: int = 10,
                 sar_ch: int = 5,
                 dem_ch_in: int = 5,
                 num_classes: int = 7,
                 feat_dim: int = 512,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 n_heads: int = 16,
                 win_size: int = 4,
                 n_layers: int = 4,
                 max_obs: int = 24,
                 n_freqs: int = 4,
                 drop_timestep_p: float = 0.1,
                 edl_dropout_p: float = 0.3,
                 edl_lambda_max: float = 0.5,
                 edl_anneal_ep: int = 50,
                 modality_dropout_p: float = 0.1,          # V6: ON by default
                 use_gradient_checkpointing: bool = True,   # V6: ON by default
                 rs_weights: str = None):
        super().__init__(
            opt_ch=opt_ch, sar_ch=sar_ch, dem_ch_in=dem_ch_in,
            num_classes=num_classes, feat_dim=feat_dim,
            backbone=backbone, pretrained=pretrained,
            n_heads=n_heads, win_size=win_size, n_layers=n_layers,
            max_obs=max_obs, n_freqs=n_freqs,
            drop_timestep_p=drop_timestep_p,
            edl_dropout_p=edl_dropout_p,
            edl_lambda_max=edl_lambda_max,
            edl_anneal_ep=edl_anneal_ep,
            modality_dropout_p=modality_dropout_p,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_v6_enhancements=True,
            rs_weights_path=rs_weights
        )

    def forward(self, opt_seq, sar_seq, dem, doy,
                cloud_mask=None, valid_count=None, epoch: int = 0,
                modality_mask=None):
        """V6 forward — always returns aux outputs.

        Returns:
            alpha: (B, K, H, W) Dirichlet parameters
            ndvi_pred: (B*T,) NDVI predictions
            consistency_loss: scalar or None
            aux: dict with keys:
                lai_pred, growth_pred, boundary_pred, scene_logits, crop_mix
        """
        result = super().forward(
            opt_seq, sar_seq, dem, doy,
            cloud_mask=cloud_mask, valid_count=valid_count,
            epoch=epoch, modality_mask=modality_mask,
            return_aux=True
        )

        alpha, ndvi_pred, consistency_loss, aux_tuple = result
        lai_pred, growth_pred, boundary_pred, scene_logits, crop_mix = aux_tuple

        if self.training:
            aux = {
                'lai': lai_pred,
                'growth': growth_pred,
                'boundary': boundary_pred,
                'scene_logits': scene_logits,
                'crop_mix': crop_mix,
            }
            return alpha, ndvi_pred, consistency_loss, aux
        return alpha  # eval: return alpha directly (compatible with V5EDL interface)


def v6_training_step(model: FusionCropNetV6, batch: dict,
                     edl_loss_fn: EDLLoss, epoch: int,
                     ndvi_channel: int = 6):
    """V6 training step with multi-task pseudo-label losses.

    Returns:
        total_loss: scalar tensor
        metrics: dict with per-task losses
    """
    from data.pseudo_labels import (
        generate_lai_pseudo, generate_growth_stage_pseudo, generate_boundary_pseudo
    )

    opt = batch['opt']; sar = batch['sar']; dem = batch['dem']
    doy = batch['doy']; y = batch['y']
    cm = batch.get('cloud_mask', None)
    vc = batch.get('valid_count', None)
    wm = batch.get('weight_map', None)

    alpha, ndvi_pred, consist_loss, aux = model(opt, sar, dem, doy, cm, vc, epoch=epoch)

    # EDL loss
    edl_loss = edl_loss_fn(alpha, y, epoch)

    if wm is not None:
        probs = alpha / alpha.sum(1, keepdim=True)
        log_p = torch.log(probs + 1e-8)
        px_ce = F.nll_loss(log_p, y.clamp(0), reduction='none', ignore_index=255)
        weighted_ce = (px_ce * wm)[y != 255].mean()
        current_lam = edl_loss_fn._current_lambda
        orig_ce = F.nll_loss(log_p, y.clamp(0), reduction='mean', ignore_index=255)
        kl_term = (edl_loss - orig_ce) / max(current_lam, 1e-8)
        edl_loss = weighted_ce + current_lam * kl_term

    ndvi_loss = compute_ndvi_loss(opt, ndvi_pred, cm, ndvi_channel)

    # LAI pseudo-labels
    ndvi_per_sample = opt[:, :, ndvi_channel].mean(dim=(-2, -1)).mean(dim=1)
    lai_target = generate_lai_pseudo(ndvi_per_sample).to(alpha.device)
    lai_loss = F.huber_loss(aux['lai'], lai_target, delta=1.0)

    # Growth stage pseudo-labels
    doy_mean = doy.mean(dim=1)
    growth_target = generate_growth_stage_pseudo(doy_mean.unsqueeze(1)).to(alpha.device)
    growth_loss = F.cross_entropy(aux['growth'], growth_target)

    # Boundary pseudo-labels
    boundary_target = generate_boundary_pseudo(dem).to(alpha.device)
    boundary_loss = F.binary_cross_entropy(aux['boundary'], boundary_target) + \
                    (1.0 - _dice_score(aux['boundary'], boundary_target))

    # Multi-task weighted total
    losses = {
        'crop': edl_loss,
        'ndvi': ndvi_loss * 0.1,
        'lai': lai_loss * 0.3,
        'growth': growth_loss * 0.2,
        'boundary': boundary_loss * 0.1,
    }
    total = model.multi_task_loss(losses) + 0.05 * consist_loss

    return total, {
        'edl_loss': edl_loss.item(),
        'ndvi_loss': ndvi_loss.item(),
        'lai_loss': lai_loss.item(),
        'growth_loss': growth_loss.item(),
        'boundary_loss': boundary_loss.item(),
        'consist': consist_loss.item(),
        'total_loss': total.item(),
    }


def _dice_score(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 7
    B, T, H, W = 2, 6, 128, 128

    print("=== FusionCropNetV6 Smoke Test ===")
    m = FusionCropNetV6(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=K,
        feat_dim=512, backbone="resnet18", pretrained=False,
        n_heads=4, n_layers=2
    ).to(dev)

    opt = torch.randn(B, T, 10, H, W, device=dev)
    sar = torch.randn(B, T, 5, H, W, device=dev)
    dem = torch.randn(B, 5, H, W, device=dev)
    doy = torch.rand(B, T, device=dev)

    # Test eval mode (returns alpha only, compatible with V5EDL interface)
    m.eval()
    with torch.no_grad():
        alpha = m(opt, sar, dem, doy)
    print(f"[eval] alpha={tuple(alpha.shape)}")

    # Test train mode
    m.train()
    alpha, ndvi, consist, aux = m(opt, sar, dem, doy)
    print(f"[train] alpha={tuple(alpha.shape)}, ndvi={tuple(ndvi.shape)}")
    print(f"        lai={tuple(aux['lai'].shape)}, growth={tuple(aux['growth'].shape)}")

    # Verify V6 defaults
    print(f"\n[V6 defaults]")
    print(f"  use_gradient_checkpointing = {m.use_grad_ckpt}")
    print(f"  modality_dropout_p = {m.modality_dropout_p}")

    # Verify all V6 components exist
    v6_attrs = ['temp_lite_s1', 'temp_lite_s2', 'temp_lite_opt_p2',
                'modal_norm', 'early_fusion',
                'dem_opt_cond', 'dem_temporal_proj',
                'cross_modal_h', 'cross_modal_h2',
                'lai_head', 'growth_head', 'boundary_head',
                'scene_head', 'multi_task_loss']
    for attr in v6_attrs:
        assert hasattr(m, attr), f"Missing V6 component: {attr}"
    print(f"  All {len(v6_attrs)} V6 components present")

    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    print("\n=== V6 Smoke Test PASSED ===")
