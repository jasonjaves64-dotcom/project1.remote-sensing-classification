# =============================================================================
# FusionCropNet V5 + Hybrid EDL-Ensemble Framework
#
# All shared components imported from ._base (single canonical source).
# EDL-specific: EDLHead, EDLLoss, dirichlet_to_predictions,
#               evidence_level_fusion, FusionCropNetV5EDL, training_step
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import (
    ConvBNGELU, SEBlock, FiLM, IRB,
    DEMEncoder, FPN, OpticalEncoder, SAREncoder,
    FourierDOYEncoding, ObsQualityToken, TemporalEncoderStream,
    CrossModalAttention, CrossModalLite, DEMSpatialConditioner, LateFusion,
    SpatialRefinement, Decoder, PhenologyAuxHead,
    LightSceneHead, time_average, ModalNormalize, DEMOpticalConditioner,
    compute_ndvi_loss,
)

# V6 Block 1: Lightweight temporal encoder
from .temporal_lite import TemporalLite
# V6 Block 5: Multi-task pseudo-label heads
from .multi_task_heads import LAIRegressionHead, GrowthStageHead, BoundaryHead, MultiTaskLoss


# =============================================================================
# EDL Head — Evidential Deep Learning output layer
# =============================================================================
class EDLHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, dropout_p: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout2d(dropout_p)
        self.net = nn.Sequential(
            ConvBNGELU(in_ch, in_ch),
            nn.Conv2d(in_ch, in_ch, 1),
            nn.GELU(),
            nn.Conv2d(in_ch, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output alpha = softplus(logits) + 1, not raw logits."""
        x = self.dropout(x)
        logits = self.net(x)
        return F.softplus(logits) + 1.0


# =============================================================================
# EDL Utilities
# =============================================================================
def dirichlet_to_predictions(alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    pred = probs.argmax(dim=1)
    vacuity = K / S.squeeze(1)
    p = probs
    dissonance = 1.0 - (p * p).sum(dim=1)
    S_sq = S + 1
    c_var = probs * (1.0 - probs) / S_sq
    return {
        "probs": probs, "pred_class": pred,
        "vacuity": vacuity, "dissonance": dissonance,
        "class_var": c_var,
    }


def evidence_level_fusion(alpha_list: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(alpha_list, dim=0)
    return stacked.mean(dim=0)


# =============================================================================
# EDL Loss
# =============================================================================
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

    def get_lambda(self, epoch: int, spear_r: float = None) -> float:
        """Compute KL weight: linear anneal. When adaptive + vacuity-error correlation > 0.3,
        accelerate the effective epoch (evidence of good uncertainty calibration)."""
        eff_epoch = float(epoch)
        if self.adaptive and spear_r is not None and spear_r > 0.3:
            eff_epoch = epoch * (1.0 + 1.5 * (spear_r - 0.3))
        base = self.lambda_max * min(1.0, eff_epoch / max(self.kl_anneal_epochs, 1))
        self._current_lambda = base
        return base

    def _kl_uniform_dirichlet(self, alpha_tilde: torch.Tensor) -> torch.Tensor:
        K = self.K
        S = alpha_tilde.sum(dim=-1)
        kl = (torch.lgamma(S)
              - torch.lgamma(torch.tensor(float(K), device=S.device))
              - torch.lgamma(alpha_tilde).sum(dim=-1)
              + ((alpha_tilde - 1) *
                 (torch.digamma(alpha_tilde) -
                  torch.digamma(S.unsqueeze(-1)))).sum(dim=-1))
        return kl

    def forward(self, alpha: torch.Tensor, targets: torch.Tensor,
                epoch: int, spear_r: float = None) -> torch.Tensor:
        B, K, H, W = alpha.shape
        lam = self.get_lambda(epoch, spear_r)
        alpha_flat = alpha.permute(0, 2, 3, 1).reshape(-1, K)
        targets_flat = targets.reshape(-1)
        valid = targets_flat != self.ignore_index
        alpha_v = alpha_flat[valid]
        tgt_v = targets_flat[valid]
        S = alpha_v.sum(dim=-1, keepdim=True)
        probs = alpha_v / S
        log_p = torch.log(probs + 1e-8)
        ce = F.nll_loss(log_p, tgt_v, reduction='mean')
        one_hot = F.one_hot(tgt_v, K).float()
        alpha_tilde = alpha_v * (1.0 - one_hot) + one_hot  # regularize non-target only
        kl = self._kl_uniform_dirichlet(alpha_tilde).mean()
        return ce + lam * kl


# =============================================================================
# FusionCropNetV5EDL — Full EDL model with modality decoupling
# =============================================================================
class FusionCropNetV5EDL(nn.Module):
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
                 modality_dropout_p: float = 0.0,
                 use_gradient_checkpointing: bool = False,
                 use_v6_enhancements: bool = False,
                 rs_weights_path: str = None):
        super().__init__()
        self.use_v6 = use_v6_enhancements
        self.modality_dropout_p = modality_dropout_p
        dem_ch = 128
        self.dem_enc = DEMEncoder(dem_ch_in, dem_ch)
        self.opt_enc = OpticalEncoder(opt_ch, feat_dim, backbone, pretrained,
                                       rs_weights_path=rs_weights_path)
        self.sar_enc = SAREncoder(sar_ch, 32, feat_dim, dem_ch)
        self.opt_temporal = TemporalEncoderStream(feat_dim, n_heads=8, n_layers=n_layers,
                                                   max_obs=max_obs, n_freqs=n_freqs)
        self.sar_temporal = TemporalEncoderStream(feat_dim, n_heads=8, n_layers=n_layers,
                                                   max_obs=max_obs, n_freqs=n_freqs)
        self.cross_modal = CrossModalAttention(feat_dim, n_heads, win_size)
        self.dem_cond = DEMSpatialConditioner(feat_dim, dem_ch)
        self.late_fuse = LateFusion(feat_dim)
        self.decoder = Decoder(feat_dim,
                               sar_ch_list=self.sar_enc.out_channels_list[:2],
                               n_heads=8, win=win_size)
        self.edl_head = EDLHead(self.decoder.pre_head_ch, num_classes, edl_dropout_p)
        self.edl_loss_fn = EDLLoss(num_classes, edl_lambda_max, edl_anneal_ep)
        self.pheno_aux = PhenologyAuxHead(feat_dim, aux_weight=0.3)

        # ── V6 enhancements (guarded by use_v6 flag) ──
        if self.use_v6:
            # Block 1: TemporalLite for high-resolution temporal encoding
            self.temp_lite_s1 = TemporalLite(64, k=3)
            self.temp_lite_s2 = TemporalLite(128, k=3)
            self.temp_lite_opt_p2 = TemporalLite(256, k=3)
            # Block 2: Early Fusion
            self.modal_norm = ModalNormalize()
            self.early_fusion = nn.Conv2d(10 + 5 + 5, 128, 1, bias=False)
            # Block 4: Multi-scale cross-modal attention
            self.cross_modal_h = CrossModalLite(64, n_heads=1)
            self.cross_modal_h2 = CrossModalLite(128, n_heads=4)
            self.opt_to_h = nn.Conv2d(256, 64, 1, bias=False)
            self.opt_to_h2 = nn.Conv2d(256, 128, 1, bias=False)
            # Block 3: DEM → Optical FiLM + Temporal modulation
            self.dem_opt_cond = DEMOpticalConditioner(feat_dim, dem_ch)
            self.dem_temporal_proj = nn.Sequential(
                nn.Linear(dem_ch, dem_ch // 2), nn.GELU(),
                nn.Linear(dem_ch // 2, feat_dim))
            # Block 5: Multi-task pseudo-label heads
            self.lai_head = LAIRegressionHead(self.decoder.pre_head_ch)
            self.growth_head = GrowthStageHead(self.decoder.pre_head_ch)
            self.boundary_head = BoundaryHead(self.decoder.pre_head_ch)
            # Block 7: Lightweight scene understanding
            self.scene_head = LightSceneHead(self.decoder.pre_head_ch, hidden=256,
                                              num_scene_types=4, num_crops=num_classes)
            self.multi_task_loss = MultiTaskLoss(num_tasks=5)
            self.consistency_proj = nn.Linear(feat_dim, 1)
            self.consistency_target_proj = nn.Linear(feat_dim, 1)
        self.drop_timestep_p = drop_timestep_p
        self.use_grad_ckpt = use_gradient_checkpointing

        self.placeholder_opt = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.placeholder_sar = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.placeholder_dem_feat = nn.Parameter(torch.zeros(1, dem_ch, 1, 1))
        nn.init.trunc_normal_(self.placeholder_opt, std=0.02)
        nn.init.trunc_normal_(self.placeholder_sar, std=0.02)
        nn.init.trunc_normal_(self.placeholder_dem_feat, std=0.02)

        self.fallback_gate_opt = nn.Parameter(torch.tensor(0.5))
        self.fallback_gate_sar = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        pretrained_modules = set()
        for m in self.opt_enc.backbone.modules():
            pretrained_modules.add(id(m))
        for m in self.modules():
            if id(m) in pretrained_modules:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _to_pixel_seq(self, feat, B, T, H2, W2, D):
        return feat.view(B, T, D, H2, W2).permute(0, 3, 4, 1, 2).reshape(B * H2 * W2, T, D)

    def _shift_inputs(self, opt_seq, sar_seq, dem, cloud_mask, valid_count, sx, sy):
        if sx == 0 and sy == 0:
            return opt_seq, sar_seq, dem, cloud_mask, valid_count
        pad_l = abs(sx) if sx < 0 else 0
        pad_r = abs(sx) if sx > 0 else 0
        pad_t = abs(sy) if sy < 0 else 0
        pad_b = abs(sy) if sy > 0 else 0
        # Pad only the last 2 spatial dims (H, W). For 5D tensors prepend (0,0) for channel.
        pad_hw = (pad_l, pad_r, pad_t, pad_b)
        opt_seq = F.pad(opt_seq, pad_hw if opt_seq.dim() == 4 else pad_hw + (0, 0), mode='replicate')
        sar_seq = F.pad(sar_seq, pad_hw if sar_seq.dim() == 4 else pad_hw + (0, 0), mode='replicate')
        dem = F.pad(dem, pad_hw, mode='replicate')
        # Crop back to original spatial size
        H_orig, W_orig = dem.shape[-2], dem.shape[-1]
        slice_t = slice(pad_t, H_orig - pad_b) if pad_t + pad_b > 0 else slice(None)
        slice_l = slice(pad_l, W_orig - pad_r) if pad_l + pad_r > 0 else slice(None)
        opt_seq = opt_seq[:, :, :, slice_t, slice_l]
        sar_seq = sar_seq[:, :, :, slice_t, slice_l]
        dem = dem[:, :, slice_t, slice_l]
        if cloud_mask is not None:
            cm_float = cloud_mask.float() if cloud_mask.dtype == torch.bool else cloud_mask
            cm_pad = F.pad(cm_float, pad_hw if cm_float.dim() == 4 else pad_hw + (0, 0), mode='replicate')
            cloud_mask = (cm_pad > 0.5) if cloud_mask.dtype == torch.bool else cm_pad
            cloud_mask = cloud_mask[:, :, slice_t, slice_l]
        if valid_count is not None:
            vc_exp = valid_count.float().unsqueeze(1) if valid_count.dim() == 3 else valid_count.float()
            vc_exp = F.pad(vc_exp, pad_hw, mode='constant')
            vc_exp = vc_exp[:, :, slice_t, slice_l] if vc_exp.dim() == 4 else vc_exp[:, slice_t, slice_l]
            valid_count = vc_exp.squeeze(1).long() if valid_count.dim() == 3 else vc_exp.long()
        return opt_seq, sar_seq, dem, cloud_mask, valid_count

    def _encode(self, opt_seq, sar_seq, dem, doy, cloud_mask, valid_count,
                modality_mask=None, dem_ablation=None):
        """Forward pass with optional DEM ablation control.

        Args:
            dem_ablation: optional dict to disable specific DEM injection points.
                Keys: 'sar_film', 'spatial_cond', 'decoder_skip',
                      'early_fusion', 'opt_cond', 'temporal_bias'.
                True = enabled (default), False = disabled.
                None = all enabled.
        """
        if dem_ablation is None:
            dem_ablation = {}
        _dem = lambda k: dem_ablation.get(k, True)  # default: enabled

        if modality_mask is None:
            use_opt, use_sar, use_dem = True, True, True
        else:
            use_opt, use_sar, use_dem = modality_mask

        if self.training and self.modality_dropout_p > 0:
            if torch.rand(1).item() < self.modality_dropout_p:
                r = torch.rand(1).item()
                if r < 0.33:      use_opt = False
                elif r < 0.66:    use_sar = False
                else:             use_dem = False

        B, T = opt_seq.shape[0], opt_seq.shape[1]
        Co = opt_seq.shape[2]
        H, W = opt_seq.shape[3], opt_seq.shape[4]

        if self.training:
            sx = torch.randint(-1, 2, (1,)).item()
            sy = torch.randint(-1, 2, (1,)).item()
            opt_seq, sar_seq, dem, cloud_mask, valid_count = self._shift_inputs(
                opt_seq, sar_seq, dem, cloud_mask, valid_count, sx, sy)

        if use_dem:
            dem_feat = self.dem_enc(dem)
            if self.training:
                dem_feat = dem_feat + 0.01 * torch.randn_like(dem_feat)
        else:
            dem_feat = self.placeholder_dem_feat.expand(B, -1, H, W)

        dem_feat = F.interpolate(dem_feat, (H, W), mode='bilinear', align_corners=False)

        # V6 Block 2: Early Fusion (only when use_v6=True)
        if self.use_v6 and use_opt and use_sar and use_dem and _dem('early_fusion'):
            opt_for_early = opt_seq[:, 0] if opt_seq.dim() == 5 else opt_seq
            sar_for_early = sar_seq[:, 0] if sar_seq.dim() == 5 else sar_seq
            unified_early = self.modal_norm(opt_for_early, sar_for_early, dem)
            unified_early = self.early_fusion(unified_early)
        else:
            unified_early = None

        if use_opt:
            opt_flat = opt_seq.view(B * T, Co, H, W)
            opt_feat, opt_p2, opt_p3 = self.opt_enc(opt_flat)
            ndvi_pred = self.pheno_aux(opt_feat)
        else:
            H2, W2 = H // 4, W // 4
            opt_feat = self.placeholder_opt.expand(B * T, -1, H2, W2)
            D_ref = opt_feat.shape[1]
            opt_p2 = torch.zeros(B * T, D_ref // 2, H2 * 2, W2 * 2, device=opt_feat.device)
            opt_p3 = opt_p2.clone()
            ndvi_pred = torch.zeros(B * T, device=opt_feat.device)

        _, D, H2, W2 = opt_feat.shape

        # V6 Block 3: DEM → Optical FiLM (only when use_v6=True)
        if self.use_v6 and use_dem and use_opt and _dem('opt_cond'):
            dem_feat_tiled = dem_feat.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, -1, H, W)
            opt_feat, opt_p2 = self.dem_opt_cond(opt_feat, opt_p2, dem_feat_tiled)

        if use_sar:
            sar_flat = sar_seq.view(B * T, sar_seq.shape[2], H, W)
            dem_tiled = dem_feat.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, -1, H, W)
            sar_s1, sar_s2, sar_s3 = self.sar_enc(sar_flat, dem_feat=dem_tiled if _dem('sar_film') else None)
        else:
            sar_s1 = torch.zeros(B * T, 64, H, W, device=opt_feat.device)
            sar_s2 = torch.zeros(B * T, 128, H // 2, W // 2, device=opt_feat.device)
            sar_s3 = self.placeholder_sar.expand(B * T, -1, H2, W2)

        if cloud_mask is not None and use_opt:
            cm_down = F.adaptive_avg_pool2d(cloud_mask.view(B * T, 1, H, W).float(), (H2, W2))
            cm_px = ((cm_down > 0.5).squeeze(1)
                     .view(B, T, H2 * W2).permute(0, 2, 1).reshape(B * H2 * W2, T))
        else:
            cm_px = None
        if self.training and self.drop_timestep_p > 0:
            drop = torch.rand(B * H2 * W2, T, device=opt_feat.device) < self.drop_timestep_p
            cm_px = (cm_px | drop) if cm_px is not None else drop

        vc_px = None
        if valid_count is not None:
            vc = F.adaptive_avg_pool2d(valid_count.float().unsqueeze(1), (H2, W2)).squeeze(1).long()
            vc_px = vc.view(B * H2 * W2)
        doy_px = doy.unsqueeze(1).unsqueeze(1).expand(B, H2, W2, T).reshape(B * H2 * W2, T)

        opt_ts = self._to_pixel_seq(opt_feat, B, T, H2, W2, D)
        sar_ts = self._to_pixel_seq(sar_s3, B, T, H2, W2, D)
        # V6 Block 3: DEM → Temporal FiLM (only when use_v6=True)
        if self.use_v6 and use_dem and _dem('temporal_bias'):
            dem_temporal = F.adaptive_avg_pool2d(dem_feat, (H2, W2))
            dem_temporal_flat = dem_temporal.permute(0, 2, 3, 1).reshape(B * H2 * W2, -1)
            dem_temporal_bias = self.dem_temporal_proj(dem_temporal_flat).unsqueeze(1)
            opt_ts = opt_ts + dem_temporal_bias
            sar_ts = sar_ts + dem_temporal_bias
        if self.use_grad_ckpt and self.training:
            opt_g, opt_seq_out = torch.utils.checkpoint.checkpoint(
                self.opt_temporal, opt_ts, doy_px, cm_px, vc_px,
                use_reentrant=False
            )
        else:
            opt_g, opt_seq_out = self.opt_temporal(opt_ts, doy_px, cloud_mask=cm_px, valid_count=vc_px)

        if self.use_grad_ckpt and self.training:
            sar_g, _ = torch.utils.checkpoint.checkpoint(
                self.sar_temporal, sar_ts, doy_px, None, vc_px, opt_g,
                use_reentrant=False
            )
        else:
            sar_g, _ = self.sar_temporal(sar_ts, doy_px, cloud_mask=None, valid_count=vc_px, fallback_feat=opt_g)
        opt_global = opt_g.view(B, H2, W2, D).permute(0, 3, 1, 2)
        sar_global = sar_g.view(B, H2, W2, D).permute(0, 3, 1, 2)

        if use_opt and use_sar:
            xm_feat = self.cross_modal(opt_global, sar_global)
        elif use_opt:
            xm_feat = opt_global
        else:
            xm_feat = sar_global

        if use_dem and _dem('spatial_cond'):
            xm_feat = self.dem_cond(xm_feat, dem_feat)

        xm_f = xm_feat.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)
        opt_f = opt_global.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)
        sar_f = sar_global.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)

        if use_opt and use_sar:
            final = self.late_fuse(xm_f, opt_f, sar_f)
        else:
            final = xm_f

        final = final.view(B, H2, W2, D).permute(0, 3, 1, 2)

        # V6 Block 1+4: TemporalLite + Multi-scale cross-modal attention
        if self.use_v6:
            opt_p2_ch = opt_p2.shape[1]
            sar_s1_ch = sar_s1.shape[1]
            sar_s2_ch = sar_s2.shape[1]
            opt_p2_seq = self._to_pixel_seq(opt_p2, B, T, H//2, W//2, opt_p2_ch)
            sar_s1_seq = self._to_pixel_seq(sar_s1, B, T, H, W, sar_s1_ch)
            sar_s2_seq = self._to_pixel_seq(sar_s2, B, T, H//2, W//2, sar_s2_ch)
            opt_p2a = self.temp_lite_opt_p2(opt_p2_seq).view(B, H//2, W//2, opt_p2_ch).permute(0, 3, 1, 2)
            sar_s1a = self.temp_lite_s1(sar_s1_seq).view(B, H, W, sar_s1_ch).permute(0, 3, 1, 2)
            sar_s2a = self.temp_lite_s2(sar_s2_seq).view(B, H//2, W//2, sar_s2_ch).permute(0, 3, 1, 2)
            # H scale: opt projection ↔ sar_s1
            if use_opt and use_sar:
                opt_h = self.opt_to_h(opt_p2a)
                opt_h = F.interpolate(opt_h, size=sar_s1a.shape[-2:], mode='bilinear', align_corners=False)
                cross_h = self.cross_modal_h(opt_h, sar_s1a)
            elif use_opt:
                opt_h = self.opt_to_h(opt_p2a)
                cross_h = F.interpolate(opt_h, size=sar_s1a.shape[-2:], mode='bilinear', align_corners=False)
            else:
                cross_h = sar_s1a
            # H/2 scale: opt projection ↔ sar_s2
            if use_opt and use_sar:
                opt_h2 = self.opt_to_h2(opt_p2a)
                cross_h2 = self.cross_modal_h2(opt_h2, sar_s2a)
            elif use_opt:
                cross_h2 = self.opt_to_h2(opt_p2a)
            else:
                cross_h2 = sar_s2a
            pre_head = self.decoder(final,
                                    opt_skips=(opt_p2a,),
                                    sar_skips=(cross_h, cross_h2),
                                    dem_skip=dem_feat if (use_dem and _dem('decoder_skip')) else None,
                                    early_skip=unified_early,
                                    target_size=(H, W))
        else:
            # V5EDL original path: time_average for skip connections
            opt_p2_reshaped = opt_p2.view(B, T, -1, H // 2, W // 2)
            opt_p2_avg = opt_p2_reshaped.mean(dim=1)  # (B, C, H//2, W//2)
            sar_s1_reshaped = sar_s1.view(B, T, -1, H, W)
            sar_s1_avg = sar_s1_reshaped.mean(dim=1)   # (B, 64, H, W)
            sar_s2_reshaped = sar_s2.view(B, T, -1, H // 2, W // 2)
            sar_s2_avg = sar_s2_reshaped.mean(dim=1)   # (B, 128, H//2, W//2)
            pre_head = self.decoder(final,
                                    opt_skips=(opt_p2_avg,),
                                    sar_skips=(sar_s1_avg, sar_s2_avg),
                                    dem_skip=dem_feat if (use_dem and _dem('decoder_skip')) else None,
                                    early_skip=unified_early,
                                    target_size=(H, W))
        return pre_head, ndvi_pred, opt_seq_out, cm_px, B, T, H2, W2, D, H, W, opt_f

    def forward(self, opt_seq, sar_seq, dem, doy,
                cloud_mask=None, valid_count=None, epoch: int = 0,
                modality_mask=None, return_aux: bool = False,
                dem_ablation=None):
        (pre_head, ndvi_pred, opt_seq_out,
         cm_px, B, T, H2, W2, D, H, W, opt_f) = self._encode(
             opt_seq, sar_seq, dem, doy, cloud_mask, valid_count,
             modality_mask=modality_mask, dem_ablation=dem_ablation)
        alpha = self.edl_head(pre_head)

        # V6 Block 5+7: Multi-task auxiliary predictions (only when use_v6=True)
        if self.use_v6 and return_aux:
            lai_pred = self.lai_head(pre_head)
            growth_pred = self.growth_head(pre_head)
            boundary_pred = self.boundary_head(pre_head)
            scene_logits, crop_mix = self.scene_head(pre_head)

        if self.training:
            if self.use_v6 and cm_px is not None:
                certainty = torch.sigmoid(self.consistency_proj(opt_seq_out)).squeeze(-1)
                unmasked_mask = ~cm_px
                if unmasked_mask.any():
                    target_certainty = torch.sigmoid(
                        self.consistency_target_proj(opt_seq_out).squeeze(-1))
                    consistency_loss = F.mse_loss(
                        certainty[unmasked_mask], target_certainty[unmasked_mask])
                else:
                    consistency_loss = alpha.sum() * 0.0
            else:
                consistency_loss = alpha.sum() * 0.0
            if self.use_v6 and return_aux:
                return alpha, ndvi_pred, consistency_loss, (lai_pred, growth_pred, boundary_pred, scene_logits, crop_mix)
            return alpha, ndvi_pred, consistency_loss
        # Eval mode
        if self.use_v6 and return_aux:
            return alpha, ndvi_pred, None, (lai_pred, growth_pred, boundary_pred, scene_logits, crop_mix)
        return alpha

    def predict_uncertainty(self, opt_seq, sar_seq, dem, doy,
                            cloud_mask=None, valid_count=None,
                            n_passes: int = 10,
                            use_tta: bool = True,
                            modality_mask=None,
                            dem_ablation=None) -> dict[str, torch.Tensor]:
        self.eval()
        for m in self.edl_head.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()

        alpha_list = []
        inputs_orig = (opt_seq, sar_seq, dem, doy, cloud_mask, valid_count)

        def _single_pass(args):
            pre_head, *_ = self._encode(*args, modality_mask=modality_mask,
                                        dem_ablation=dem_ablation)
            return self.edl_head(pre_head)

        with torch.no_grad():
            for _ in range(n_passes):
                alpha_list.append(_single_pass(inputs_orig))
            if use_tta:
                opt_f = opt_seq.flip(-1)
                sar_f = sar_seq.flip(-1)
                dem_f = dem.flip(-1)
                cm_f = cloud_mask.flip(-1) if cloud_mask is not None else None
                vc_f = valid_count.flip(-1) if valid_count is not None else None
                inputs_flip = (opt_f, sar_f, dem_f, doy, cm_f, vc_f)
                for _ in range(n_passes):
                    alpha_list.append(_single_pass(inputs_flip).flip(-1))

        for m in self.edl_head.modules():
            if isinstance(m, nn.Dropout2d):
                m.eval()

        alpha_fused = evidence_level_fusion(alpha_list)
        result = dirichlet_to_predictions(alpha_fused)
        result['alpha_fused'] = alpha_fused
        return result


# =============================================================================
# Training step helper
# =============================================================================
def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice coefficient for boundary detection."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def training_step(model: FusionCropNetV5EDL,
                  batch: dict,
                  edl_loss_fn: EDLLoss,
                  epoch: int,
                  ndvi_channel: int = 6) -> tuple[torch.Tensor, dict]:
    opt = batch['opt']; sar = batch['sar']; dem = batch['dem']
    doy = batch['doy']; y = batch['y']
    cm = batch.get('cloud_mask', None)
    vc = batch.get('valid_count', None)
    wm = batch.get('weight_map', None)

    is_v6 = 'V6' in model.__class__.__name__
    if is_v6:
        alpha, ndvi_pred, consist_loss, (lai_pred, growth_pred, boundary_pred, scene_logits, crop_mix) = model(
            opt, sar, dem, doy, cm, vc, epoch=epoch, return_aux=True)
    else:
        alpha, ndvi_pred, consist_loss = model(
            opt, sar, dem, doy, cm, vc, epoch=epoch)

    edl_loss = edl_loss_fn(alpha, y, epoch)

    if wm is not None:
        probs = (alpha / alpha.sum(1, keepdim=True))
        log_p = torch.log(probs + 1e-8)
        px_ce = F.nll_loss(log_p, y.clamp(0), reduction='none', ignore_index=255)
        weighted_ce = (px_ce * wm)[y != 255].mean()
        current_lam = edl_loss_fn._current_lambda
        orig_ce = F.nll_loss(log_p, y.clamp(0), reduction='mean', ignore_index=255)
        kl_term = (edl_loss - orig_ce) / max(current_lam, 1e-8)
        edl_loss = weighted_ce + current_lam * kl_term

    ndvi_loss = compute_ndvi_loss(opt, ndvi_pred, cm, ndvi_channel)
    aux_w = PhenologyAuxHead.schedule_weight(epoch)

    if is_v6:
        from data.pseudo_labels import generate_lai_pseudo, generate_growth_stage_pseudo, generate_boundary_pseudo
        ndvi_per_sample = opt[:, :, ndvi_channel].mean(dim=(-2, -1)).mean(dim=1)
        lai_target = generate_lai_pseudo(ndvi_per_sample).to(alpha.device)
        lai_loss = F.huber_loss(lai_pred, lai_target, delta=1.0)

        doy_mean = doy.mean(dim=1)
        growth_target = generate_growth_stage_pseudo(doy_mean.unsqueeze(1)).to(alpha.device)
        growth_loss = F.cross_entropy(growth_pred, growth_target)

        boundary_target = generate_boundary_pseudo(dem).to(alpha.device)
        boundary_loss = F.binary_cross_entropy(boundary_pred, boundary_target) +                         (1.0 - dice_score(boundary_pred, boundary_target))

        losses = {
            'crop': edl_loss,
            'ndvi': ndvi_loss * 0.1,
            'lai': lai_loss * 0.3,
            'growth': growth_loss * 0.2,
            'boundary': boundary_loss * 0.1,
        }
        total = model.multi_task_loss(losses) + 0.05 * consist_loss

        return total, {
            'edl_loss': edl_loss.item(), 'ndvi_loss': ndvi_loss.item(),
            'lai_loss': lai_loss.item(), 'growth_loss': growth_loss.item(),
            'boundary_loss': boundary_loss.item(), 'consist': consist_loss.item(),
            'total_loss': total.item(),
        }
    else:
        total = edl_loss + aux_w * ndvi_loss
        return total, {
            'edl_loss': edl_loss.item(), 'ndvi_loss': ndvi_loss.item(),
            'total_loss': total.item(),
        }

# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 7
    m = FusionCropNetV5EDL(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=K,
        feat_dim=512, backbone="resnet50", pretrained=False,
        n_heads=16, win_size=4, n_layers=4,
        edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50).to(dev)
    B, T, H, W = 2, 12, 32, 32
    opt = torch.randn(B, T, 10, H, W).to(dev)
    sar = torch.randn(B, T, 5, H, W).to(dev)
    dem = torch.randn(B, 5, H, W).to(dev)
    doy = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).to(dev)
    cm = (torch.rand(B, T, H, W) < 0.3).to(dev)
    vc = torch.randint(0, T, (B, H, W)).to(dev)
    m.train()
    alpha, ndvi, cl = m(opt, sar, dem, doy, cm, vc, epoch=10)
    print(f"[train] alpha={tuple(alpha.shape)} ndvi={tuple(ndvi.shape)} "
          f"consistency_loss={cl.item():.4f}")
    preds = dirichlet_to_predictions(alpha.detach())
    print(f"[train] probs={tuple(preds['probs'].shape)} "
          f"vacuity={preds['vacuity'].mean().item():.4f} "
          f"dissonance={preds['dissonance'].mean().item():.4f}")
    m.eval()
    result = m.predict_uncertainty(opt, sar, dem, doy, cm, vc, n_passes=5, use_tta=True)
    print(f"[unc] probs={tuple(result['probs'].shape)}")
    print(f" vacuity (aleatoric): {result['vacuity'].mean().item():.4f}")
    print(f" dissonance (epistemic): {result['dissonance'].mean().item():.4f}")
    print(f" class_var (per-class): {result['class_var'].mean().item():.6f}")
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
