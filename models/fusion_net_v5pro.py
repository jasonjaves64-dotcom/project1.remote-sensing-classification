"""
FusionCropNetV5Pro — Enhanced multi-modal crop classification.

Extends V5EDL architecture with:
  - Pluggable backbone (ResNet/ConvNeXt/EfficientNet)
  - Multi-scale cross-modal fusion (mid-level gating)
  - CARAFE upsampling (content-aware, fewer artifacts)
  - Dynamic temporal dropout (curriculum schedule)
  - Adaptive KL annealing (vacuity-error correlation driven)

All shared components from ._base; EDL components from .fusion_net_v5_edl.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ._base import (
    ConvBNGELU, DEMEncoder, OpticalEncoder, SAREncoder,
    TemporalEncoderStream, CrossModalAttention, DEMSpatialConditioner,
    DEMOpticalConditioner, CrossModalAttentionLight,
    LateFusion, Decoder, PhenologyAuxHead, time_average, _BACKBONE_CHANNELS,
)
from .fusion_net_v5_edl import (
    EDLHead, EDLLoss, dirichlet_to_predictions, evidence_level_fusion,
)


class FusionCropNetV5Pro(nn.Module):
    def __init__(self, opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                 feat_dim=512, backbone='resnet50', pretrained=True,
                 n_heads=16, win_size=4, n_layers=4, max_obs=24, n_freqs=4,
                 drop_timestep_p=0.1, edl_dropout_p=0.3,
                 edl_lambda_max=0.5, edl_anneal_ep=50,
                 modality_dropout_p=0.0, use_carafe=True,
                 dynamic_dropout=True, adaptive_kl=True,
                 rs_weights_path: str = None):
        super().__init__()
        self.dynamic_dropout = dynamic_dropout
        self.adaptive_kl = adaptive_kl
        self.drop_timestep_p = drop_timestep_p
        self.modality_dropout_p = modality_dropout_p
        self.feat_dim = feat_dim
        self.num_classes = num_classes

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
        self.dem_opt_cond = DEMOpticalConditioner(feat_dim, dem_ch)
        self.mid_xattn = CrossModalAttentionLight(feat_dim // 2, self.sar_enc.out_channels_list[1])
        self.late_fuse = LateFusion(feat_dim)
        self.decoder = Decoder(feat_dim, self.sar_enc.out_channels_list[:2],
                               n_heads=8, win=win_size, use_carafe=use_carafe)
        self.edl_head = EDLHead(self.decoder.pre_head_ch, num_classes, edl_dropout_p)
        self.edl_loss_fn = EDLLoss(num_classes, edl_lambda_max, edl_anneal_ep, adaptive=adaptive_kl)
        self.pheno_aux = PhenologyAuxHead(feat_dim, aux_weight=0.3)
        self.consistency_proj = nn.Linear(feat_dim, 1)
        self.consistency_target_proj = nn.Linear(feat_dim, 1)

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
            if id(m) in pretrained_modules:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _get_drop_p(self, epoch, total_epochs):
        """Curriculum dropout: low→high→low over training."""
        if not self.dynamic_dropout:
            return self.drop_timestep_p
        progress = epoch / max(total_epochs, 1)
        return 0.05 + 0.15 * math.sin(math.pi * progress)

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

    def _encode(self, opt_seq, sar_seq, dem, doy, cloud_mask, valid_count,
                modality_mask=None):
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

        if use_dem:
            dem_feat = self.dem_enc(dem)
            if self.training:
                dem_feat = dem_feat + 0.01 * torch.randn_like(dem_feat)
        else:
            dem_feat = self.placeholder_dem_feat.expand(B, -1, H, W)

        if self.training:
            sx = torch.randint(-1, 2, (1,)).item()
            sy = torch.randint(-1, 2, (1,)).item()
            opt_seq, sar_seq, dem, cloud_mask, valid_count = self._shift_inputs(
                opt_seq, sar_seq, dem, cloud_mask, valid_count, sx, sy)

        dem_feat = F.interpolate(dem_feat, (H, W), mode='bilinear', align_corners=False)

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

        # V5Pro: DEM conditions optical features (terrain-aware optical encoding)
        if use_opt and use_dem:
            dem_tiled_BT = dem_feat.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, -1, H, W)
            opt_feat, opt_p2 = self.dem_opt_cond(opt_feat, opt_p2, dem_tiled_BT)

        if use_sar:
            sar_flat = sar_seq.view(B * T, sar_seq.shape[2], H, W)
            dem_tiled = dem_feat.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, -1, H, W)
            sar_s1, sar_s2, sar_s3 = self.sar_enc(sar_flat, dem_feat=dem_tiled)
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

        # V5Pro: dynamic temporal dropout
        eff_drop = self._get_drop_p(getattr(self, '_current_epoch', 0),
                                     getattr(self, '_total_epochs', 80))
        if self.training and eff_drop > 0:
            drop = torch.rand(B * H2 * W2, T, device=opt_feat.device) < eff_drop
            cm_px = (cm_px | drop) if cm_px is not None else drop

        vc_px = None
        if valid_count is not None:
            vc = F.adaptive_avg_pool2d(valid_count.float().unsqueeze(1), (H2, W2)).squeeze(1).long()
            vc_px = vc.view(B * H2 * W2)
        doy_px = doy.unsqueeze(1).unsqueeze(1).expand(B, H2, W2, T).reshape(B * H2 * W2, T)

        opt_ts = self._to_pixel_seq(opt_feat, B, T, H2, W2, D)
        sar_ts = self._to_pixel_seq(sar_s3, B, T, H2, W2, D)
        opt_g, opt_seq_out = self.opt_temporal(opt_ts, doy_px, cloud_mask=cm_px, valid_count=vc_px)
        sar_g, _ = self.sar_temporal(sar_ts, doy_px, cloud_mask=None, valid_count=vc_px, fallback_feat=opt_g)
        opt_global = opt_g.view(B, H2, W2, D).permute(0, 3, 1, 2)
        sar_global = sar_g.view(B, H2, W2, D).permute(0, 3, 1, 2)

        if use_opt and use_sar:
            xm_feat = self.cross_modal(opt_global, sar_global)
        elif use_opt:
            xm_feat = opt_global
        else:
            xm_feat = sar_global

        if use_dem:
            xm_feat = self.dem_cond(xm_feat, dem_feat)

        # V5Pro: mid-level cross-modal attention (H/2 x W/2 scale)
        opt_p2a = time_average(opt_p2, B, T)
        sar_s2a = time_average(sar_s2, B, T)
        mid_fused = self.mid_xattn(opt_p2a, sar_s2a)

        xm_f = xm_feat.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)
        opt_f = opt_global.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)
        sar_f = sar_global.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)

        if use_opt and use_sar:
            final = self.late_fuse(xm_f, opt_f, sar_f)
        else:
            final = xm_f

        final = final.view(B, H2, W2, D).permute(0, 3, 1, 2)

        sar_s1a = time_average(sar_s1, B, T)
        pre_head = self.decoder(final,
                                opt_skips=(opt_p2a,),
                                sar_skips=(sar_s1a, mid_fused),
                                target_size=(H, W))
        return pre_head, ndvi_pred, opt_seq_out, cm_px, B, T, H2, W2, D, H, W, opt_f

    def forward(self, opt_seq, sar_seq, dem, doy,
                cloud_mask=None, valid_count=None, epoch=0, total_epochs=80,
                modality_mask=None, spear_r=None):
        self._current_epoch = epoch
        self._total_epochs = total_epochs

        (pre_head, ndvi_pred, opt_seq_out,
         cm_px, B, T, H2, W2, D, H, W, opt_f) = self._encode(
             opt_seq, sar_seq, dem, doy, cloud_mask, valid_count,
             modality_mask=modality_mask)

        alpha = self.edl_head(pre_head)

        if self.training:
            if cm_px is not None:
                certainty = torch.sigmoid(self.consistency_proj(opt_seq_out)).squeeze(-1)
                unmasked_mask = ~cm_px
                if unmasked_mask.any():
                    target_certainty = torch.sigmoid(
                        self.consistency_target_proj(opt_seq_out)).squeeze(-1)
                    consistency_loss = F.mse_loss(
                        certainty[unmasked_mask], target_certainty[unmasked_mask])
                else:
                    consistency_loss = alpha.sum() * 0.0
            else:
                consistency_loss = alpha.sum() * 0.0
            return alpha, ndvi_pred, consistency_loss
        return alpha

    def predict_uncertainty(self, opt_seq, sar_seq, dem, doy,
                            cloud_mask=None, valid_count=None,
                            n_passes=10, use_tta=True,
                            modality_mask=None):
        self.eval()
        for m in self.edl_head.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()

        alpha_list = []
        inputs_orig = (opt_seq, sar_seq, dem, doy, cloud_mask, valid_count)

        def _single_pass(args):
            pre_head, *_ = self._encode(*args, modality_mask=modality_mask)
            return self.edl_head(pre_head)

        with torch.no_grad():
            for _ in range(n_passes):
                alpha_list.append(_single_pass(inputs_orig))
            if use_tta:
                opt_f = opt_seq.flip(-1); sar_f = sar_seq.flip(-1); dem_f = dem.flip(-1)
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


def training_step(model: FusionCropNetV5Pro, batch: dict,
                  edl_loss_fn: EDLLoss, epoch: int, total_epochs: int = 80,
                  ndvi_channel: int = 6, spear_r: float = None):
    opt = batch['opt']; sar = batch['sar']; dem = batch['dem']
    doy = batch['doy']; y = batch['y']
    cm = batch.get('cloud_mask', None); vc = batch.get('valid_count', None)
    wm = batch.get('weight_map', None)
    alpha, ndvi_pred, consist_loss = model(
        opt, sar, dem, doy, cm, vc, epoch=epoch, total_epochs=total_epochs, spear_r=spear_r)

    edl_loss = edl_loss_fn(alpha, y, epoch, spear_r=spear_r)

    if wm is not None:
        probs = (alpha / alpha.sum(1, keepdim=True))
        log_p = torch.log(probs + 1e-8)
        px_ce = F.nll_loss(log_p, y.clamp(0), reduction='none', ignore_index=255)
        weighted_ce = (px_ce * wm)[y != 255].mean()
        lam = edl_loss_fn._current_lambda
        edl_loss = weighted_ce + lam * (edl_loss - F.nll_loss(
            log_p, y.clamp(0), reduction='mean', ignore_index=255))

    B, T = opt.shape[:2]
    ndvi_tgt = opt[:, :, ndvi_channel].mean(dim=(-2, -1)).reshape(B * T)
    cm_bt = cm.view(B * T, -1).any(-1) if cm is not None else None
    ndvi_loss = PhenologyAuxHead.compute_loss(ndvi_pred, ndvi_tgt, cm_bt)
    aux_w = PhenologyAuxHead.schedule_weight(epoch)
    total = edl_loss + aux_w * ndvi_loss + 0.05 * consist_loss
    return total, {
        'edl_loss': edl_loss.item(), 'ndvi_loss': ndvi_loss.item(),
        'consist': consist_loss.item(), 'aux_weight': aux_w,
    }
