"""
FusionCropNetV5 — Multi-modal crop classification model.
All shared components imported from ._base (single canonical source).

Known bugs fixed:
  BUG1: consistency_proj created nn.Linear inside forward → moved to __init__
  BUG2: CrossModalAttention iterated .children() → direct Sequential call (in _base)
  BUG3: SpatialRefinement full self-attn O(n²) → windowed attention (in _base)
  BUG4: DEM shift unsynced → _shift_inputs() syncs all modalities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import (
    ConvBNGELU, SEBlock, FiLM, IRB,
    DEMEncoder, FPN, OpticalEncoder, SAREncoder,
    FourierDOYEncoding, ObsQualityToken, TemporalEncoderStream,
    CrossModalAttention, DEMSpatialConditioner, LateFusion,
    SpatialRefinement, PhenologyAuxHead, Decoder,
    time_average,
)


class FusionCropNetV5(nn.Module):
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
                 drop_timestep_p: float = 0.1):
        super().__init__()
        dem_ch = 128
        self.dem_enc = DEMEncoder(dem_ch_in, dem_ch)
        self.opt_enc = OpticalEncoder(opt_ch, feat_dim, backbone, pretrained)
        self.pheno_aux = PhenologyAuxHead(feat_dim, aux_weight=0.3)
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
        self.cls_head = nn.Sequential(
            ConvBNGELU(64, 64), nn.Dropout2d(0.3),
            ConvBNGELU(64, 64), nn.Dropout2d(0.3),
            nn.Conv2d(64, num_classes, 1))
        self.consistency_proj = nn.Linear(feat_dim, 1)
        self.drop_timestep_p = drop_timestep_p
        self._init_weights()

    def _init_weights(self):
        pretrained_id = id(self.opt_enc.backbone)
        for m in self.modules():
            if id(m) == pretrained_id:
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
        """BUG4 fix: Sync shift all modalities with DEM shift."""
        if sx == 0 and sy == 0:
            return opt_seq, sar_seq, dem, cloud_mask, valid_count
        pad_l = abs(sx) if sx < 0 else 0
        pad_r = abs(sx) if sx > 0 else 0
        pad_t = abs(sy) if sy < 0 else 0
        pad_b = abs(sy) if sy > 0 else 0
        pad_hw = (pad_l, pad_r, pad_t, pad_b)
        opt_seq = F.pad(opt_seq, pad_hw if opt_seq.dim() == 4 else pad_hw + (0, 0), mode='replicate')
        sar_seq = F.pad(sar_seq, pad_hw if sar_seq.dim() == 4 else pad_hw + (0, 0), mode='replicate')
        dem = F.pad(dem, pad_hw, mode='replicate')
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

    def forward(self, opt_seq, sar_seq, dem, doy, cloud_mask=None, valid_count=None):
        B, T, Co, H, W = opt_seq.shape

        if self.training:
            shift_x = torch.randint(-1, 2, (1,)).item()
            shift_y = torch.randint(-1, 2, (1,)).item()
            opt_seq, sar_seq, dem, cloud_mask, valid_count = self._shift_inputs(
                opt_seq, sar_seq, dem, cloud_mask, valid_count, shift_x, shift_y)

        dem_feat = self.dem_enc(dem)
        if self.training:
            dem_feat = dem_feat + 0.01 * torch.randn_like(dem_feat)
        dem_feat = F.interpolate(dem_feat, (H, W), mode='bilinear', align_corners=False)

        opt_flat = opt_seq.view(B * T, Co, H, W)
        opt_feat, opt_p2, opt_p3 = self.opt_enc(opt_flat)
        ndvi_pred = self.pheno_aux(opt_feat)
        _, D, H2, W2 = opt_feat.shape

        sar_flat = sar_seq.view(B * T, sar_seq.shape[2], H, W)
        dem_tiled = dem_feat.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, -1, H, W)
        sar_s1, sar_s2, sar_s3 = self.sar_enc(sar_flat, dem_feat=dem_tiled)

        opt_ts = self._to_pixel_seq(opt_feat, B, T, H2, W2, D)
        sar_ts = self._to_pixel_seq(sar_s3, B, T, H2, W2, D)

        if cloud_mask is not None:
            cm_down = F.adaptive_avg_pool2d(cloud_mask.view(B * T, 1, H, W).float(), (H2, W2))
            cm_px = ((cm_down > 0.5).squeeze(1).view(B, T, H2 * W2)
                      .permute(0, 2, 1).reshape(B * H2 * W2, T))
        else:
            cm_px = None

        vc_px = None
        if valid_count is not None:
            vc = F.adaptive_avg_pool2d(valid_count.float().unsqueeze(1), (H2, W2)).squeeze(1).long()
            vc_px = vc.view(B * H2 * W2)

        doy_px = doy.unsqueeze(1).unsqueeze(1).expand(B, H2, W2, T).reshape(B * H2 * W2, T)

        if self.training and cm_px is not None and self.drop_timestep_p > 0:
            drop_mask = torch.rand_like(cm_px.float()) < self.drop_timestep_p
            cm_px = cm_px | drop_mask
        elif self.training and cm_px is None and self.drop_timestep_p > 0:
            drop_mask = torch.rand(B * H2 * W2, T, device=opt_ts.device) < self.drop_timestep_p
            cm_px = drop_mask

        opt_g, opt_seq_out = self.opt_temporal(opt_ts, doy_px, cloud_mask=cm_px, valid_count=vc_px)
        sar_g, _ = self.sar_temporal(sar_ts, doy_px, cloud_mask=None, valid_count=vc_px, fallback_feat=opt_g)

        consistency_loss = None
        if self.training and cm_px is not None:
            certainty = torch.sigmoid(self.consistency_proj(opt_seq_out)).squeeze(-1)
            entropy = -certainty * torch.log(certainty + 1e-6)
            cloud_weight = cm_px.float()
            consistency_loss = ((1 - cloud_weight) * entropy).mean()

        opt_global = opt_g.view(B, H2, W2, D).permute(0, 3, 1, 2)
        sar_global = sar_g.view(B, H2, W2, D).permute(0, 3, 1, 2)

        xm_feat = self.cross_modal(opt_global, sar_global)
        xm_feat = self.dem_cond(xm_feat, dem_feat)

        xm_flat = xm_feat.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)
        opt_flat_g = opt_global.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)
        sar_flat_g = sar_global.permute(0, 2, 3, 1).reshape(B * H2 * W2, D)
        final_flat = self.late_fuse(xm_flat, opt_flat_g, sar_flat_g)
        final_map = final_flat.view(B, H2, W2, D).permute(0, 3, 1, 2)

        opt_p2_avg = time_average(opt_p2, B, T)
        sar_s1_avg = time_average(sar_s1, B, T)
        sar_s2_avg = time_average(sar_s2, B, T)

        pre_head = self.decoder(final_map,
                                opt_skips=(opt_p2_avg,),
                                sar_skips=(sar_s1_avg, sar_s2_avg),
                                target_size=(H, W))
        logits = self.cls_head(pre_head)

        if self.training:
            return logits, ndvi_pred, consistency_loss
        return logits
