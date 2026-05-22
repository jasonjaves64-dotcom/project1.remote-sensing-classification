import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from einops import rearrange

from models.dem_encoder import DEMEncoder, ThreeWayFusion
from models.temporal import TemporalEncoderStream, LateFusion, FourierDOYEncoding
from models.heads import SpatialRefinement, UncertaintyHead, PhenologyAuxHead

class ConvBNGELU(nn.Module):
    def __init__(self, i, o, k=3, p=1, g=1):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, k, 1, p, groups=g, bias=False),
            nn.BatchNorm2d(o),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.b(x)

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // r, 4)),
            nn.ReLU(),
            nn.Linear(max(ch // r, 4), ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)

class IRB(nn.Module):
    def __init__(self, i, o, e=4):
        super().__init__()
        m = i * e
        self.c = nn.Sequential(
            nn.Conv2d(i, m, 1, bias=False),
            nn.BatchNorm2d(m),
            nn.GELU(),
            nn.Conv2d(m, m, 3, 1, 1, groups=m, bias=False),
            nn.BatchNorm2d(m),
            nn.GELU(),
            SEBlock(m),
            nn.Conv2d(m, o, 1, bias=False),
            nn.BatchNorm2d(o)
        )
        self.s = nn.Conv2d(i, o, 1, bias=False) if i != o else nn.Identity()
    
    def forward(self, x):
        return F.gelu(self.c(x) + self.s(x))

class SAREncoder(nn.Module):
    def __init__(self, ic=5, bc=32, oc=512):
        super().__init__()
        self.stem = ConvBNGELU(ic, bc)
        self.s1 = nn.Sequential(IRB(bc, bc*2), IRB(bc*2, bc*2))
        self.d1 = nn.Conv2d(bc*2, bc*2, 3, 2, 1)
        self.s2 = nn.Sequential(IRB(bc*2, bc*4), IRB(bc*4, bc*4))
        self.d2 = nn.Conv2d(bc*4, bc*4, 3, 2, 1)
        self.s3 = nn.Sequential(IRB(bc*4, oc), IRB(oc, oc))
        self.out_channels_list = [bc*2, bc*4, oc]
    
    def forward(self, x):
        x = self.stem(x)
        s1 = self.s1(x)
        s2 = self.s2(self.d1(s1))
        s3 = self.s3(self.d2(s2))
        return s1, s2, s3

_CM = {"resnet18": [64, 128, 256, 512], "resnet50": [256, 512, 1024, 2048]}

class FPN(nn.Module):
    def __init__(self, chs, out):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, out, 1) for c in chs])
        self.outs = nn.ModuleList([ConvBNGELU(out, out) for _ in chs])
    
    def forward(self, feats):
        t = feats[0].shape[-2:]
        m = None
        fps = []
        for i, f in enumerate(feats):
            l = self.lats[i](f)
            if m is not None:
                l = l + F.interpolate(m, l.shape[-2:], 
                                     mode='bilinear', align_corners=False)
            m = self.outs[i](l)
            fps.append(m)
        main = sum(F.interpolate(f, t, mode='bilinear', align_corners=False) for f in fps)
        p2 = F.interpolate(fps[0], scale_factor=2, mode='bilinear', align_corners=False)
        return main, p2, fps[0]

class OpticalEncoder(nn.Module):
    def __init__(self, ic, fd, bb="resnet50", pt=True):
        super().__init__()
        self.bb = timm.create_model(bb, pretrained=pt, features_only=True, out_indices=(1, 2, 3, 4))
        oc = self.bb.conv1
        w = oc.weight.data
        nw = w.mean(1, keepdim=True).repeat(1, ic, 1, 1)
        nc = nn.Conv2d(ic, w.shape[0], oc.kernel_size, oc.stride, oc.padding, bias=False)
        nc.weight.data = nw
        self.bb.conv1 = nc
        self.fpn = FPN(_CM.get(bb, [64, 128, 256, 512]), fd)
        self.sp2 = nn.Conv2d(fd, fd // 2, 1)
        self.sp3 = nn.Conv2d(fd, fd // 2, 1)
    
    def forward(self, x):
        m, p2, p3 = self.fpn(self.bb(x))
        return m, self.sp2(p2), self.sp3(p3)

class SWBlock(nn.Module):
    def __init__(self, dim, win=4, nh=8, shift=False):
        super().__init__()
        self.win = win
        self.sh = win // 2 if shift else 0
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nh, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
    
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

class CrossModalAttention(nn.Module):
    def __init__(self, ch, nh=16, win=4):
        super().__init__()
        self.nh = nh
        self.scale = (ch // nh) ** -0.5
        self.qo = nn.Conv2d(ch, ch, 1)
        self.ks = nn.Conv2d(ch, ch, 1)
        self.vs = nn.Conv2d(ch, ch, 1)
        self.qs = nn.Conv2d(ch, ch, 1)
        self.ko = nn.Conv2d(ch, ch, 1)
        self.vo = nn.Conv2d(ch, ch, 1)
        self.sw_o2s = nn.ModuleList([SWBlock(ch, win, nh, i % 2 == 1) for i in range(2)])
        self.sw_s2o = nn.ModuleList([SWBlock(ch, win, nh, i % 2 == 1) for i in range(2)])
        self.gate = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.Sigmoid())
        self.proj = nn.Sequential(ConvBNGELU(ch * 2, ch), SEBlock(ch))
        self.norm = nn.GroupNorm(32, ch)
    
    def _xattn(self, qf, kvf, qp, kp, vp):
        B, C, H, W = qf.shape
        h, d = self.nh, C // self.nh
        Q = qp(qf).view(B, h, d, -1).permute(0, 1, 3, 2)
        K = kp(kvf).view(B, h, d, -1).permute(0, 1, 3, 2)
        V = vp(kvf).view(B, h, d, -1).permute(0, 1, 3, 2)
        a = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        return (a @ V).permute(0, 1, 3, 2).reshape(B, C, H, W)
    
    def forward(self, opt, sar):
        if sar.shape[-2:] != opt.shape[-2:]:
            sar = F.interpolate(sar, opt.shape[-2:], mode='bilinear', align_corners=False)
        
        o2s = self._xattn(opt, sar, self.qo, self.ks, self.vs)
        for b in self.sw_o2s:
            o2s = b(o2s)
        
        s2o = self._xattn(sar, opt, self.qs, self.ko, self.vo)
        for b in self.sw_s2o:
            s2o = b(s2o)
        
        g = self.gate(torch.cat([o2s, s2o], 1))
        return self.norm(self.proj(torch.cat([g * o2s + (1 - g) * s2o, opt], 1)) + opt)

class Decoder(nn.Module):
    def __init__(self, fd, sc, nc, nh=8, win=4):
        super().__init__()
        od = fd // 2
        self.u1 = nn.Sequential(
            nn.ConvTranspose2d(fd, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.m1 = ConvBNGELU(128 + od + sc[1], 128)
        self.sr = SpatialRefinement(128, nh, win)
        self.u2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.m2 = ConvBNGELU(64 + od + sc[0], 64)
    
    def _a(self, f, hw):
        return F.interpolate(f, hw, mode='bilinear', align_corners=False) if f.shape[-2:] != hw else f
    
    def forward(self, x, os, ss, tsz):
        o2, _ = os
        s1, s2 = ss
        
        x = self.u1(x)
        x = self.m1(torch.cat([x, self._a(o2, x.shape[-2:]), self._a(s2, x.shape[-2:])], 1))
        x = self.sr(x)
        x = self.u2(x)
        x = self.m2(torch.cat([x, self._a(o2, x.shape[-2:]), self._a(s1, x.shape[-2:])], 1))
        return self._a(x, tsz)

class FusionCropNetV4(nn.Module):
    def __init__(self, oc=10, sc=5, dc=5, nc=7, fd=512,
                 bb="resnet50", pt=True, nh=16, win=4, nl=4,
                 max_obs=24, n_freqs=4, mc_dropout=0.3,
                 use_checkpointing: bool = False):
        super().__init__()
        
        self.opt_enc = OpticalEncoder(oc, fd, bb, pt)
        self.sar_enc = SAREncoder(sc, 32, fd)
        self.dem_enc = DEMEncoder(dc, 128)
        self.dem_proj = nn.Linear(128, fd)  # Project DEM features to temporal dim

        self.xmodal = CrossModalAttention(fd, nh, win)
        
        self.opt_temporal = TemporalEncoderStream(fd, 8, nl, max_obs=max_obs, n_freqs=n_freqs,
                                                  use_checkpointing=use_checkpointing)
        self.sar_temporal = TemporalEncoderStream(fd, 8, nl, max_obs=max_obs, n_freqs=n_freqs,
                                                  use_checkpointing=use_checkpointing)
        self.late_fusion = LateFusion(fd)
        
        self.decoder = Decoder(fd, self.sar_enc.out_channels_list[:2], nc, 8, win)
        
        self.cls_head = UncertaintyHead(64, nc, mc_dropout)
        self.aux_head = PhenologyAuxHead(fd)
        
        self._init()
    
    def _init(self):
        skip = {self.opt_enc.bb}
        for m in self.modules():
            if any(m is s for s in skip):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _tavg(self, f, B, T):
        _, C, h, w = f.shape
        return f.view(B, T, C, h, w).mean(1)
    
    def forward(self, opt, sar, dem, doy, cloud_mask=None, valid_count=None):
        B, T, Co, H, W = opt.shape
        of = opt.view(B * T, Co, H, W)
        sf = sar.view(B * T, sar.shape[2], H, W)
        
        om, os2, os3 = self.opt_enc(of)
        ss1, ss2, ss3 = self.sar_enc(sf)
        dem_feat = self.dem_enc(dem)
        _, D, H2, W2 = om.shape
        
        ss3a = F.interpolate(ss3, (H2, W2), mode='bilinear', align_corners=False)
        fused = self.xmodal(om, ss3a)
        
        if cloud_mask is not None:
            cmd = F.adaptive_avg_pool2d(cloud_mask.view(B * T, 1, H, W).float(), (H2, W2))
            cmd = (cmd > .5).squeeze(1).view(B, T, H2 * W2).permute(0, 2, 1).reshape(B * H2 * W2, T)
        else:
            cmd = None
        
        if valid_count is not None:
            vc = F.adaptive_avg_pool2d(valid_count.float().unsqueeze(1), (H2, W2)).squeeze(1).long()
            vc_flat = vc.view(B * H2 * W2)
        else:
            vc_flat = None
        
        def tseq(f):
            return f.view(B, T, D, H2, W2).permute(0, 3, 4, 1, 2).reshape(B * H2 * W2, T, D)
        
        de = doy.unsqueeze(1).unsqueeze(1).expand(B, H2, W2, T).reshape(B * H2 * W2, T)
        
        dem_cond = F.interpolate(dem_feat, (H2, W2), mode='bilinear', align_corners=False)
        dem_cond = dem_cond.flatten(2).transpose(1, 2)  # (B, H2*W2, 128)
        dem_cond = self.dem_proj(dem_cond).reshape(B * H2 * W2, D)  # project to (B*H2*W2, D)
        
        opt_g, opt_seq = self.opt_temporal(tseq(fused), de, cmd, vc_flat, dem_cond=dem_cond)
        sar_g, sar_seq = self.sar_temporal(tseq(ss3a), de, None, vc_flat, fallback_feat=opt_g, dem_cond=dem_cond)
        fused_g = self.late_fusion(opt_g, sar_g)
        fused_map = fused_g.view(B, H2, W2, D).permute(0, 3, 1, 2)
        
        pre_cls = self.decoder(
            fused_map,
            (self._tavg(os2, B, T), self._tavg(os3, B, T)),
            (self._tavg(ss1, B, T), self._tavg(ss2, B, T)),
            (H, W)
        )
        
        logits = self.cls_head(pre_cls)
        
        if self.training:
            ndvi_aux = self.aux_head(opt_seq[:, :T, :])
            return logits, ndvi_aux
        return logits
    
    def predict_with_uncertainty(self, opt, sar, dem, doy,
                                  cloud_mask=None, valid_count=None,
                                  mc_samples=20):
        self.eval()
        self.cls_head.train()
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self.forward(opt, sar, dem, doy, cloud_mask, valid_count)
                preds.append(torch.softmax(logits, dim=1))
        self.cls_head.eval()
        
        preds = torch.stack(preds)
        mean = preds.mean(0)
        unc = -(mean * (mean + 1e-6).log()).sum(1)
        return mean, unc

def training_step(model, batch, criterion, epoch, ndvi_channel=6):
    opt = batch['opt']
    sar = batch['sar']
    dem = batch['dem']
    doy = batch['doy']
    y = batch['y']
    wm = batch.get('weight_map', None)
    cm = batch.get('cloud_mask', None)
    vc = batch.get('valid_count', None)
    
    logits, ndvi_aux = model(opt, sar, dem, doy, cm, vc)
    
    seg_loss = criterion(logits, y, wm)
    
    B, T, _, H, W = opt.shape
    H2, W2 = logits.shape[-2:]
    
    ndvi_tgt = F.adaptive_avg_pool2d(
        opt[:, :, ndvi_channel, :, :].view(B * T, 1, H, W),
        (H2, W2)
    ).squeeze(1).view(B, T, H2 * W2).permute(0, 2, 1).reshape(B * H2 * W2, T)
    
    if cm is not None:
        cmd = F.adaptive_avg_pool2d(cm.view(B * T, 1, H, W).float(), (H2, W2))
        cmd = (cmd > .5).squeeze(1).view(B, T, H2 * W2).permute(0, 2, 1).reshape(B * H2 * W2, T)
    else:
        cmd = None
    
    aux_loss = PhenologyAuxHead.loss(ndvi_aux, ndvi_tgt, cmd)
    aux_w = PhenologyAuxHead.aux_weight(epoch)
    
    return seg_loss + aux_w * aux_loss, {
        'seg_loss': seg_loss.item(),
        'aux_loss': aux_loss.item(),
        'aux_weight': aux_w
    }