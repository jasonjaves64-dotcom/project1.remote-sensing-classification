import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w

class FeaturePyramidNeck(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.outputs = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels) for _ in in_channels_list
        ])
        self.in_channels_list = in_channels_list
    
    def forward(self, features):
        target_size = features[0].shape[-2:]
        merged = None
        intermediate = []
        for i, feat in enumerate(features):
            lat = self.laterals[i](feat)
            if merged is not None:
                lat = lat + F.interpolate(merged, size=lat.shape[-2:],
                                        mode="bilinear", align_corners=False)
            merged = self.outputs[i](lat)
            intermediate.append(merged)
        final = F.interpolate(merged, size=target_size,
                            mode="bilinear", align_corners=False)
        return final, intermediate

class OpticalEncoder(nn.Module):
    def __init__(self, in_channels=10, out_channels=512,
                 backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        
        original_conv = self.backbone.conv1 if hasattr(self.backbone, 'conv1') \
            else list(self.backbone.children())[0]
        orig_weight = original_conv.weight.data
        new_weight = orig_weight.mean(dim=1, keepdim=True) \
            .repeat(1, in_channels, 1, 1)
        
        new_conv = nn.Conv2d(
            in_channels, orig_weight.shape[0],
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        new_conv.weight.data = new_weight
        self.backbone.conv1 = new_conv
        
        backbone_channels = self._get_backbone_channels(backbone)
        self.fpn = FeaturePyramidNeck(backbone_channels, out_channels)
        
        self.channel_attention = SEBlock(out_channels)
    
    def _get_backbone_channels(self, name):
        channel_map = {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
            "efficientnet_b0": [24, 40, 112, 320],
            "efficientnet_b4": [24, 48, 120, 336],
            "vit_base_patch16_224": [64, 128, 256, 512],
        }
        return channel_map.get(name, [256, 512, 1024, 2048])
    
    def forward(self, x):
        features = self.backbone(x)
        fpn_out, intermediates = self.fpn(features)
        return self.channel_attention(fpn_out), intermediates

class SAREncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 64, k=3, p=1),
            ConvBNReLU(64, 64, k=3, p=1, groups=64),
            ConvBNReLU(64, 128, k=1, p=0),
            ConvBNReLU(128, 128, k=3, p=1, groups=128),
            ConvBNReLU(128, 256, k=1, p=0),
            ConvBNReLU(256, 256, k=3, p=1, groups=256),
            ConvBNReLU(256, 512, k=1, p=0),
            ConvBNReLU(512, 512, k=3, p=1, groups=512),
            ConvBNReLU(512, out_channels, k=1, p=0),
            SEBlock(out_channels),
        )
        self.out_channels = out_channels
    
    def forward(self, x):
        out = self.encoder(x)
        return out, out

class WindowAttention(nn.Module):
    def __init__(self, dim, n_heads, window_size, shift=False):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift = shift
        self.scale = (dim // n_heads) ** -0.5
        
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.norm = GroupNorm(32, dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        if self.shift:
            x = F.pad(x, (self.window_size//2, self.window_size//2, 
                          self.window_size//2, self.window_size//2))
            pad_l = pad_t = self.window_size//2
            pad_r += self.window_size//2
            pad_b += self.window_size//2
        
        x_padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        B, C, Hp, Wp = x_padded.shape
        
        num_windows = (Hp // self.window_size) * (Wp // self.window_size)
        
        qkv = self.qkv(x_padded)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.n_heads, C//self.n_heads, Hp, Wp)
        k = k.view(B, self.n_heads, C//self.n_heads, Hp, Wp)
        v = v.view(B, self.n_heads, C//self.n_heads, Hp, Wp)
        
        q = self.window_partition(q, self.window_size)
        k = self.window_partition(k, self.window_size)
        v = self.window_partition(v, self.window_size)
        
        q = q.permute(0, 1, 2, 4, 3)
        k = k.permute(0, 1, 2, 4, 3)
        v = v.permute(0, 1, 2, 4, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 2, 4, 3)
        
        out = self.window_reverse(out, self.window_size, Hp, Wp)
        out = self.out_proj(out)
        
        if self.shift:
            out = out[:, :, self.window_size//2:-self.window_size//2, 
                      self.window_size//2:-self.window_size//2]
        
        return self.norm(out + x)
    
    def window_partition(self, x, window_size):
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        x = x.view(B, -1, H//window_size, window_size, W//window_size, window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, -1, x.shape[3], x.shape[4]*x.shape[5])
        return x
    
    def window_reverse(self, x, window_size, H, W):
        B, _, C, _ = x.shape
        x = x.view(B, H//window_size, W//window_size, C, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, H, W)
        return x

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, channels=512, n_heads=16):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.scale = (channels // n_heads) ** -0.5
        
        self.q_opt = nn.Conv2d(channels, channels, 1)
        self.k_sar = nn.Conv2d(channels, channels, 1)
        self.v_sar = nn.Conv2d(channels, channels, 1)
        
        self.q_sar = nn.Conv2d(channels, channels, 1)
        self.k_opt = nn.Conv2d(channels, channels, 1)
        self.v_opt = nn.Conv2d(channels, channels, 1)
        
        self.out_proj = nn.Sequential(
            ConvBNReLU(channels * 2, channels),
            SEBlock(channels)
        )
        self.norm = GroupNorm(32, channels)
    
    def _spatial_attention(self, q_feat, k_feat, v_feat):
        B, C, H, W = q_feat.shape
        h = self.n_heads
        d = C // h
        
        q = self.q_opt(q_feat).view(B, h, d, H*W).permute(0, 1, 3, 2)
        k = self.k_sar(k_feat).view(B, h, d, H*W).permute(0, 1, 3, 2)
        v = self.v_sar(v_feat).view(B, h, d, H*W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        return out.permute(0, 1, 3, 2).reshape(B, C, H, W)
    
    def forward(self, opt_feat, sar_feat):
        if sar_feat.shape[-2:] != opt_feat.shape[-2:]:
            sar_feat = F.interpolate(sar_feat, size=opt_feat.shape[-2:],
                                    mode="bilinear", align_corners=False)
        
        opt_enhanced = self._spatial_attention(opt_feat, sar_feat, sar_feat)
        sar_enhanced = self._spatial_attention(sar_feat, opt_feat, opt_feat)
        
        fused = self.out_proj(
            torch.cat([opt_enhanced, sar_enhanced], dim=1)
        )
        return self.norm(fused + opt_feat)

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x, doy=None):
        x = x + self.pe[:x.size(0)]
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, d_model=512, n_heads=16, num_layers=4, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.fallback_embed = nn.Parameter(torch.randn(1, d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.spatial_context_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1)
        )
    
    def forward(self, x, cloud_mask=None, spatial_context=None):
        B, T, D = x.shape
        full_mask = None
        
        if cloud_mask is not None:
            full_mask = cloud_mask.all(dim=1)
            if full_mask.any():
                fallback = self.fallback_embed.unsqueeze(0).expand(B, T, D)
                
                if spatial_context is not None and full_mask.any():
                    fallback_context = self.spatial_context_conv(spatial_context)
                    fallback_context = fallback_context.flatten(2).transpose(1, 2)
                    full_mask_idx = full_mask.nonzero().squeeze(1)
                    if len(full_mask_idx) > 0:
                        fallback[full_mask_idx] = fallback_context[full_mask_idx]
                
                x = torch.where(full_mask.unsqueeze(1).unsqueeze(2), fallback, x)
                cloud_mask = cloud_mask.clone()
                cloud_mask[full_mask] = False
        
        x = self.pos_enc(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        mask = None
        if cloud_mask is not None:
            mask = cloud_mask
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        if cloud_mask is not None:
            mask_expanded = (1 - cloud_mask.float()).unsqueeze(-1)
            x_weighted = x * mask_expanded
            x_sum = x_weighted.sum(dim=1)
            mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-6)
            out = x_sum / mask_sum
        else:
            out = x.mean(dim=1)
        
        return self.norm(out), x, full_mask

class SpatialRefinement(nn.Module):
    def __init__(self, in_channels, window_size=8, n_heads=16):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.window_attn1 = WindowAttention(in_channels, n_heads, window_size, shift=False)
        self.window_attn2 = WindowAttention(in_channels, n_heads, window_size, shift=True)
        self.window_attn3 = WindowAttention(in_channels, n_heads, window_size, shift=False)
        
        self.residual_conv = ConvBNReLU(in_channels, in_channels, k=3, p=1, groups=in_channels)
        
        self.norm = GroupNorm(32, in_channels)
    
    def forward(self, x, valid_count=None):
        attn_gate = self.attention_gate(x)
        x = x * attn_gate
        
        x = self.window_attn1(x)
        x = self.window_attn2(x)
        x = self.window_attn3(x)
        
        residual = self.residual_conv(x)
        x = self.norm(x + residual)
        
        if valid_count is not None:
            count_embed = self._get_count_embedding(valid_count, x.shape[1])
            x = x + count_embed
        
        return x
    
    def _get_count_embedding(self, count, channels):
        B, H, W = count.shape
        count_norm = count.float() / count.max()
        count_embed = count_norm.view(B, 1, H, W).expand(B, channels, H, W)
        return count_embed * 0.1

class DecoderWithSkipConnections(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, in_channels//2, k=3, p=1)
        self.conv2 = ConvBNReLU(in_channels//2, in_channels//4, k=3, p=1)
        
        self.sar_skip_conv = ConvBNReLU(in_channels//2, in_channels//4, k=1, p=0)
        self.opt_skip_conv = ConvBNReLU(in_channels//2, in_channels//4, k=1, p=0)
        
        self.fusion_conv = ConvBNReLU(in_channels//2, in_channels//4, k=3, p=1)
        
        self.classifier = nn.Sequential(
            ConvBNReLU(in_channels//4, 64, k=3, p=1),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x, sar_skip=None, opt_skip=None):
        x = self.conv1(x)
        
        skip_features = []
        if sar_skip is not None:
            sar_skip = F.interpolate(sar_skip, size=x.shape[-2:], mode="bilinear")
            skip_features.append(self.sar_skip_conv(sar_skip))
        
        if opt_skip is not None:
            opt_skip = F.interpolate(opt_skip, size=x.shape[-2:], mode="bilinear")
            skip_features.append(self.opt_skip_conv(opt_skip))
        
        if skip_features:
            x = self.fusion_conv(torch.cat([x] + skip_features, dim=1))
        
        x = self.conv2(x)
        return self.classifier(x)

class ObservationEmbedding(nn.Module):
    def __init__(self, max_count, feat_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_count + 1, feat_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, valid_count):
        B, H, W = valid_count.shape
        embedded = self.embedding(valid_count.clamp(max=self.embedding.num_embeddings - 1))
        return embedded.permute(0, 3, 1, 2)

class FusionCropNet(nn.Module):
    def __init__(self,
                 opt_channels: int = 10,
                 sar_channels: int = 3,
                 num_classes: int = 7,
                 feat_dim: int = 512,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 n_heads: int = 16,
                 n_temp_layers: int = 4):
        super().__init__()
        self.opt_encoder = OpticalEncoder(opt_channels, feat_dim,
                                          backbone, pretrained)
        self.sar_encoder = SAREncoder(sar_channels, feat_dim)
        self.fusion = CrossModalAttentionFusion(feat_dim, n_heads)
        
        self.temporal_enc = TemporalEncoder(
            d_model=feat_dim, n_heads=n_heads, num_layers=n_temp_layers
        )
        
        self.spatial_refinement = SpatialRefinement(feat_dim, window_size=8, n_heads=n_heads)
        
        self.obs_embedding = ObservationEmbedding(max_count=365, feat_dim=feat_dim)
        
        self.decoder = DecoderWithSkipConnections(feat_dim, feat_dim//2, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.sar_encoder, self.fusion, self.spatial_refinement, 
                  self.decoder, self.obs_embedding]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, opt_seq, sar_seq, doy, cloud_mask=None, valid_count=None):
        B, T, C_opt, H, W = opt_seq.shape
        _, _, C_sar, _, _ = sar_seq.shape
        
        opt_flat = opt_seq.view(B*T, C_opt, H, W)
        sar_flat = sar_seq.view(B*T, C_sar, H, W)
        
        opt_feat, opt_intermediates = self.opt_encoder(opt_flat)
        sar_feat, sar_intermediate = self.sar_encoder(sar_flat)
        
        fused_feat = self.fusion(opt_feat, sar_feat)
        D, H2, W2 = fused_feat.shape[1:]
        
        fused_feat = fused_feat.view(B, T, D, H2, W2)
        fused_feat = fused_feat.permute(0, 3, 4, 1, 2)
        fused_feat_flat = fused_feat.reshape(B*H2*W2, T, D)

        # Reshape cloud_mask to per-pixel (B*H2*W2, T) matching fused_feat_flat
        if cloud_mask is not None:
            cm_down = F.adaptive_avg_pool2d(
                cloud_mask.view(B * T, 1, H, W).float(), (H2, W2))
            cloud_mask_px = ((cm_down > 0.5).squeeze(1)
                             .view(B, T, H2 * W2)
                             .permute(0, 2, 1)
                             .reshape(B * H2 * W2, T))
        else:
            cloud_mask_px = None

        spatial_context = fused_feat.mean(dim=3).permute(0, 3, 1, 2)

        temp_feat, _, _ = self.temporal_enc(fused_feat_flat, cloud_mask_px, spatial_context)
        
        temp_feat = temp_feat.view(B, H2, W2, D)
        temp_feat = temp_feat.permute(0, 3, 1, 2)
        
        temp_feat = F.interpolate(temp_feat, size=(H, W),
                                  mode="bilinear", align_corners=False)
        
        if valid_count is not None:
            obs_embed = self.obs_embedding(valid_count)
            temp_feat = temp_feat + obs_embed
        
        refined_feat = self.spatial_refinement(temp_feat, valid_count)
        
        opt_skip = opt_intermediates[-2] if len(opt_intermediates) >= 2 else None
        sar_skip = sar_intermediate
        
        logits = self.decoder(refined_feat, sar_skip=sar_skip, opt_skip=opt_skip)
        
        return logits
    
    def get_temporal_attention_map(self):
        return None

class PretrainedWeightManager:
    """Supports both FusionCropNet (V1: opt_encoder/sar_encoder) and
    FusionCropNetV5/V5Pro (opt_enc/sar_enc)."""
    def __init__(self, model):
        self.model = model

    def _get_opt_encoder(self):
        if hasattr(self.model, 'opt_enc'):
            return self.model.opt_enc
        return self.model.opt_encoder

    def freeze_backbone(self, freeze_layers: int = 2):
        opt_enc = self._get_opt_encoder()
        children = list(opt_enc.backbone.children())
        for i, child in enumerate(children):
            if i < freeze_layers:
                for param in child.parameters():
                    param.requires_grad = False
                print(f"冻结层 {i}: {child.__class__.__name__}")

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        print("已解冻所有层参数")

    def get_layerwise_lr_params(self, base_lr: float, backbone_lr_ratio: float = 0.1):
        opt_enc = self._get_opt_encoder()
        backbone_params = list(opt_enc.backbone.parameters())
        other_params = [p for p in self.model.parameters()
                        if not any(p is bp for bp in backbone_params)]
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_ratio,
             "name": "backbone"},
            {"params": other_params, "lr": base_lr,
             "name": "new_layers"}
        ]
    
    def save_checkpoint(self, path: str, epoch: int, metrics: dict, optimizer=None):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "metrics": metrics,
            "model_config": {
                "opt_channels": 10,
                "sar_channels": 3,
                "num_classes": 7,
                "feat_dim": 512,
                "backbone": "resnet50",
                "n_heads": 16,
                "n_temp_layers": 4
            }
        }
        if optimizer:
            checkpoint["optimizer_state"] = optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"[OK] Checkpoint saved: {path} (Epoch {epoch})")
    
    def load_checkpoint(self, path: str, strict: bool = True):
        checkpoint = torch.load(path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(
            checkpoint["model_state"], strict=strict
        )
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")
        print(f"[OK] Loaded checkpoint: {path} (Epoch {checkpoint['epoch']})")
        return checkpoint
    
    def transfer_from_source_domain(self, source_ckpt_path: str):
        checkpoint = torch.load(source_ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state"]
        filtered = {k: v for k, v in state_dict.items()
                    if not k.startswith("classifier") and 
                       not k.startswith("decoder")}
        self.model.load_state_dict(filtered, strict=False)
        print("✓ 已迁移特征提取层权重（分类头重新初始化）")