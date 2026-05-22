import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        return x_flat.transpose(1, 2).view(B, C, H, W)

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_layers=2, use_checkpointing=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(in_channels, num_heads) for _ in range(num_layers)
        ])
        self.use_checkpointing = use_checkpointing
    
    def _checkpointed_block(self, block, x):
        return torch.utils.checkpoint.checkpoint(block, x)
    
    def forward(self, x):
        x = self.conv(x)
        for block in self.transformer_blocks:
            if self.use_checkpointing and self.training:
                x = self._checkpointed_block(block, x)
            else:
                x = block(x)
        return x

class ViTBottleneck(nn.Module):
    def __init__(self, in_channels=512, embed_dim=512, num_heads=16, num_layers=4):
        super().__init__()
        self.patch_size = 4
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        encoder_layers = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        x_proj = self.proj(x)
        _, D, Hp, Wp = x_proj.shape
        
        x_flat = x_proj.flatten(2).transpose(1, 2)
        x_transformed = self.transformer(x_flat)
        x_transformed = self.norm(x_transformed)
        
        x_out = x_transformed.transpose(1, 2).view(B, D, Hp, Wp)
        
        return F.interpolate(x_out, size=(H, W), mode='bilinear', align_corners=False)

class UNetTransformer(nn.Module):
    def __init__(self, n_channels=10, n_classes=7, bilinear=False, 
                 use_transformer=True, num_heads=8, depth=4,
                 use_checkpointing=False):
        super(UNetTransformer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_transformer = use_transformer
        self.use_checkpointing = use_checkpointing
        
        chs = [64, 128, 256, 512, 1024]
        self.channels = chs[:depth+1]
        
        self.inc = DoubleConv(n_channels, self.channels[0])
        
        self.down_layers = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        
        for i in range(depth):
            self.down_layers.append(Down(self.channels[i], self.channels[i+1]))
            
            if use_transformer and i >= 1:
                self.transformer_layers.append(SpatialTransformer(
                    self.channels[i+1], num_heads, use_checkpointing=use_checkpointing
                ))
            else:
                self.transformer_layers.append(nn.Identity())
        
        self.bottleneck = ViTBottleneck(self.channels[-1], self.channels[-1], num_heads, num_layers=4)
        
        self.up_layers = nn.ModuleList()
        self.up_transformer_layers = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            self.up_layers.append(Up(self.channels[i+1] + self.channels[i], self.channels[i], bilinear))
            
            if use_transformer and i >= 1:
                self.up_transformer_layers.append(SpatialTransformer(
                    self.channels[i], num_heads // 2, use_checkpointing=use_checkpointing
                ))
            else:
                self.up_transformer_layers.append(nn.Identity())
        
        self.outc = OutConv(self.channels[0], n_classes)
    
    def forward(self, x):
        features = []
        
        x = self.inc(x)
        features.append(x)
        
        for down, transformer in zip(self.down_layers, self.transformer_layers):
            x = down(x)
            x = transformer(x)
            features.append(x)
        
        x = self.bottleneck(x)
        
        for i, (up, transformer) in enumerate(zip(self.up_layers, self.up_transformer_layers)):
            skip = features[-(i + 2)]
            x = up(x, skip)
            x = transformer(x)
        
        logits = self.outc(x)
        return logits

class UNetTransformerWithSAR(nn.Module):
    def __init__(self, opt_channels=10, sar_channels=5, n_classes=7, 
                 use_transformer=True, num_heads=8, depth=4,
                 use_checkpointing=False):
        super().__init__()
        
        self.opt_encoder = UNetTransformer(
            n_channels=opt_channels, 
            n_classes=n_classes,
            use_transformer=use_transformer,
            num_heads=num_heads,
            depth=depth,
            use_checkpointing=use_checkpointing
        )
        
        self.sar_encoder = UNetTransformer(
            n_channels=sar_channels,
            n_classes=n_classes,
            use_transformer=use_transformer,
            num_heads=num_heads,
            depth=depth,
            use_checkpointing=use_checkpointing
        )
        
        self.fusion_conv = nn.Sequential(
            DoubleConv(n_classes * 2, n_classes),
            nn.Conv2d(n_classes, n_classes, 1)
        )
    
    def forward(self, opt, sar, dem=None, doy=None, cloud_mask=None, valid_count=None):
        if dem is not None or doy is not None:
            import warnings
            warnings.warn(
                "UNetTransformer does not use dem/doy/cloud_mask/valid_count. "
                "These inputs are ignored. Temporal information is collapsed via "
                "mean-pooling across the time dimension. For full multi-modal "
                "temporal modeling, use FusionCropNetV5EDL instead.")
        B, T, C, H, W = opt.shape

        opt_flat = opt.view(B * T, C, H, W)
        sar_flat = sar.view(B * T, sar.shape[2], H, W)

        opt_logits = self.opt_encoder(opt_flat)
        sar_logits = self.sar_encoder(sar_flat)

        opt_logits = opt_logits.view(B, T, -1, H, W).mean(dim=1)
        sar_logits = sar_logits.view(B, T, -1, H, W).mean(dim=1)

        fused = self.fusion_conv(torch.cat([opt_logits, sar_logits], dim=1))

        return fused