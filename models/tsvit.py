import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=10, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TemporalViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), x, x)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TSViT(nn.Module):
    def __init__(self, in_channels=10, num_classes=7,
                 embed_dim=128, depth=4, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TemporalViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, doy=None):
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        x = x.reshape(B, T, -1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        return self.head(cls_output)


class SpatialViT(nn.Module):
    def __init__(self, in_channels=10, num_classes=7,
                 embed_dim=128, depth=2, num_heads=8,
                 patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=self.patch_size, stride=self.patch_size)

        self.num_patches = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None

        self.blocks = nn.ModuleList([
            TemporalViTBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def _init_pos_embed(self, num_patches):
        if self.pos_embed is None or self.pos_embed.shape[1] != num_patches + 1:
            self.num_patches = num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.proj(x)

        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        self._init_pos_embed(num_patches)

        x = x.reshape(B, T, self.embed_dim, num_patches_h, num_patches_w)
        x = x.permute(0, 1, 3, 2, 4).reshape(B, T * num_patches_h * num_patches_w, self.embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])