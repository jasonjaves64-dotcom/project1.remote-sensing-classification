# FusionCropNet V6 — 基础模型骨干技术设计

**日期**: 2026-05-19
**状态**: 设计完成，待实施
**依赖**: [[2026-05-19-fusion-v6-design|V6 总体设计]] · [[2026-05-19-model-comparison|竞品对比]]

---

## 一、设计目标

V6 骨干需要解决 V5Pro 的四个核心瓶颈：

| 瓶颈 | 根因 | V6 方案 |
|------|------|---------|
| 无遥感域预训练 | ImageNet → 遥感域迁移有限（SpectralEarth 论文：差 30pp OA） | 冻结遥感预训练 ViT + LoRA |
| 逐像素时序 O((THW)²) | Joint 时空注意，H=W=64 时 ~6×10^8 ops | 解耦时空注意 → O(T² + (HW)²) |
| 模态隔离 | opt/sar/dem 在浅层无交互 | 三级融合 (Early Concat + Mid CrossAttn + Late Ensemble) |
| 单层时序 | 缺乏田块/场景上下文 | Scene → Region → Pixel 层次化编码器 |

---

## 二、架构总图

```
INPUT: opt(B,T,10,H,W)  sar(B,T,5,H,W)  dem(B,5,H,W)  doy(B,T)

┌─ EARLY FUSION ──────────────────────────────────────────────┐
│ ModalNormalize → [Opt│SAR│DEM] → 1×1 Conv → unified_early   │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                   ▼
    OpticalFPN          SARFPN             DEMEncoder
    p2,p3,p4           s1,s2,s3            dem_feat
         │                 │                   │
         └────────┬────────┘                   │
                  │                            │
┌─ MID FUSION ────┼────────────────────────────┼─────────────┐
│ H/2: opt_p2 ↔ sar_s1 → fusion_h2 → Decoder skip           │
│ H/4: opt_p3 ↔ sar_s2 → fusion_h4 → Region Encoder input   │
│ H/8: opt_p4 ↔ sar_s3 → fusion_h8 → LateFusion input       │
└──────────────────┬─────────────────────────┬───────────────┘
                   │                         │
┌─ HIERARCHICAL ───┼─────────────────────────┼───────────────┐
│                  │                         │               │
│ Pixel Encoder    │        SCENE ENCODER    │               │
│ (Decoupled T/S)  │        GeoFM ViT        │               │
│   ↓              │        + LoRA           │               │
│ Region Encoder   │        → scene_token    │               │
│ (Swin W-MSA)     │        → scene_prior    │               │
│   ↑↓             │                         │               │
│ Cross-Hierarchy  │                         │               │
└──────────────────┬─────────────────────────┬───────────────┘
                   │                         │
┌─ DECODER ────────┼─────────────────────────┼───────────────┐
│ CARAFE + Skip Connections + Scene Broadcast + DEM Spatial  │
└──────────────────┬─────────────────────────────────────────┘
                   │
┌─ LATE FUSION ────┼─────────────────────────────────────────┐
│ Expert_opt | Expert_sar | Expert_fused                     │
│ Vacuity-weighted ensemble → final_pred                     │
└──────────────────┬─────────────────────────────────────────┘
                   │
┌─ MULTI-TASK HEADS ─────────────────────────────────────────┐
│ CropType(EDL) | LAI/Huber | Growth/CE | Boundary/Dice      │
└────────────────────────────────────────────────────────────┘
```

---

## 三、预训练遥感基础模型

### 3.1 候选对比

| 模型 | 预训练数据 | 方法 | 骨干 | 适用模态 | 可用性 |
|------|-----------|------|------|---------|--------|
| **GeoFM** (首选) | 多源遥感 S2+S1+DEM | MAE + 对比 | ViT-L | 三模态 | 待确认 |
| SSL4EO-S12 (备选) | 25万 S2 patches | MoCo-v3 | ViT-B | 多光谱 S2 | 公开 |
| SpectralEarth-ViT | 53.8万 EnMAP patch | 监督 | ViT-G | 高光谱 | 公开 |
| CropSTS | S2 PASTIS | JEPA + 蒸馏 | ViT-S | 多光谱 | 公开 |
| SatMAE | fMoW 多时相 | MAE | ViT-B/L | 多光谱 | 公开 |

**选择逻辑**: GeoFM 是唯一一个在三模态（Opt+SAR+DEM）数据上预训练的模型，与 V5Pro 模态完全匹配。若其权重最终不可用，回退到 SSL4EO-S12 + 额外 Cross-Modal Adapter 桥接 SAR/DEM。

### 3.2 集成架构

```
GeoFM ViT (frozen)
  │
  ├── LoRA @ [qkv, proj, fc1]  ← 仅 0.5% 参数可训练
  ├── LayerNorm γ,β Adapters   ← 适应遥感像素分布
  └── ViTFeaturePyramid        ← 将 ViT 单尺度输出转为多尺度
```

```python
# Modal Adapters — 将多通道映射到 3ch ViT 输入
modal_adapter = {
    "opt": Conv1x1(10, 64) → GELU → Conv1x1(64, 3),
    "sar": Conv1x1(5, 32)  → GELU → Conv1x1(32, 3),
    "dem": Conv1x1(5, 64)  → GELU → Conv1x1(64, 3),
}

# LoRA config
lora_config = {
    "target_modules": ["qkv", "proj", "fc1"],
    "rank": 16, "alpha": 32, "dropout": 0.1
}
# 300M total → 0.6M trainable via LoRA
```

### 3.3 备选：SSL4EO-S12 Cross-Modal Adapter

若 GeoFM 不可用，SAR 和 DEM 需额外对齐到 SSL4EO 的光学特征空间：

```python
class CrossModalAdapter(nn.Module):
    """以光学特征为 anchor，对齐 SAR/DEM"""
    def __init__(self, in_ch, hidden=128):
        self.proj = Conv1x1(in_ch, hidden)
        self.cross_attn = CrossAttention(hidden, heads=4)

    def forward(self, sar_feat, opt_feat):
        sar_proj = self.proj(sar_feat)
        aligned = self.cross_attn(
            query=opt_feat,      # 光学提供空间锚点
            key=sar_proj,
            value=sar_proj
        )
        return aligned + sar_proj
```

---

## 四、三级融合体系

### 4.1 Level 1: Early Fusion（特征拼接）

**设计**: 在所有模态独立 FPNE 之前，先做模态归一化 + 通道拼接 + 1×1 压缩。

```python
class ModalNormalize(nn.Module):
    """Per-modality LayerNorm — 解决数值范围不一致"""
    def forward(self, opt, sar, dem):
        return torch.cat([
            F.layer_norm(opt, opt.shape[1:]),   # [0,1] reflectances
            F.layer_norm(sar, sar.shape[1:]),   # [-25,5] dB
            F.layer_norm(dem, dem.shape[1:]),   # [0,8848] meters
        ], dim=1)

# Pipeline
unified = ModalNormalize()(opt, sar, dem)  # (B,T,256,H,W)
unified = Conv1x1(256, 128)(unified)       # compress
```

### 4.2 Level 2: Mid Fusion（三尺度 Cross-Attention）

| 尺度 | 分辨率 | 功能 | 注意头 | Q 来源 |
|------|--------|------|--------|--------|
| H/2 | 128×128 | 纹理对齐 | 1 (Light) | unified |
| H/4 | 64×64 | 结构对齐 | 4 | unified |
| H/8 | 32×32 | 语义对齐 | 8 (新增) | unified |

Q 来自 unified 而非纯 opt — 确保云遮挡时 SAR 信息仍在 Q 中。K/V 从原始模态取，保留模态纯度。

```python
# H/2: Light — 单头，追求速度
fusion_h2 = CrossModalAttentionLight(
    query=unified_h2, key=sar_s1, value=sar_s1
)

# H/4: Standard — 4头
fusion_h4 = CrossModalAttention(
    query=unified_h4, key=sar_s2, value=sar_s2
)

# H/8: Deep (新增) — 8头语义对齐
fusion_h8 = CrossModalAttention(
    query=unified_h8, key=sar_s3, value=sar_s3,
    num_heads=8
)
```

### 4.3 Level 3: Late Fusion（决策级融合）

三路 Expert 各自预测，EDL vacuity 自动加权：

```python
# Three experts with shared backbone
logits_opt, alpha_opt   = CropHead(opt_path_features)
logits_sar, alpha_sar   = CropHead(sar_path_features)
logits_fused, alpha_fused = CropHead(shared_features)

# Vacuity-based weights (per-pixel)
vacuity_opt   = K / alpha_opt.sum(dim=1)     # 证据不足 → 权重低
vacuity_sar   = K / alpha_sar.sum(dim=1)
vacuity_fused = K / alpha_fused.sum(dim=1)

weights = softmax(MLP([vacuity_opt, vacuity_sar, vacuity_fused]))

final_logits = (w_opt * logits_opt +
                w_sar * logits_sar +
                w_fused * logits_fused)
```

缺失模态自动 fallback: 模态缺失 → vacuity → ∞ → weight → 0。

---

## 五、层次化编码器

### 5.1 Pixel Encoder: 解耦时空注意

CropSTS (Yan et al. 2025) 的关键洞察：遥感像素坐标在时序上不变，因此时间注意和空间注意可以解耦。

```
传统 Joint: softmax(Q[THW]·K[THW]^T / √d)   → O(T²×(HW)²)
解耦:
  h_time  = softmax(Q_time·K_time^T / √d)·V    → O(T²) per pixel
  h_space = softmax(Q_space·K_space^T / √d)·V  → O((HW)²) per timestep
  output  = γₜ·h_time + γₛ·h_space             (γₜ,γₛ 可学习)
```

H=W=64, T=12: 从 6.0×10^8 → 1.2×10^7 (~50× 加速)。

```python
class DecoupledSpatiotemporalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        self.temporal_attn = MHSA(dim, heads)   # attend over T
        self.spatial_attn  = MHSA(dim, heads)   # attend over HW
        self.gamma_t = nn.Parameter(torch.tensor(0.5))
        self.gamma_s = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):  # x: (B, T, HW, C)
        # Temporal-first (物候优先)
        x = x + self.gamma_t * self.temporal_attn(LN(x))
        x = x + self.gamma_s * self.spatial_attn(LN(x))
        return x + FFN(LN(x))
```

### 5.2 Region Encoder: Swin W-MSA

| 属性 | 值 | 理由 |
|------|-----|------|
| window_size | 8 | 8×8=64 像素 ≈ 小块田 |
| depth | 4 | 4 层交替 W-MSA / SW-MSA |
| shift_size | 4 | 避免窗口边界 artifact |
| 输出 | K²×512 tokens | K = H/32 |

```python
region_encoder = SwinTransformer(
    patch_size=8,           # 8×8 pixel window
    embed_dim=512,
    depths=[2, 2],          # 2 W-MSA + 2 SW-MSA layers
    num_heads=[8, 8],
    window_size=8,
)
region_tokens = region_encoder(pixel_tokens)
```

### 5.3 Scene Encoder: GeoFM ViT + 场景先验

```
unified_pooled (B,T,128,H/32,W/32)
    → temporal_mean → (B,128,H/32,W/32)
    → GeoFM ViT
    → scene_token (B,1024)
    → MLP:
        ├ scene_type ∈ {水田,旱地,混合,设施农业}
        ├ crop_mix ∈ R^K      (主要作物分布)
        └ terrain ∈ {平原,丘陵,山地,梯田}
```

### 5.4 Cross-Hierarchy Attention

```python
# Scene → Region (top-down guidance)
region_tokens = region_tokens + CrossAttn(
    Q=region_tokens, K=scene_token, V=scene_token
)

# Region → Pixel (bottom-up refinement)
pixel_tokens = pixel_tokens + CrossAttn(
    Q=pixel_tokens, K=region_tokens, V=region_tokens
)
```

---

## 六、训练策略

### 6.1 三阶段训练

| 阶段 | Epochs | LR | 冻结 | 训练 |
|------|--------|-----|------|------|
| Phase 1: Warm-up | 10 | 1e-3 | GeoFM ViT, LoRA | ModalAdapter, FPN, CrossAttn, Decoder, Heads |
| Phase 2: Main | 50 | 1e-4 | GeoFM ViT | LoRA, CrossAttn, Hierarchical, Heads |
| Phase 3: Fine-tune | 20 | 1e-5 | — | 全部 (含 GeoFM 顶层 2 层) |

### 6.2 多任务损失

| 任务 | Head | Loss | 权重 | 调度 |
|------|------|------|------|------|
| 作物分类 | EDLHead | EDL Loss (CE + KL) | 1.0 | 恒定 |
| LAI/Biomass | MLP(64→32→1) | Huber (δ=1.0) | 0.3 | 恒定 |
| 生长阶段 | MLP(64→32→N) | CrossEntropy | 0.2 | 恒定 |
| 田块边界 | UNetEdgeHead | Dice + BCE | 0.1 | 恒定 |
| NDVI Aux | PhenoAux | MSE | 0.1 | Cosine anneal → 0 |

### 6.3 正则化 & 优化

- **Optimizer**: AdamW (β₁=0.9, β₂=0.999, weight_decay=0.05)
- **Scheduler**: CosineAnnealingWarmRestarts (T₀=10, T_mult=2)
- **Curriculum Dropout**: `p_drop = 0.05 + 0.15·sin(π·epoch/total)` (继承 V5Pro)
- **Adaptive KL Annealing**: Spearman 秩相关驱动 (继承 V5Pro)
- **AMP**: FP16 自动混合精度
- **Gradient Checkpointing**: 时序编码器反向传播重算

---

## 七、内存 & 计算优化

| 技术 | 应用位置 | 预期收益 |
|------|---------|---------|
| Flash Attention | 所有 Transformer 层 | 内存 O(N²)→O(N), 速度 2-4× |
| 梯度检查点 | 时序编码器 BPTT | 显存 -40% |
| 混合精度 AMP | 全局 | 显存 -40%, 速度 +30% |
| 分块推理 | 推理脚本 | 任意大影像可处理 |
| LoRA 低秩 | GeoFM ViT | 可训练参数 0.2% |

---

## 八、V5Pro → V6 模块变更

| 模块 | V5Pro | V6 | 操作 |
|------|-------|-----|------|
| Backbone | ResNet/ConvNeXt (ImageNet) | GeoFM/SSL4EO ViT (遥感) | **替换** |
| Early Fusion | — | ModalNorm + Concat | **新增** |
| Mid Fusion | H/4 + H/2 (2层) | H/8 + H/4 + H/2 (3层) | 扩展 |
| Pixel Encoder | Joint Attn O((THW)²) | Decoupled O(T²+(HW)²) | **重写** |
| Region Encoder | — | Swin W-MSA | **新增** |
| Scene Encoder | — | GeoFM ViT + LoRA | **新增** |
| Cross-Hierarchy | — | Scene→Region→Pixel | **新增** |
| Late Fusion | 单头 | 3-Expert Ensemble | **新增** |
| Decoder | CARAFE | CARAFE + scene_broadcast | 增强 |
| Multi-Task | NDVI aux | 5 heads | 扩展 |
| AMP + GC | — | 全局 | **新增** |

---

## 九、风险 & 对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| GeoFM 权重不可用 | 中 | 需重新设计模态适配 | 回退 SSL4EO-S12 + Cross-Modal Adapter |
| 多任务梯度冲突 | 低 | 收敛慢或不收敛 | Uncertainty Weighting + PCGrad |
| 三层 CrossAttn OOM | 中 | 无法训练 | 顺序计算 + GC + 梯度累积 |
| 解耦注意精度下降 | 低 | OA 低于 V5Pro | 添加 joint attention skip connection 兜底 |
| CPU only 训练慢 | 高 | 迭代周期长 | 小 batch + 梯度累积模拟大 batch |

---

## 十、预期指标

| 指标 | V5Pro | V6 目标 | 改善来源 |
|------|-------|---------|---------|
| mIoU (合成) | ~0.15 | ~0.30 | 预训练 + 层次化 + 三级融合 |
| 推理速度 | 1× | 1.5-2× | 解耦注意 + AMP + Flash Attn |
| 推理显存 | 1× | 0.4× | 分块推理 + GC |
| 训练显存 | 1× | 0.6× | AMP + GC + LoRA |
| 标签效率 | 1× | 2-3× | 遥感预训练 + 主动学习 |
| 缺失模态鲁棒 | ★★★★ | ★★★★★ | Late Fusion 自动 fallback |

---

## 十一、实施路线图

Phase 1: 解耦时空注意 + AMP + GC（1-2 周）
Phase 2: 预训练模型集成 + LoRA + ModalAdapter（2-3 周）
Phase 3: Early Concat + 三级 CrossAttn（1 周）
Phase 4: Region + Scene + Cross-Hierarchy（2-3 周）
Phase 5: Late Fusion + 多任务 Heads（1-2 周）
Phase 6: 测试 + 文档（1 周）

总工期: 8-12 周（取决于 Phase 2 权重获取情况）。

---

*关联: [[2026-05-19-fusion-v6-design|V6 总体设计]] · [[2026-05-19-model-comparison|竞品对比]]*
*Obsidian: [[model-FusionCropNet-V6-骨干设计]]*
