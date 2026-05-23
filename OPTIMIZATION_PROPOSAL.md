# FusionCropNet 模型分析与优化方案

> 更新日期: 2026-05-16

## 一、当前模型架构总览

### 1.1 版本演进

| 版本 | 核心创新 | 输入模态 | 时序建模 | 不确定性 |
|------|---------|---------|---------|---------|
| V1 (fusion_net.py) | 光学+SAR跨模态注意融合 | 光学+SAR | 标准Transformer | 无 |
| V4 (fusion_net_v4.py) | DEM编码器+SWin注意力 | 光学+SAR+DEM | Transformer+FiLM | MC-Dropout |
| V5 (fusion_net_v5.py) | FiLM调制+DEM空间条件+NDVI辅助 | 光学+SAR+DEM | Transformer+质量Token | 训练辅助 |
| V5EDL | EDL不确定性+模态解耦+时序Dropout | 光学+SAR+DEM | Transformer+质量Token | EDL证据理论 |

### 1.2 V5 完整架构流程

```
输入: opt_seq(B,T,10,H,W)  sar_seq(B,T,5,H,W)  dem(B,5,H,W)  doy(B,T)
                                                                    │
    ┌───────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ DEMEncoder   │   │ OpticalEnc   │   │   SAREnc     │
│ (5→128)      │   │ (10→512,FPN) │   │ (5→512,IRB)  │
│ local+global │   │ ResNet50骨干 │   │ FiLM×DEM     │
│ skip连接     │   │              │   │ 3级下采样    │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │dem_feat          │opt_feat           │sar_s1,s2,s3
       │(B,128,H,W)       │(B*T,512,H/4,W/4)  │
       │                  │opt_p2, opt_p3     │
       │                  │                   │
       │           ┌──────▼──────┐    ┌──────▼──────┐
       │           │ Phenology   │    │              │
       │           │ AuxHead     │    │              │
       │           │ (NDVI预测)  │    │              │
       │           └─────────────┘    │              │
       │                              │              │
       │    ┌─────────────────────────▼──────────────▼──┐
       │    │  Pixel-wise Temporal Encoding             │
       │    │  opt_ts: (B*H2*W2, T, 512)               │
       │    │  sar_ts: (B*H2*W2, T, 512)               │
       │    │  + FourierDOY + ObsQualityToken           │
       │    │  + CloudMask padding + Fallback            │
       │    └──────────────┬────────────────────────────┘
       │                   │ opt_g, sar_g (B*H2*W2, 512)
       │                   │
       │    ┌──────────────▼──────────────────────────┐
       │    │  Spatial Reshape → (B,512,H2,W2)        │
       │    └──────────────┬──────────────────────────┘
       │                   │
       │    ┌──────────────▼──────────┐
       │    │  CrossModalAttention    │
       │    │  opt↔SAR双向交叉注意    │
       │    │  门控融合                │
       │    └──────────────┬──────────┘
       │                   │ xm_feat (B,512,H2,W2)
       │                   │
       └───────────────────►├──────────────────────────┐
                            │  DEMSpatialConditioner    │
                            │  FiLM + Gate + Mix       │
                            └──────────────┬───────────┘
                                           │
                            ┌──────────────▼───────────┐
                            │  LateFusion              │
                            │  Gate(xm, opt, sar)      │
                            │  → (B,H2,W2,512)         │
                            └──────────────┬───────────┘
                                           │
                            ┌──────────────▼───────────┐
                            │  Decoder                 │
                            │  2×ConvTranspose2d       │
                            │  + Skipped opt_p2, sar_s1,s2 │
                            │  + SpatialRefinement     │
                            │  → pre_head (B,64,H,W)   │
                            └──────────────┬───────────┘
                                           │
                         ┌─────────────────▼────────────┐
                         │  V5: cls_head → logits       │
                         │  V5EDL: EDLHead → alpha      │
                         └──────────────────────────────┘
```

### 1.3 关键组件详解

| 组件 | 输入 | 输出 | 关键设计 |
|------|------|------|---------|
| **DEMEncoder** | DEM(5ch) | 128ch特征 | 局部CNN + 全局MLP门控 + skip连接 |
| **OpticalEncoder** | 光学(10ch) | 512ch+多尺度 | ResNet50骨干 + FPN + 通道适应 |
| **SAREncoder** | SAR(5ch) | 64/128/512ch | IRB + FiLM×DEM调节 + 3级下采样 |
| **TemporalEncoderStream** | 逐像素时序 | 全局+序列特征 | FourierDOY + 质量Token + CLS + Fallback |
| **CrossModalAttention** | opt(512)+sar(512) | 融合特征(512) | 双向交叉注意 + 门控 + 残差 |
| **DEMSpatialConditioner** | 融合+DEM | 条件化特征 | FiLM + 门控混合 + GroupNorm |
| **LateFusion** | xm+opt+sar | 最终特征 | 三门控 + 投影 |
| **Decoder** | 融合特征+跳跃连接 | 预分类特征(64) | 2×上采样 + SpatialRefinement |
| **EDLHead** (V5EDL) | 64ch特征 | Dirichlet α | Dropout + softplus → 不确定性 |
| **PhenologyAuxHead** | 光学特征 | NDVI预测 | 辅助任务 + 退火权重 |

---

## 二、模型优缺点分析

### 2.1 优势

| 优势 | 说明 |
|------|------|
| **多模态融合完整** | 光学+SAR+DEM三种互补模态，光学提供光谱，SAR全天候结构，DEM地形先验 |
| **时序建模精细** | 逐像素Transformer编码，Fourier日期编码，云掩膜处理，观测质量token |
| **物理先验集成** | NDVI物候辅助任务，DEM地形调节SAR特征(FiLM)，增强可解释性 |
| **不确定性量化(EDL)** | 证据深度学习分离vacuity(数据不确定)和dissonance(分布不确定) |
| **两阶段训练策略** | 阶段1冻结骨干→阶段2全量微调+分层学习率 |
| **模态鲁棒性(EDL)** | 模态Dropout训练、可学习占位符、fallback门控 |
| **数据增强完善** | DEM噪声注入、同步空间平移、时序Dropout、频谱噪声 |

### 2.2 劣势与改进空间

| 劣势 | 严重度 | 状态 |
|------|--------|------|
| **代码重复严重** | 高 | ✅ 已修复 — `_base.py` 统一管理 |
| **V5存在7个已知Bug** | 高 | ✅ 已修复 |
| **全像素自注意O(n²)** | 高 | ✅ 已修复 — 窗口化注意力 |
| **时序编码计算量大** | 中 | 待优化 — 逐像素Transformer(B×H2×W2条序列) |
| **缺乏层次化时序建模** | 中 | 待实现 — 仅像素级编码 |
| **光学骨干固定为ResNet** | 中 | 待实现 — 不支持Swin/ConvNeXt |
| **交叉注意仅在单尺度** | 中 | 待实现 — 多尺度融合 |
| **DEM使用有限** | 低 | 仅在SAR FiLM和后期条件中使用 |
| **无自监督预训练** | 低 | 依赖ImageNet预训练(遥感域迁移有限) |
| **EDL KL退火简单** | 低 | 线性退火，可改用余弦或自适应 |

---

## 三、已完成工作 (v5.1 — 2026-05-16)

### 3.1 代码架构重构

**目标**: 消除4个文件间的15个组件重复定义

```
之前:
├── fusion_net_v5.py       (528行) — 内联所有组件定义
├── fusion_net_v5_edl.py   (890行) — 内联所有组件定义
├── dem_encoder.py         — 独立定义 ConvBNGELU, SEBlock, DEMEncoder
├── heads.py               — 独立定义 ConvBNGELU, SEBlock, SpatialRefinement
└── temporal.py            — 独立定义 FourierDOYEncoding, ObsQualityToken

之后:
├── _base.py               (314行) — 规范来源，15个共享组件
├── fusion_net_v5.py       (150行) — 仅模型组装 + V5特有 cls_head
├── fusion_net_v5_edl.py   (350行) — 仅EDL组件 + 模型组装
├── dem_encoder.py         — 导入_base，保留 FiLMLayer + ThreeWayFusion
├── heads.py               — 导入_base，保留 SWBlock + UncertaintyHead
└── temporal.py            — 导入_base，保留 FiLMModulation + V4 TemporalEncoderStream
```

**`_base.py` 包含的规范组件**:
```
ConvBNGELU, SEBlock, FiLM, IRB,
DEMEncoder, FPN, OpticalEncoder, SAREncoder,
FourierDOYEncoding, ObsQualityToken, TemporalEncoderStream,
CrossModalAttention, DEMSpatialConditioner, LateFusion,
SpatialRefinement, Decoder, PhenologyAuxHead, time_average
```

### 3.2 7个Bug修复明细

| # | Bug | 位置 | 症状 | 修复方式 |
|---|-----|------|------|---------|
| BUG1 | `nn.Linear` 在 forward() 内创建 | fusion_net_v5.py L499 | 内存泄漏，梯度错误 | 移至 `__init__` 为 `self.consistency_proj` |
| BUG2 | CrossModalAttention 迭代 `.children()` | fusion_net_v5.py L271-274 | 每个子模块用相同输入 → 破坏计算图 | `self.sw_o2s(o2s)` 直接调用 Sequential |
| BUG3 | SpatialRefinement 展平全空间做自注意 | fusion_net_v5.py L319-325 | H'×W'=256时O(n²)=65K → OOM | 窗口化注意力 (einops rearrange) |
| BUG4 | DEM空间平移未同步其他模态 | fusion_net_v5.py L450-458 | 光学/SAR/掩膜与DEM不对齐 | `_shift_inputs()` 统一平移5种输入 |
| BUG5 | 5D张量 F.pad 维度顺序错误 | _shift_inputs | RuntimeError: padding size invalid | `pad_hw + (0,0)` 正确顺序 |
| BUG6 | Bool张量不支持 replicate padding | _shift_inputs | NotImplementedError for Bool | float→pad→还原为bool |
| BUG7 | valid_count 多维度切片索引错误 | _shift_inputs | 3D/4D张量索引维度混淆 | 按 `dim()` 区分索引方式 |

### 3.3 其他改进

- ✅ `models/__init__.py` 导出全版本（V1/V4/V5/V5EDL/V5MIL/UNet + 共享组件）
- ✅ `FusionCropDatasetEDL` 从脚本提取到 `data/datasets/fusion_dataset.py`
- ✅ 消除跨脚本依赖（两个训练脚本统一从 `data.datasets` 导入）
- ✅ 清理 ~650MB 构建产物和过时文件
- ✅ 更新 `.gitignore`
- ✅ 31/31 单元测试通过

### 3.4 当前代码结构（重构后）

```
models/
├── _base.py                ← 规范来源 — 15个共享组件
├── __init__.py              ← 导出全部模型版本
├── fusion_net.py            ← V1 (独立，架构不同)
├── fusion_net_v4.py         ← V4 (自包含，组件接口不同)
├── fusion_net_v5.py         ← V5 (150行，组装_base组件)
├── fusion_net_v5_edl.py     ← V5EDL (350行，+EDL特有组件)
├── dem_encoder.py           ← 导入_base + FiLMLayer + ThreeWayFusion + compute_dem_bands
├── heads.py                 ← 导入_base + SWBlock + UncertaintyHead + V4变体
├── temporal.py              ← 导入_base + FiLMModulation + V4 TemporalEncoderStream
├── mil_module.py            ← MIL扩展 (独立)
├── unet_transformer.py      ← UNet替代架构 (独立)
└── tsvit.py                 ← Temporal ViT (独立)
```

---

## 四、Fusion Net V5 Pro 设计方案

### 4.1 设计哲学

V5 Pro 在 V5EDL 成熟架构上定向增强：
- 保留已验证的核心流程(多模态→时序→融合→解码)
- 在计算允许的前提下引入现代架构组件
- 增强可扩展性和可维护性

### 4.2 改进项目

| 优先级 | 改进项 | 效果 | 实现复杂度 |
|--------|--------|------|-----------|
| 高 | 可插拔光学骨干 (ConvNeXt/Swin) | 特征提取能力↑ | 低 — 仅修改 OpticalEncoder |
| 高 | 多尺度跨模态融合 (3层) | 细节保留↑，小类IoU↑ | 中 — 需新增融合模块 |
| 高 | 层次化时序聚合 | 推理速度↑10-15%，减少O(n²) | 高 — 架构变更较大 |
| 中 | CARAFE 上采样替代转置卷积 | 边界质量↑，棋盘伪影↓ | 低 — 替换 Decoder up 层 |
| 中 | 动态时序 Dropout (课程式) | 鲁棒性↑，过拟合↓ | 低 — 修改 drop_p 调度 |
| 低 | 自适应 KL 退火 | ECE↓ 0.02-0.05 | 低 — 修改 EDLLoss |
| 低 | 对比学习遥感预训练 | 数据效率↑，迁移能力↑ | 高 — 需大规模预训练 |

### 4.3 可插拔骨干

```python
_SUPPORTED_BACKBONES = {
    "resnet50":        [256, 512, 1024, 2048],
    "convnext_tiny":   [96, 192, 384, 768],
    "convnext_small":  [96, 192, 384, 768],
    "swin_tiny":       [96, 192, 384, 768],
    "efficientnet_b4": [24, 48, 120, 336],
    "maxvit_tiny":     [64, 128, 256, 512],
}
```

### 4.4 V5 Pro 架构

```
                    ┌─────────┐  ┌─────────┐  ┌─────────┐
                    │  Optical │  │   SAR   │  │   DEM   │
                    │ B×T×10  │  │ B×T×5   │  │  B×5    │
                    └────┬─────┘  └────┬─────┘  └────┬─────┘
                         │             │              │
          ┌──────────────▼──┐  ┌──────▼──────┐  ┌────▼──────────┐
          │ Hierarchical    │  │ SAR Enc     │  │ DEM Enc       │
          │ Opt Enc         │  │ (IRB+FiLM)  │  │ (CNN+Global)  │
          │ (ConvNeXt/Swin) │  │ 3级特征     │  │ → 128ch       │
          └──────┬──────────┘  └──────┬──────┘  └────┬──────────┘
                 │                    │              │
    ┌────────────▼────────┐  ┌───────▼──────┐       │
    │ Multi-Scale         │  │              │       │
    │ Temporal Aggregation│  │              │       │
    │ (PCT + MSTrans)     │  │              │       │
    └────────────┬────────┘  │              │       │
                 │             │              │       │
    ┌────────────▼─────────────▼──────────────▼───────┐
    │         Multi-Scale Cross-Modal Fusion           │
    │  Level 3 (high-semantic): Cross-Attn + DEM Cond │
    │  Level 2 (mid): Cross-Attn + SE-Gate            │
    │  Level 1 (low): Concat + CBAM                   │
    └──────────────────────┬──────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │  LateFusion (3-gate)│
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Decoder            │
                │  (CARAFE上采样)     │
                │  + 多尺度跳跃连接   │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  EDL Head           │
                │  → α ∈ R^K          │
                └─────────────────────┘
```

### 4.5 预期提升

| 指标 | V5EDL | V5 Pro (预期) | 提升来源 |
|------|-------|---------------|---------|
| mIoU | baseline | +2~4% | 多尺度融合 + 层次化时序 |
| 推理速度 | baseline | +10~15% | 层次化时序减少计算量 |
| ECE | baseline | -0.02~0.05 | 自适应KL退火 |
| 小类IoU | baseline | +3~6% | 多尺度融合保留细节 |
| 参数效率 | baseline | 相当或更少 | 高效骨干(ConvNeXt) |
| 模态鲁棒性 | baseline | +5~10% | 动态Dropout + 课程学习 |

---

## 五、实现优先级路线图

### Phase 1: Bug修复 + 代码重构 ✅ 已完成

- [x] 修复 fusion_net_v5.py 的 7 个已知 Bug
- [x] 创建 `models/_base.py` 消除代码重复
- [x] 重构 V5/V5EDL 从 890/528 行精简至 350/150 行
- [x] 修复 `models/__init__.py` 的导出
- [x] 提取共享数据集类
- [x] 清理无关文件 (~650MB)
- [x] 31/31 测试通过

### Phase 2: 架构增强 (1-2周) ✅ 已完成 (2026-05-16)

- [x] 可插拔骨干网络 (ResNet50/ConvNeXt-Tiny/EfficientNet-B0/B4)
- [x] 多尺度跨模态融合 (2层: 高级语义CrossModal + 中级MultiScaleFusion)
- [x] CARAFE 上采样

### Phase 3: 训练优化 (2-4周) ✅ 已完成 (2026-05-16)

- [x] 动态时序 Dropout (课程式调度)
- [x] 自适应 KL 退火
- [ ] 层次化时序聚合 (延后)

### Phase 4: 长期

- [ ] 对比学习遥感预训练
- [ ] 半监督学习扩展
- [ ] 跨区域迁移学习
