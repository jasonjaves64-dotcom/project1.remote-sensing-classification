# FusionCropNet V6 设想与实现技术日志

**日期**: 2026-05-19
**状态**: 设计阶段
**前代**: V5Pro (MIL + HPO + CARAFE + EDL Ensemble)

---

## 一、项目当前状态审计

| 项目 | 状态 |
|------|------|
| 测试 | 105 passed, 0 failed |
| 代码 | `_base.py` 统一管理，无重复 |
| V5Pro | MIL + HPO + 多尺度 + CARAFE + 可插拔骨干 |
| 已知缺陷 | TransformerEncoder norm_first 警告（无害） |
| GPU | CPU only（当前环境） |

---

## 二、SOTA 竞品调研（2026-05-19）

### 2.1 参比模型

| 模型 | 来源 | 核心方法 | 性能亮点 |
|------|------|---------|---------|
| **TempCNN + Feature Fusion** | Mena et al. (JAG 2025) | 1D-CNN + 多视图特征融合 | OA=86.2% (CropHarvest 全球) |
| **SpectralEarth-ViT** | 多机构 (2025) | 53.8万高光谱patch预训练 | OA=93.5% (cereal) |
| **CropSTS** | Yan et al. (RS 2025) | 解耦时空注意 + JEPA 蒸馏 | SOTA PASTIS-R |
| **SITS-Siam** | Pinto et al. (CEA 2025) | 非对比孪生 SSL | 3 数据集 SSL 最优 |
| **CropNet** | Li et al. (JSTARS 2025) | GLPA + ECFEM 轻量分割 | 参数最少精度最高 |

### 2.2 功能对比速览

| 功能 | V5Pro | TempCNN | SpectralEarth | CropSTS | SITS-Siam | CropNet |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 三模态 (Opt+SAR+DEM) | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| EDL 不确定性 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SSL 预训练 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| 基础模型骨干 | ❌ | ❌ | ✅ ViT-G | ✅ ViT-S | ❌ | ❌ |
| 缺失模态鲁棒 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 跨模态注意 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 层次化时序 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| 多任务 | ✅ NDVI | ❌ | ❌ | ❌ | ❌ | ❌ |
| 边界增强 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ ECFEM |
| OSM 先验 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

### 2.3 竞品启发

| 启发 | 来源 | 引入价值 |
|------|------|---------|
| 解耦时空注意 (时间优先) | CropSTS | 更精确捕获物候动态 |
| JEPA 自监督预训练 | CropSTS | 小型 ViT-S 即可达 SOTA |
| 知识蒸馏 (DINO → 遥感) | CropSTS | 利用 web-scale 模型先验 |
| ECFEM 边界增强 | CropNet | 田块边界更清晰 |
| ERA5 气象融合 | TempCNN | SAR 后向散射受天气影响，气象可校正 |
| SSL 对比预训练 | SITS-Siam | 少量标注即可微调 |

---

## 三、V5Pro 剩余瓶颈（V6 设计动机）

| 瓶颈 | 严重度 | 说明 |
|------|--------|------|
| 逐像素时序编码 O(n²) | 高 | B×H²×W² 条序列，64×64 特征图 → 4096 条序列，每个 T×512 |
| 无遥感域预训练 | 高 | 依赖 ImageNet 预训练，遥感域迁移有限 |
| 时序建模单尺度 | 中 | 仅像素级，缺乏 patch/region/场景三级层次 |
| 交叉注意仅在顶层 | 中 | opt↔sar 融合仅在最深特征层（1/4 分辨率） |
| DEM 利用浅 | 中 | 仅在 FiLM 和后融合中使用，未参与时序/注意 |
| 无多任务学习 | 中 | 仅 NDVI 辅助，可扩展至 LAI/生物量/土壤类型 |
| 无主动学习 | 低 | 无法识别高价值未标注样本 |
| 场景级理解缺失 | 低 | 逐像素分类缺乏对田块/场景的整体理解 |

---

## 三、V6 核心设想

### 3.1 总体架构：层次化基础模型 + 多任务

```
┌─────────────────────────────────────────────────────────┐
│                  FusionCropNet V6                         │
│  "Hierarchical Foundation Model with Multi-Task Heads"   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Layer 3: SCENE (1 per image)                             │
│  ┌──────────────────────────────────────────┐            │
│  │  Scene Encoder (ViT-G/GeoFM pretrained)  │            │
│  │  → scene_token: 场景级作物分布先验       │            │
│  └──────────────┬───────────────────────────┘            │
│                 │ cross-attention                         │
│  Layer 2: REGION (patch-based)               │            │
│  ┌──────────────────────────────────────────┐            │
│  │  Swin/W-MSA Region Encoder               │            │
│  │  → region_tokens: 田块级特征              │            │
│  └──────────────┬───────────────────────────┘            │
│                 │ cross-attention                         │
│  Layer 1: PIXEL (per-pixel, inherited from V5)│          │
│  ┌──────────────────────────────────────────┐            │
│  │  Temporal Encoder (V5Pro optimized)       │            │
│  │  → pixel_tokens: 逐像素时序特征           │            │
│  └──────────────────────────────────────────┘            │
│                                                           │
│  Multi-Task Heads:                                        │
│  ├── Crop Type (main)                                     │
│  ├── LAI / Biomass                                       │
│  ├── Growth Stage (phenology)                             │
│  ├── Uncertainty (EDL)                                    │
│  └── Field Boundary (aux)                                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 关键技术特性

#### ① 层次化时序建模（Layer 1-2-3）

| 层级 | 分辨率 | 编码器 | 时序策略 |
|------|--------|--------|---------|
| Scene | 1 token/image | ViT-G / GeoFM 冻结 | 时序池化 + 场景分类头 |
| Region | 8×8 patches | Swin Transformer W-MSA | Patch 时序注意 (O(T²) per patch) |
| Pixel | 逐像素 | V5Pro 优化版 | 窗口化像素注意 (O(T²) per window) |

**创新点**: Region 层共享像素编码输出 → 减少逐像素序列数 → 计算量从 O(H²W²×T²) 降至 O(K²×T² + H²W²×T)（K=patch 数）

#### ② 遥感基础模型预训练

**候选预训练模型**（更新于 SOTA 调研后）：

| 模型 | 预训练数据 | 方法 | 适用模态 | 可用性 |
|------|-----------|------|---------|--------|
| **SpectralEarth-ViT** | 53.8万 EnMAP 高光谱 patches | 监督预训练 | 高光谱 | 公开 |
| **SSL4EO-S12** | 25万 Sentinel-2 patches | SSL 对比 | 多光谱 S2 | 公开 |
| **CropSTS** | Sentinel-2 PASTIS | JEPA + 蒸馏 | 多光谱 S2 | 公开 |
| **SatMAE** | fMoW 多时相 | 掩码自编码器 | 多光谱 | 公开 |
| **GeoFM** | 遥感多源 | 基础模型 | 多模态 | 待确认 |

**集成策略**: 冻结骨干 + LoRA 适配器，仅训练 Adapter + 时序 + 融合 + 解码器
**预期收益**: 
- 标签效率 2-3×（CropSTS 已验证：小型 ViT-S + 少样本 → PASTIS-R SOTA）
- 跨区域泛化 +5-10% OA（SpectralEarth-ViT: 93.5% vs 62.6% 无预训练）

#### ②-bis 备选：解耦时空注意 (Decoupled Spatiotemporal Attention)

CropSTS 2025 验证了一个关键洞察：**遥感影像的空间坐标在时序上不变**（不同于视频），因此时间注意和空间注意可以分开建模：

```
传统: [T×H×W, C] → Joint Attention → O((T×H×W)²)
CropSTS: [T, H×W, C] → Temporal Attention → Spatial Attention → O(T² + (H×W)²)
```

- **时间优先**: 先沿时间轴做注意（捕获物候动态）→ 再沿空间轴做注意（捕获空间上下文）
- **可学习缩放系数** γₜ, γₛ 自动平衡时-空权重
- **V6 选项**: 作为 `TemporalEncoderStream` 的替代方案，可大幅减少时序编码计算量

#### ③ 多尺度交叉注意（Deep Fusion）

V5Pro 仅在顶层 (feat_dim, H/4, W/4) 做 opt↔sar 注意。
V6 扩展到三层：

```
opt_fpn_p2 (H/2) ←→ sar_s1 (H/2)   [浅层：纹理融合]
opt_fpn_p3 (H/4) ←→ sar_s2 (H/4)   [中层：结构融合]
opt_fpn_p4 (H/4) ←→ sar_s3 (H/4)   [深层：语义融合]  ← V5Pro 已有
```

#### ④ 多任务学习

| 任务 | 头 | 损失 | 权重 |
|------|-----|------|------|
| 作物分类 | EDL Head | EDL Loss | 1.0 |
| LAI/生物量回归 | MLP Head | Huber Loss | 0.3 |
| 生长阶段 | 分类 Head | CrossEntropy | 0.2 |
| 田块边界 | UNet 边 Head | Dice Loss | 0.1 |
| NDVI 预测 | PhenologyAux | MSE | 0.1（退火） |

共用特征提取器，多任务梯度提升泛化能力。

#### ⑤ 主动学习支持

- **不确定性采样**: 选择 vacuity 最高（最不确定）的像素/样本
- **多样性采样**: Core-set 选择覆盖特征空间的子集
- **输出**: 一个 `labeling_priority.tif` 栅格 → ArcGIS 直接叠加到影像上标注

#### ⑥ 内存优化

| 技术 | 说明 |
|------|------|
| 梯度检查点 (Gradient Checkpointing) | 时序编码器反向传播时重算中间激活 |
| 混合精度 (AMP) | FP16 训练，减少 40% 显存 |
| 分块推理 (Tile Inference) | 大影像自动切块 + 重叠 + 拼接 |
| Flash Attention | Transformer 加速，O(N²)→O(N) 内存 |

---

## 四、与 V5Pro 的兼容性

| 组件 | V5Pro | V6 | 兼容 |
|------|-------|-----|------|
| `_base.py` 共享组件 | 使用 | 扩展（加 RegionEncoder, DeepCrossModalAttention） | ✅ 追加 |
| EDL Head | 使用 | 使用 | ✅ 直接复用 |
| OpticalEncoder | ResNet/ConvNeXt/EfficientNet | + ViT / GeoFM 冻结 | ✅ 追加 |
| SAREncoder | IRB + FiLM | 保留 | ✅ 不变 |
| TemporalEncoderStream | 逐像素 Transformer | 窗口化优化 | ✅ 替换 |
| CARAFE Decoder | 使用 | 使用 | ✅ 不变 |
| Multi-task heads | 仅 NDVI aux | 4 个新 head | ✅ 追加 |
| 训练策略 | 两阶段 | 三阶段（预训练→多任务→微调） | ✅ 扩展 |

---

## 五、实现路线图

### Phase 1: 内存与速度优化（1-2 周）
- [ ] 窗口化像素注意（Flash Attention 加速）
- [ ] 梯度检查点 + AMP 混合精度
- [ ] 分块推理脚本

### Phase 2: 层次化时序（2-3 周）
- [ ] Region Encoder (Swin W-MSA)
- [ ] Scene Encoder（ViT-G 冻结 + LoRA）
- [ ] 三层交叉注意级联

### Phase 3: 预训练集成（2-3 周）
- [ ] GeoFM / SatMAE 预训练权重加载
- [ ] LoRA 微调适配器
- [ ] 域自适应训练策略

### Phase 4: 多任务学习（1-2 周）
- [ ] LAI/生物量回归头
- [ ] 生长阶段分类头
- [ ] 田块边界检测头
- [ ] 多任务损失平衡策略（Uncertainty Weighting）

### Phase 5: 主动学习（1-2 周）
- [ ] 不确定性 + 多样性采样
- [ ] labeling_priority.tif 输出
- [ ] ArcGIS 标注流程集成

### Phase 6: 文档与测试（1 周）
- [ ] 完整测试套件
- [ ] 训练教程 + 使用文档

---

## 六、风险与对策

| 风险 | 概率 | 对策 |
|------|------|------|
| 预训练模型不可用 | 中 | 回退到 ImageNet 预训练 + 遥感域微调 |
| 多任务梯度冲突 | 低 | Uncertainty Weighting 动态调整 |
| 三层交叉注意内存爆炸 | 中 | Gradient Checkpointing + 量化 |
| CPU only 训练太慢 | 高 | 优先实现分块推理，训练建议用 GPU |
| 标注数据不足 | 中 | 主动学习 + 半监督学习 |

---

## 七、预期量化收益

| 指标 | V5Pro 当前 | V6 目标 | 提升 |
|------|-----------|---------|------|
| mIoU (作物分类) | ~0.15 (合成) | ~0.30+ (合成) | 2× |
| 训练速度 | 1× | 1.5-2× | 梯度检查点+AMP |
| 推理内存 | 1× | 0.4× | 分块推理 |
| 标签效率 | 1× | 2-3× | 预训练+主动学习 |
| 支持任务数 | 1 (分类) | 5 (分类+LAI+生长+边界+NDVI) | 5× |

---

## 八、竞品对标总结

### 8.1 功能覆盖雷达图（文本版）

```
                    FusionCropNet V5Pro  TempCNN+FF  SpectralEarth  CropSTS
多模态融合              ████████████         ████████      ████          ████
时序建模                ████████             ██████        ██████        ████████████
不确定性量化            ████████████         ██            ██            ██
泛化能力                ██████               ████████      ████████████  ████████
训练效率                ██████               ████████      ████          ██████
生产就绪                ████████             ██████        ████          ████
```

### 8.2 V6 完成后预期功能覆盖

```
                    V5Pro 当前    V6 目标    差距补齐
多模态融合              ████████████  ████████████  —
时序建模                ████████      ████████████  ↑ (层次化)
不确定性量化            ████████████  ████████████  —
泛化能力                ██████        ████████████  ↑ (SSL预训练)
训练效率                ██████        ████████████  ↑ (AMP+GC)
生产就绪                ████████      ████████████  ↑ (多任务)
```

---

## 九、V6 更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-05-19 (初版) | V6 架构初稿：层次化时序 + 多任务 + SSL |
| 2026-05-19 (更新) | 新增 SOTA 竞品调研（6 模型对比） |
| 2026-05-19 (更新) | 新增解耦时空注意备选方案 |
| 2026-05-19 (更新) | 更新预训练模型候选列表（SpectralEarth-ViT, SSL4EO-S12, CropSTS） |

---

*日志保存于 `docs/superpowers/specs/2026-05-19-fusion-v6-design.md`*
*竞品对比详表：`docs/superpowers/specs/2026-05-19-model-comparison.md`*
