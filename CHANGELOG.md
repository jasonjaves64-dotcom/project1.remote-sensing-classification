# FusionCropNet 更新日志

## 概述

本项目致力于开发基于深度学习的遥感影像作物分类系统。以下记录了从项目启动至今的所有重要更新和改进。

***

## 目录

1. [版本 5.1 - 代码重构与Bug修复 (2026-05-16)](#版本-51---代码重构与bug修复-2026-05-16)
2. [版本 5.0 - 系统完善与部署](#版本-50---系统完善与部署)
3. [版本 4.0 - EDL-Ensemble 不确定性框架](#版本-40---edl-ensemble-不确定性框架)
4. [版本 3.0 - FusionCropNetV5](#版本-30---fusioncropnetv5)
5. [版本 2.0 - FusionCropNetV4](#版本-20---fusioncropnetv4)
6. [版本 1.0 - FusionCropNet (初始版本)](#版本-10---fusioncropnet-初始版本)

***

## 版本 5.1 - 代码重构与Bug修复 (2026-05-16)

### 代码重复消除

- ✅ 创建 `models/_base.py` 共享组件模块，作为所有基础组件的唯一来源
- ✅ 消除 4 个文件间的重复定义（ConvBNGELU, SEBlock, FiLM, IRB, DEMEncoder, FPN, OpticalEncoder, SAREncoder, TemporalEncoderStream, CrossModalAttention, DEMSpatialConditioner, LateFusion, SpatialRefinement, Decoder, PhenologyAuxHead 共15个组件）
- ✅ `fusion_net_v5_edl.py` 从 890 行减少至 350 行（60%）
- ✅ `fusion_net_v5.py` 从 528 行减少至 150 行（70%）
- ✅ 更新 `dem_encoder.py`, `heads.py`, `temporal.py` 导入基础组件，保留各自特有变体
- ✅ 修复 `models/__init__.py` 导出全部模型版本和共享组件

### V5 关键Bug修复

- ✅ **BUG1 修复**: `consistency_proj` 从 `forward()` 内每次创建移至 `__init__`（内存泄漏）
- ✅ **BUG2 修复**: `CrossModalAttention` 迭代 `.children()` 改为直接 `Sequential` 调用（破坏计算图）
- ✅ **BUG3 修复**: `SpatialRefinement` 全像素自注意 O(n²) 改为窗口化注意力（OOM）
- ✅ **BUG4 修复**: DEM 空间平移新增 `_shift_inputs()` 同步所有模态（数据不一致）
- ✅ **BUG5 修复**: 5D 张量 `F.pad` 维度顺序错误（运行时崩溃）
- ✅ **BUG6 修复**: Bool 张量不支持 `replicate` 填充（运行时崩溃）
- ✅ **BUG7 修复**: `valid_count` 多维度切片索引维度错误（形状错误）

### 跨脚本依赖解耦

- ✅ 提取 `FusionCropDatasetEDL` 从 `scripts/train_fusion_edl.py` 至 `data/datasets/fusion_dataset.py`
- ✅ `train_fusion.py` 和 `train_fusion_edl.py` 统一从 `data.datasets` 导入
- ✅ 移除 `train_fusion_edl.py` 中未使用的导入（nn, Dataset, training_step）

### 项目文件清理

- ✅ 删除约 650MB 构建产物（build/, dist/, test_model.onnx.data）
- ✅ 删除 4 个冗余 `.spec` 文件，保留 `CropClassifier.spec`
- ✅ 删除过时 `.txt` 架构文档（已被 `.md` 版本取代）
- ✅ 更新 `.gitignore` 防止未来文件堆积

### 测试

- ✅ 31/31 单元测试全部通过（19 V5 + 12 EDL）
- ✅ V5EDL 和 V5 烟雾测试通过（训练/推理/不确定性）

***

## 版本 5.2 - FusionCropNetV5Pro (2026-05-16)

### 新模型: FusionCropNetV5Pro

- ✅ `models/fusion_net_v5pro.py` — V5增强版，新增5项关键改进

### 可插拔骨干网络

- ✅ `OpticalEncoder` 支持 ResNet50 / ConvNeXt-Tiny / EfficientNet-B0 / EfficientNet-B4
- ✅ 自动适配骨干的第一层卷积适应多光谱输入(10ch)
- ✅ 兼容 timm FeatureListNet 包装器
- ✅ 配置: `_BACKBONE_CHANNELS` 注册表（`models/_base.py`）

### CARAFE 上采样

- ✅ 新增 `CARAFEUp` 模块 — 基于内容的上下采样，避免转置卷积的棋盘伪影
- ✅ `Decoder` 支持 `use_carafe=True/False` 切换上采样方式

### 多尺度跨模态融合

- ✅ 新增 `MultiScaleFusion` 模块 — H/2×W/2 级别的光学-SAR门控融合
- ✅ 解码器跳跃连接使用融合后的中层特征

### 动态时序 Dropout

- ✅ 课程式调度: `drop_p = 0.05 + 0.15 × sin(π × epoch/total)` — 低→高→低
- ✅ 增强训练鲁棒性，减少过拟合

### 自适应 KL 退火

- ✅ `EDLLoss` 新增 `adaptive=True` 模式
- ✅ 基于 vacuity-error Spearman 相关系数加速/减速KL权重
- ✅ 非自适应模式完全向后兼容

### 训练脚本更新

- ✅ `train_fusion.py` 支持 `--v5pro` 和 `--backbone` 参数
- ✅ `train_fusion_edl.py` 支持 V5Pro 模型创建
- ✅ V5Pro `training_step` 传递 `epoch`, `total_epochs`, `spear_r`

### 支持骨干

| 骨干 | 状态 | 备注 |
|------|------|------|
| resnet50 | ✅ | 默认 |
| convnext_tiny | ✅ | 需128×128以上输入 |
| efficientnet_b0 | ✅ | 参数量更少 |
| swin_tiny | ❌ | Swin特征维度与FPN不兼容 |

### 脚本全面更新 (50个文件)

- ✅ `utils/trainer.py` — FusionTrainer 新增 `_model_forward()` 自动Detect模型类型，V5Pro自动传 `epoch`/`total_epochs`；`fit()` 追踪 `_current_epoch` 供动态Dropout使用
- ✅ `app.py` — 新增 V5Pro 模型加载选项；`load_model_cached()` 支持 `"FusionCropNetV5Pro"` 类型
- ✅ `scripts/predict.py` — 新增 `--v5pro` 和 `--backbone` 参数
- ✅ `scripts/diagnose_model.py` — 新增 `--v5pro` 参数和 V5Pro 模型创建
- ✅ `scripts/train_mil.py` — 修复跨脚本导入 `from scripts.train_fusion_edl` → `from data.datasets`
- ✅ 全部 50 个 .py 文件编译通过 (0 失败)

### 测试

- ✅ 27/32 单元测试通过 (5 个预存问题: test_trainer 使用 V1 PretrainedWeightManager 与 V5 不兼容 + SAR 通道数不匹配)
- ✅ V5Pro 烟雾测试通过 (训练/推理/不确定性，3种骨干)

***

## 版本 1.0 - FusionCropNet (初始版本)

***

## 版本 1.0 - FusionCropNet (初始版本)

**发布日期**: 2026年3月

### 核心功能

- ✅ 光学+SAR双模态融合架构
- ✅ CNN+Transformer混合架构
- ✅ 时序Transformer编码器
- ✅ 跨模态注意力机制

### 技术实现

| 模块     | 说明                  |
| ------ | ------------------- |
| 光学编码器  | ResNet50 + FPN      |
| SAR编码器 | 多层卷积 + IRB模块        |
| 时序编码器  | Transformer编码器 × 4层 |
| 融合方式   | 简单拼接融合              |
| 解码器    | 上采样 + 特征融合          |

### 文件结构

```
project1/
├── models/
│   ├── fusion_net.py      # 核心模型
│   ├── temporal.py        # 时间编码器
│   └── heads.py           # 头部模块
├── scripts/
│   ├── train_fusion.py    # 训练脚本
│   └── predict.py         # 推理脚本
└── utils/
    ├── trainer.py         # 训练框架
    └── losses.py          # 损失函数
```

***

## 版本 2.0 - FusionCropNetV4

**发布日期**: 2026年4月

### 主要改进

#### 🔧 架构优化

- ✅ **DEM融合位置优化**：使用FiLM调制将DEM提前注入时间编码器内部
- ✅ **时间编码器修复**：改为时间维度加权平均，保留时序动态特征
- ✅ **跨模态注意力改进**：增强特征交互能力

#### 🐛 Bug修复

| Bug  | 问题描述                   | 修复方案         |
| ---- | ---------------------- | ------------ |
| BUG1 | 时间编码器只取第一个时间步          | 改为时间维度加权平均   |
| BUG2 | DEM在Decoder后期融合，梯度难以传播 | 使用FiLM调制提前注入 |
| BUG3 | 全遮挡像素使用固定向量            | 添加空间上下文感知    |

#### 📈 性能优化

- ✅ **梯度检查点优化**：44个梯度检查点，显存占用减少约50%
- ✅ **混合精度训练**：支持FP16训练，加速训练过程

### 新增模块

- `dem_encoder.py` - DEM编码器
- `unet_transformer.py` - UNet-Transformer新架构

***

## 版本 3.0 - FusionCropNetV5

**发布日期**: 2026年5月

### 主要改进

#### 🔧 架构升级

- ✅ **三模态融合**：光学+SAR+DEM完整融合
- ✅ **窗口注意力**：解决全局自注意力OOM问题
- ✅ **改进的时序建模**：支持云掩膜加权，自动跳过被遮挡时间步

#### 🐛 Bug修复

| Bug  | 问题描述                                  | 修复方案            |
| ---- | ------------------------------------- | --------------- |
| BUG1 | consistency\_proj在forward中重复创建        | 移到\_\_init\_\_中 |
| BUG2 | CrossModalAttention中不必要的.children()迭代 | 移除冗余迭代          |
| BUG3 | SpatialRefinement使用全局自注意力导致OOM        | 改为窗口注意力         |

#### 📁 文件更新

```
models/
├── fusion_net_v5.py        # V5核心模型
└── fusion_net_v5_edl.py    # V5 + EDL（后续版本添加）
```

***

## 版本 4.0 - EDL-Ensemble 不确定性框架

**发布日期**: 2026年5月

### 核心功能

#### 🎯 EDL-Ensemble框架

- ✅ **EDLHead**：证据深度学习头部，输出Dirichlet分布参数
- ✅ **EDLLoss**：结合交叉熵和KL正则化的损失函数
- ✅ **三种不确定性度量**：
  - **Vacuity**：数据不确定性（证据不足程度）
  - **Dissonance**：认知不确定性（类间证据冲突）
  - **Class Variance**：每类预测方差

#### 🔧 推理增强

- ✅ **predict\_uncertainty()**：支持多轮推理和TTA
- ✅ **证据级融合**：Dirichlet参数平均融合策略

#### 🐛 EDL相关Bug修复

| Bug  | 问题描述                  | 修复方案                   |
| ---- | --------------------- | ---------------------- |
| EDL1 | KL损失惩罚正确类             | 修改alpha\_tilde计算方式     |
| EDL2 | 预训练骨干被重新初始化           | 正确跳过所有子模块              |
| EDL3 | BN在推理时处于训练模式          | 只启用dropout，保持BN在eval模式 |
| EDL4 | CE损失重复计算              | 从总损失中减去原始CE            |
| EDL5 | DEM位移与标签不同步           | 同步所有模态的位移              |
| EDL6 | consistency\_loss缺乏监督 | 添加ground truth学习目标     |

***

## 版本 5.0 - 系统完善与部署

**发布日期**: 2026年5月

### 系统功能

- ✅ **GeoTIFF直接读取**：支持直接处理原始遥感影像
- ✅ **混合精度训练**：FP16训练支持，加速训练过程
- ✅ **SQL数据库集成**：训练记录、不确定性指标存储、实验管理
- ✅ **FastAPI服务**：RESTful API接口，支持请求限流和API认证
- ✅ **监控与日志**：多级别日志、性能监控、错误告警
- ✅ **Web应用 (Streamlit)**：可视化推理、不确定性热力图
- ✅ **桌面应用 (PyQt)**：离线推理、结果导出
- ✅ **ArcGIS集成**：支持EMD/DLPK格式导出

***

## 版本 6.0 — EDL不确定性校准验证 & 模型可解释性

**发布日期**: 2026年5月14日

### 概述

本版本新增两大核心分析能力：
1. **EDL不确定性校准验证体系** — 从统计严格性角度评估EDL输出的不确定性质量
2. **模型可解释性分析框架** — 多维度拆解模型决策依据

新增文件 2 个，更新文件 12 个，新增测试用例 16 个。

***

### 一、EDL不确定性校准验证 (`utils/calibration.py`)

#### 1.1 设计动机

EDL模型输出Dirichlet分布的参数α，并从中导出不确定性度量（vacuity, dissonance）。
然而，这些不确定性是否**真正反映了预测质量**，需要通过严格校准指标来验证。
一个"校准良好"的模型应满足：高置信度对应高准确率，高不确定性对应高错误率。

#### 1.2 校准指标体系

##### 期望校准误差 (Expected Calibration Error, ECE)

实现两种变体：
- **等宽分箱ECE** (`expected_calibration_error`)：将[0,1]均匀划分为N个bin，计算每bin内的|置信度均值 - 准确率均值|按bin样本量加权求和。这是Guo et al. (2017)的标准实现。
- **等质量分箱ECE** (`adaptive_ece`)：每bin包含相同数量的样本，对偏斜置信度分布更鲁棒。
- **最大校准误差MCE** (`maximum_calibration_error`)：取所有bin中|conf - acc|的最大值，衡量最坏情况。

##### Dirichlet专用评分规则 (Proper Scoring Rules)

- **Dirichlet NLL** (`negative_log_likelihood_dirichlet`)：
  ```
  NLL = -E_q[log p(y|x; α)]
      = -log(α_y / S) + KL(post || prior)
  ```
  衡量Dirichlet分布与真实标签的拟合优度。

- **Dirichlet Brier Score** (`brier_score_dirichlet`)：
  ```
  Brier = E[(p - one_hot)²]  where p_k = α_k / S
  ```
  多类Brier Score的Dirichlet推广。

##### 预测锐度与证据分散度

- **Sharpness**：Var(confidence)，置信度方差，越高表示预测越"锐利"（极端化）。
- **Dispersion**：mean(1/(1+S))，S = Σα_k为总证据强度。低分散度表示模型输出强证据，高分散度表示弱证据。

##### 不确定性-错误相关性

- **Spearman ρ** (`uncertainty_error_correlation`)：vacuity与预测错误之间的排序相关性。正相关且显著表明不确定性有效跟踪了错误。
- **AUROC误差检测** (`uncertainty_auroc`)：将vacuity视为"错误检测得分"（越高越可能错误），计算ROC-AUC。>0.5即优于随机。
- **PR-AUC** (`uncertainty_pr_auc`)：精度-召回曲线下面积，更适合类别不平衡场景。

##### OOD/错误检测

- **分位数检测** (`ood_detection_metrics`)：按vacuity的给定分位数阈值标记"可疑"预测，计算Precision/Recall/F1。
- **拒绝曲线** (`uncertainty_rejection_curve`)：按vacuity升序保留最确定像素，绘制准确率-保留率曲线。理想情况下，保留率下降时准确率应上升。

#### 1.3 完整校准报告

`calibration_report()` 函数接收Dirichlet alpha和真实标签，输出包含所有上述指标的字典，
以及每类校准分解（逐类ECE、准确率、置信度、vacuity均值）和原始数据数组。

`print_calibration_report()` 提供格式化控制台输出。

#### 1.4 数学原理

EDL框架下，模型对每个像素输出证据向量e_k ≥ 0，Dirichlet参数α_k = e_k + 1。
总证据强度S = Σα_k = K + Σe_k。vacuity = K/S衡量"所见证据不足"，
dissonance = 1 - Σp_k²衡量"所见证据互相矛盾"。

校准验证的核心问题是：**vacuous预测是否真的更可能出错？dissonant预测是否处于类别边界？**
上述指标从不同角度回答这个问题。

***

### 二、模型可解释性分析 (`utils/interpretability.py`)

#### 2.1 Grad-CAM空间归因

`GradCAM_EDL` 将Grad-CAM适配到EDL输出空间：
- 目标信号：对目标类k，使用α_k的总和作为反向传播目标
- 激活层：可配置的目标层（默认decoder）
- 权重计算：梯度全局平均池化后与激活图做加权和
- 输出：与输入同分辨率的[0,1]热力图

`gradcam_per_class()` 循环生成全部K个类别的Grad-CAM热力图。

#### 2.2 模态消融分析

`modality_ablation()` 通过系统性地移除每个模态来量化其贡献：
- 对{full, no_opt, no_sar, no_dem}四种配置分别推理
- 测量与完整模型的预测一致性(agreement)、概率分布偏移(KL散度)、vacuity变化
- 输出归一化的相对重要性：optical + sar + dem = 1.0

该分析的工程意义在于判断：
1. 是否存在"搭便车"模态（移除后几乎无影响）
2. 各模态间是否存在冗余
3. 特定场景下是否可以只用单模态降本

#### 2.3 时序重要性分析

`temporal_importance()` 逐个时间步遮挡（置零），测量预测翻转率，输出归一化的T维重要性向量。
`temporal_entropy_analysis()` 分析每个单独时间步对预测熵的降低程度。

#### 2.4 光谱波段重要性

`spectral_band_importance()` 使用逐波段遮挡法：
- 逐个将光学10波段和SAR 5波段置零
- 测量预测变化比例
- 输出归一化重要性得分

光学波段：B2_Blue, B3_Green, B4_Red, B8_NIR, NDVI, NDWI, EVI, LSWI, BSI, NBR
SAR波段：VV, VH, VV/VH, RVI, NLI

#### 2.5 跨模态注意力分析

`cross_modal_attention_analysis()` 钩取CrossModalAttention模块的门控值和输出范数：
- gate值反映光学→SAR的融合权重空间分布
- 输出范数反映融合后的特征激活强度

#### 2.6 混淆区域分析

`confusion_region_analysis()` 识别模型最困惑的类别对：
- 对每对(i,j)，找到true=i但runner-up=j的像素（及反向）
- 统计这些像素的平均vacuity和dissonance
- 高vacuity + 高dissonance通常表示"真混淆"（两类确实难以区分）

#### 2.7 像素级解释报告

`pixel_explanation_report()` 分别统计正确预测和错误预测像素的：
- vacuity、dissonance、置信度、top-2 margin、熵
- 输出mean/std/median/p10/p90分位数

理想情况：错误像素应有显著更高的vacuity和更低的margin。
若错误像素的vacuity与正确像素无差异，说明不确定性估计失效。

#### 2.8 Integrated Gradients（可选）

`integrated_gradients_attribution()` 通过Captum库的IntegratedGradients实现全特征归因。
需要 `pip install captum`。

***

### 三、可视化增强 (`scripts/visualize.py`)

新增 10 个可视化函数：

| 函数 | 输出 | 用途 |
|------|------|------|
| `plot_reliability_diagram()` | 可靠性图 + 置信度直方图 | 校准质量一目了然 |
| `plot_uncertainty_error_map()` | 不确定性-错误覆盖图 | 空间上的不确定性vs错误对应关系 |
| `plot_rejection_curve()` | 拒绝曲线 | 不确定性的实用价值评估 |
| `plot_per_class_calibration()` | 每类校准柱状图 | 识别特定类别的校准问题 |
| `plot_gradcam_heatmaps()` | 逐类Grad-CAM网格 | 各类别的空间决策依据 |
| `plot_modality_contribution()` | 模态贡献对比图 | 消融分析结果可视化 |
| `plot_temporal_importance()` | 时序重要性柱状图 | 关键时间窗口识别 |
| `plot_band_importance()` | 波段重要性柱状图 | 光谱通道筛选 |
| `plot_pixel_explanation()` | 正确vs错误对比图 | 不确定性区分度可视化 |
| `run_full_analysis()` | 一次性运行全部分析 | 一键生成完整分析报告 |

`run_full_analysis()` 函数执行6步完整pipeline：
1. 推理获得alpha和vacuity/dissonance
2. 计算校准指标并绘图
3. 像素级解释分析
4. 模态消融
5. 时序重要性
6. 波段重要性
最终将JSON格式报告保存到输出目录。

***

### 四、脚本更新清单

#### 新文件
| 文件 | 说明 |
|------|------|
| `utils/calibration.py` | EDL校准验证（ECE/NLL/Brier/OOD检测/拒绝曲线） |
| `utils/interpretability.py` | 模型可解释性（Grad-CAM/消融/时序/波段/混淆/像素分析） |

#### 更新文件
| 文件 | 更新内容 |
|------|---------|
| `models/fusion_net_v5_edl.py` | 修复`_encode`中use_opt=False时的ConvBNGELU下标错误 |
| `scripts/visualize.py` | 新增10个可视化和run_full_analysis() |
| `scripts/train_fusion_edl.py` | 每epoch校准日志，训练结束完整校准+可解释性报告 |
| `scripts/train_fusion.py` | EDL模式训练后自动输出校准报告 |
| `scripts/predict.py` | 新增`--calibration`和`--interpretability`参数 |
| `scripts/test_full_system.py` | 新增test_calibration_validation()和test_interpretability() |
| `scripts/diagnose_model.py` | 新增check_uncertainty_calibration()和check_interpretability() |
| `scripts/test_predict.py` | 新增`--calibration`参数 |
| `scripts/test_train_with_plot.py` | 训练历史增加ECE/NLL追踪和校准子图 |
| `scripts/export_edl.py` | 导出时计算校准基线并嵌入EMD文件 |
| `scripts/sql_train_integration_edl.py` | 每epoch记录ECE/NLL/Brier到数据库 |
| `sql/db_utils_edl.py` | add_uncertainty_metrics()新增ece/nll/brier参数 |
| `sql/update_schema_edl.py` | uncertainty_metrics表新增ece/nll/brier列 |
| `test_model_bug.py` | 新增校准验证和可解释性检查 |
| `tests/test_suite.py` | 新增TestEDLCalibration(8项)和TestModelInterpretability(8项) |

***

### 五、技术细节

#### 为什么需要校准验证

EDL的vacuity和dissonance基于证据理论的数学推导，但在实际训练中：
1. KL退火强度λ的选择会影响不确定性尺度
2. 模型可能在ID数据上校准良好，在OOD/分布外数据上失效
3. 不同类别的校准质量可能差异很大
4. vacuity作为"错误指示器"的实际区分能力需要量化验证

ECE和拒绝曲线提供了与模型准确率无关的独立校准质量评估。

#### 为什么需要可解释性

遥感作物分类的应用场景要求：
1. **农业决策可追溯**：农学家需要知道模型依据什么判断
2. **错误诊断**：当模型出错时，需要知道是哪个模态/波段/时间步导致的
3. **特征工程**：波段重要性分析指导数据采集策略
4. **模型可信度**：Grad-CAM直观展示模型是否关注了合理的空间区域

#### 关键实现考量

1. **ECELoss梯度截断**：校准报告仅在eval模式下计算，不参与训练。原因是ECE本身不可微，无法直接作为训练损失。

2. **消融中的占位策略**：移除某模态时使用零张量+modality_mask参数，而非直接修改模型结构。`_encode`通过placeholder参数处理缺失模态。

3. **Grad-CAM与EDL的适配**：标准Grad-CAM对分类logits求导。在EDL中，α_k（而非p_k）既是可微输出也是Dirichlet参数。使用α_k的总和作为目标信号保持了梯度流通过证据网络。

4. **Bayesian校验的一致性**：vacuity = K/S可以用Bayesian视角理解——
   令先验为Uniform Dirichlet(1,1,...,1)，则prior evidence = K，
   观测evidence = Σe_k，后验总evidence = K + Σe_k = S，
   vacuity = prior_evidence / posterior_evidence，即"先验的相对权重"。
   高vacuity = 数据提供的evidence不足以压倒先验。

5. **时序重要性的累积效应**：单个时间步单独使用和联合使用的信息量不等价。
   遮挡法测量的是"联合场景中缺失该步的边际损失"，更能反映时间步的实际贡献。

***

### 数据预处理管道

- ✅ **GeoTIFF直接读取**：支持直接处理原始遥感影像
- ✅ **SAR对数变换**：先做log(1+x)变换再标准化
- ✅ **全局统计量归一化**：使用预计算的全局均值和标准差
- ✅ **云掩码跟随插值**：标记所有插值位置为无效
- ✅ **长时间空缺屏蔽**：超过阈值的空缺完全标记为无效

### API服务封装

- ✅ **FastAPI服务**：RESTful API接口
- ✅ **请求限流**：防止API滥用
- ✅ **API认证**：API密钥验证
- ✅ **异步任务处理**：支持后台训练任务

### 监控与日志系统

- ✅ **多级别日志**：DEBUG/INFO/WARNING/ERROR/CRITICAL
- ✅ **结构化日志**：JSON格式输出
- ✅ **性能监控**：推理时间、内存、GPU使用
- ✅ **错误告警**：邮件和钉钉告警支持
- ✅ **日志搜索**：支持关键词搜索

### 应用界面

- ✅ **Web应用 (Streamlit)**：可视化推理、不确定性热力图
- ✅ **桌面应用 (PyQt)**：离线推理、结果导出
- ✅ **ArcGIS集成**：支持EMD/DLPK格式导出

***

## 技术亮点汇总

### 1. 时序动态特征建模

- 时间编码器正确聚合所有时间步特征
- 支持云掩膜加权，自动跳过被遮挡的时间步

### 2. DEM几何约束

- 使用FiLM调制技术将DEM作为条件信息注入
- 每一层Transformer都受DEM几何特征约束

### 3. 内存优化

- 梯度检查点技术，显存占用减少约50%
- 支持更大batch\_size和更高分辨率输入

### 4. 多模态融合

- 光学、SAR、DEM三模态协同
- 跨模态注意力机制增强特征交互

### 5. 不确定性估计

- EDL-Ensemble框架提供可靠的不确定性度量
- 支持数据不确定性和认知不确定性分离

### 6. 工程化部署

- 完整的API服务和监控系统
- 支持多种部署方式（Web、桌面、ArcGIS）

***

## 项目状态

| 模块                 | 状态    |
| ------------------ | ----- |
| FusionCropNetV5EDL | ✅ 完成  |
| EDL-Ensemble框架     | ✅ 完成  |
| 数据预处理管道            | ✅ 完成  |
| API服务              | ✅ 完成  |
| 监控与日志              | ✅ 完成  |
| Web应用              | ✅ 运行中 |
| ArcGIS集成           | ✅ 完成  |

***

## 下一步计划

- [ ] 数据集准备与训练
- [ ] 模型性能评估与对比
- [x] 不确定性估计效果验证 (← v6.0校准验证模块)
- [x] 模型可解释性分析 (← v6.0可解释性模块)
- [ ] 论文撰写与投稿

***

*更新日志最后更新于: 2026年5月14日*
