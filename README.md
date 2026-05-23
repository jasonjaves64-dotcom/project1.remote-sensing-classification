# 基于深度学习的遥感影像光谱分类

## 📋 项目简介

本项目实现了一个基于深度学习的多模态遥感影像农作物分类系统，融合光学影像（Landsat）、SAR影像（Sentinel-1）和DEM地形数据进行精准作物分类。系统集成了**EDL-Ensemble不确定性估计框架**，提供可靠的决策支持。

## 🔥 最新特性

- **FusionCropNetV5Pro**: 最新旗舰模型，集成MIL-Module + Multi-Head + HPO + 校准
- **FusionCropNetV5EDL**: EDL-Ensemble版本，支持三模态融合 + 不确定性估计
- **EDL-Ensemble不确定性估计**: 提供数据不确定性(Vacuity)和认知不确定性(Dissonance)度量
- **ArcGIS集成**: 支持导出为EMD/DLPK格式，可在ArcGIS Pro中部署
- **SQL数据库集成**: 训练记录和不确定性指标存储

## 🚀 一键部署

### 环境要求
- Python 3.8+
- 支持 GPU（推荐）或 CPU

### 安装步骤

1. **进入项目目录**
   ```bash
   cd project1
   ```

2. **一键安装**
   ```bash
   python install.py
   ```

3. **启动应用**
   ```bash
   python start.py
   ```

4. **打开浏览器**
   - 应用启动后会自动打开浏览器，默认地址：`http://localhost:8501`

## 📁 项目结构

```
project1/
├── app.py                   # Streamlit 前端应用（支持EDL）
├── install.py               # 一键安装脚本
├── start.py                 # 一键启动脚本
├── config.yaml              # 配置文件（含EDL参数）
├── pyproject.toml           # 包配置
├── README.md                # 项目文档
├── PROJECT_PROGRESS.md      # 项目进展报告
├── gee_setup_guide.md       # GEE配置指南
├── data/                    # 数据预处理模块
│   ├── datasets/            # 数据集定义
│   ├── preprocessing/       # 预处理子模块
│   └── ...
├── models/                  # 模型定义
│   ├── fusion_net.py        # 融合网络模型（基准版）
│   ├── fusion_net_v4.py     # V4版本
│   ├── fusion_net_v5.py     # V5基础版本
│   ├── fusion_net_v5_edl.py # V5版本（EDL-Ensemble）
│   ├── fusion_net_v5pro.py  # V5Pro旗舰版本（MIL + HPO）
│   ├── unet_transformer.py  # UNet-Transformer架构
│   └── ...
├── utils/                   # 工具模块
│   ├── config.py            # 配置管理
│   ├── logger.py            # 日志系统
│   ├── trainer.py           # 训练器
│   ├── losses.py            # 损失函数（含EDLLoss）
│   └── ...
├── scripts/                 # 运行脚本
│   ├── train_fusion.py      # 训练脚本（支持--edl）
│   ├── predict.py           # 推理脚本（支持不确定性估计）
│   ├── train_fusion_edl.py  # EDL专用训练脚本
│   ├── export_to_arcgis.py  # ArcGIS格式导出
│   └── ...
├── sql/                     # 数据库模块（支持EDL指标）
├── api/                     # API服务（FastAPI）
├── tests/                   # 单元测试
└── logs/                    # 日志文件
```

## 📊 功能特性

| 功能 | 说明 |
|------|------|
| **模型配置** | 支持自定义模型路径和推理参数 |
| **数据上传** | 支持上传光学、SAR时序数据和DEM数据 |
| **一键推理** | 点击按钮即可进行预测 |
| **不确定性估计** | EDL-Ensemble框架提供可靠的不确定性度量 |
| **结果可视化** | 显示分类图、不确定性热力图和统计图表 |
| **ArcGIS导出** | 支持导出为EMD/DLPK格式 |

## 🛠️ 技术栈

| 分类 | 技术 |
|------|------|
| **框架** | PyTorch 2.0+ |
| **前端** | Streamlit |
| **遥感处理** | rasterio, GDAL |
| **可视化** | matplotlib |
| **数据库** | SQLite/MySQL |
| **API** | FastAPI |

## 🧠 模型架构

### FusionCropNetV5Pro (最新旗舰)

```
┌─────────────────────────────────────────────────────────────┐
│                   FusionCropNetV5Pro                        │
├─────────────────────────────────────────────────────────────┤
│  输入层                                                     │
│  ├── 光学时序 [B, T, 10, H, W]                            │
│  ├── SAR时序 [B, T, 5, H, W]                              │
│  ├── DEM [B, 5, H, W]                                     │
│  └── DOY [B, T]                                           │
├─────────────────────────────────────────────────────────────┤
│  编码器层                                                   │
│  ├── OpticalEncoder (ResNet50 + FPN)                      │
│  ├── SAREncoder (IRB + FiLM)                              │
│  └── DEMEncoder                                           │
├─────────────────────────────────────────────────────────────┤
│  MIL-Module (多实例学习)                                    │
│  ├── Instance Aggregation                                 │
│  ├── Attention Pooling                                    │
│  └── Bag-Level Classification                             │
├─────────────────────────────────────────────────────────────┤
│  Multi-Head 输出                                           │
│  ├── Classification Head                                  │
│  ├── Uncertainty Head (EDL)                               │
│  └── Calibration Head                                     │
├─────────────────────────────────────────────────────────────┤
│  HPO + Calibration                                         │
│  ├── Optuna超参数搜索                                      │
│  ├── 温度缩放校准                                          │
│  └── ECE期望校准误差                                        │
├─────────────────────────────────────────────────────────────┤
│  输出                                                       │
│  ├── 分类图 [B, K, H, W]                                   │
│  ├── 不确定性热力图 (Vacuity + Dissonance)                  │
│  └── 校准置信度                                             │
└─────────────────────────────────────────────────────────────┘
```

### FusionCropNetV5EDL

```
┌─────────────────────────────────────────────────────────────┐
│                   FusionCropNetV5EDL                       │
├─────────────────────────────────────────────────────────────┤
│  输入层                                                     │
│  ├── 光学时序 [B, T, 10, H, W]                            │
│  ├── SAR时序 [B, T, 5, H, W]                              │
│  ├── DEM [B, 5, H, W]                                     │
│  └── DOY [B, T]                                           │
├─────────────────────────────────────────────────────────────┤
│  编码器层                                                   │
│  ├── OpticalEncoder (ResNet50 + FPN)                      │
│  ├── SAREncoder (IRB + FiLM)                              │
│  └── DEMEncoder                                           │
├─────────────────────────────────────────────────────────────┤
│  时序Transformer                                            │
│  ├── TemporalEncoderStream (光学)                          │
│  └── TemporalEncoderStream (SAR)                          │
├─────────────────────────────────────────────────────────────┤
│  跨模态融合 + 解码器                                        │
│  └── EDLHead (Dirichlet分布)                              │
├─────────────────────────────────────────────────────────────┤
│  输出                                                       │
│  ├── 预测概率 [B, K, H, W]                                 │
│  ├── Vacuity (数据不确定性)                                 │
│  └── Dissonance (认知不确定性)                              │
└─────────────────────────────────────────────────────────────┘
```

## 📈 不确定性估计

系统支持两种不确定性度量：

| 度量 | 说明 | 用途 |
|------|------|------|
| **Vacuity** | 数据不确定性/证据不足程度 | 识别数据质量差的区域 |
| **Dissonance** | 认知不确定性/类间冲突程度 | 识别模型不确定的区域 |

## 🚀 快速开始

### 训练模型

```bash
# 训练标准模型
python scripts/train_fusion.py

# 训练EDL模型（推荐）
python scripts/train_fusion.py --edl
```

### 推理预测

```bash
# 标准推理
python scripts/predict.py

# 带不确定性估计的推理
python scripts/predict.py --edl --n_passes 5
```

### 导出为ArcGIS格式

```bash
python scripts/export_to_arcgis.py --edl
```

## 📦 打包为EXE应用程序

将整个项目打包为独立的Windows可执行文件（.exe），无需安装Python即可运行。

### 打包方式

```bash
# 交互式打包（推荐）
python build_exe.py

# 文件夹模式（启动快 ~5-15秒）
python build_exe.py --mode onedir

# 单文件模式（便携但启动慢 ~30秒-2分钟）
python build_exe.py --mode onefile

# 仅检查环境
python build_exe.py --check-only
```

### 输出文件

打包完成后在 `dist/` 目录下：

| 输出 | 说明 | 体积 | 启动速度 |
|------|------|------|---------|
| `遥感影像作物分类系统_portable/` | 文件夹版 (推荐) | ~4-8 GB | 5-15秒 |
| `遥感影像作物分类系统.exe` | 单文件版 | ~2-4 GB | 30秒-2分钟 |

### 使用方式

```
# 文件夹版
双击: dist/遥感影像作物分类系统_portable/遥感影像作物分类系统.exe

# 单文件版
双击: dist/遥感影像作物分类系统.exe
```

启动后会显示功能选择菜单：
1. 桌面GUI推理 (PyQt5) — 图形界面
2. Web界面 (Streamlit) — 浏览器运行
3. API服务 (FastAPI) — REST API
4. 命令行推理 — 批量处理
5. EDL校准分析 — 生成校准报告
6. 模型诊断 — 健康检查

### 减小体积

如果不需要GPU推理，安装CPU版PyTorch可减少3-4GB体积：

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python build_exe.py --mode onedir
```

### 前置条件

打包前确保已安装：
```bash
pip install pyinstaller PyQt5
```

## 📝 License

MIT License