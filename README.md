# FusionCropNet — 多模态遥感影像农作物精细分类系统

> **From pixels to parcels, with quantified uncertainty.**

一个完整的深度学习遥感作物分类解决方案：三模态数据融合（光学 + SAR + DEM）、层次化多尺度架构、证据深度学习不确定性估计、从训练到部署的全链路工具。

[![GitHub stars](https://img.shields.io/github/stars/jasonjaves64-dotcom/project1.remote-sensing-classification?style=flat)](https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/🤗-HF%20Spaces-orange)](https://jjjj111qq111-fusioncropnet-v6.hf.space)

---

## 主题

**多模态遥感融合 + 不确定性感知** —— 这是贯穿整个系统的两条主线。

遥感作物分类长期面临三个问题：(1) 单一模态信息不足，(2) 模型预测缺乏置信度度量，(3) 生产部署链路断裂。FusionCropNet 从 V1 到 V6，始终围绕"融合更多模态、量化不确定性、打通部署"三步走。

---

## 模型族

```
V1 ─── V4 ─── V5 ─── V5EDL ─── V5Pro ─── ★ V6 (当前)
双模态    +DEM    标准化    EDL     旗舰      下一代
```

| 版本 | 模态 | 核心创新 | 参数量 |
|------|------|----------|--------|
| V1 | 光学 + SAR | 双模态融合 + 单路时序 | — |
| V4 | +DEM | 三模态 + 双路时序 + MC-Dropout | — |
| V5 | 同上 | 组件重构 + 4 bug修复 | 47.8M |
| V5EDL | 同上 | Dirichlet 证据学习 + 不确定性分解 | 47.8M |
| V5Pro | 同上 | MIL + HPO + CARAFE + 多尺度 | 49.0M |
| **V6** ★ | 同上 | **层次化多尺度多任务，14 新组件** | 49.0M |

---

## 最新更新: V6.1.1 (2026-05-25)

- **Bug 修复**: V5Pro Decoder 通道适配 + TSViT 输出层修复, 5 个模型 API 全部 200 OK
- **安全加固**: docker-compose 密码环境变量化, .env 豁免提交, SQL 注入修复
- **前端修复**: Vue 大屏依赖补全 (pinia/vue-router/echarts), API 代理端口对接
- **部署优化**: Dockerfile 多阶段构建, 依赖清单统一 (pyproject.toml → requirements.txt)

---

## V6 更新 (2026-05-23)

### 核心创新

**14 个新组件**，围绕"层次化多尺度 + 多任务学习"重构：

| Block | 组件 | 功能 |
|-------|------|------|
| **Block 1** | TemporalLite | 轻量时序编码 (FFN替代Self-Attention,~48× 加速) |
| **Block 2** | MultiScaleFusion | 层次化多尺度特征融合 (s1/s2 联合) |
| **Block 3** | CARAFE Upsampler | 内容感知上采样 |
| **Block 4** | DEMDeepFuser | DEM 深度特征注入 |
| **Block 5** | BoundaryAware | 边缘感知损失 + 边界细化 |
| **Block 6** | MultiTaskHead | 5 任务输出 (分类 + 边缘 + 语义 + 变分 + 重建) |
| **Block 7** | PretrainedEncoder | SeCo 预训练权重 + 域自适应 |
| **Block 8** | ActiveSampler | 主动学习采样策略 |
| **Block 9** | SceneParser | 场景级上下文理解 |

### 训练基础设施

- **AMP 混合精度训练** — 显存降低 40%，速度提升 2×
- **梯度裁剪 + LR Warmup** — 稳定千轮训练
- **断点续训** — 任意中断可无损恢复
- **TensorBoard 集成** — 5 任务 Loss 实时曲线
- **自动化全量测试** — 230 tests, 0 failures

### 数据工程

- **统一预处理管道** — 5 套管道合并为 1 套可配置管道 (代码量 -86.1%)
- **LRU 数据缓存** — DataLoader 吞吐量 +85% (4.2× 缓存加速)
- **异步预加载** — GPU 零等待
- **自动数据验证** — shape 检查 + NaN 检测

### 生产部署

- **Streamlit** Web 界面 — 暗色/亮色主题、批量推理、不确定性热力图
- **Gradio** HF Spaces 在线 Demo — 零安装浏览器即用
- **FastAPI** REST 服务 — 标准化推理端点
- **Docker** 支持 — 一键部署
- **GitHub Actions CI/CD** — push 自动测试 + 自动部署

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      用户层                                │
│  Streamlit UI │ Gradio HF Spaces │ FastAPI │ Desktop EXE    │
├──────────────────────────────────────────────────────────┤
│                      推理引擎层                             │
│  predict.py │ EDL-Ensemble │ Calibration │ Uncertainty Viz │
├──────────────────────────────────────────────────────────┤
│                      训练引擎层                             │
│  trainer.py │ AMP 混合精度 │ 断点续训 │ TensorBoard        │
├──────────────────────────────────────────────────────────┤
│                      V6 模型核心                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Optical(S2) ──┐                                   │   │
│  │ SAR(S1)    ───┼──→ FFN 时序 ──→ 多尺度融合 ──→ 5-Task Head  │
│  │ DEM         ──┘     (48×加速)     (s1/s2联合)    │   │
│  │ DOY + Scene ────────────────────────────────────→  │   │
│  └──────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────┤
│                      数据层                                │
│  统一预处理管道 │ LRU 缓存(4.2×) │ 异步预加载 │ 数据验证      │
├──────────────────────────────────────────────────────────┤
│                  输入: 光学 + SAR + DEM + DOY                │
│                  输出: 分类图 + 不确定性热力图                │
└──────────────────────────────────────────────────────────┘
```

---

## 不确定性估计 (EDL)

系统基于 **Dirichlet 证据深度学习**提供两类不确定性：

| 度量 | 含义 | 应用场景 |
|------|------|----------|
| **Vacuity** | 数据不确定性（证据不足） | 识别云覆盖/数据质量差的区域 |
| **Dissonance** | 认知不确定性（类间冲突） | 标记边界模糊区域 → 主动学习 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification.git
cd project1.remote-sensing-classification
python install.py
```

### 推理

```bash
# Web 界面
python start.py

# 命令行
python scripts/predict.py --model V6 --edl --input your_data.npy

# API 服务
uvicorn api.main:app --port 8000
```

### 训练

```bash
# V6 完整训练
python scripts/train_fusion_edl.py --v6 --amp --resume

# V5EDL 标准训练
python scripts/train_fusion_edl.py
```

### 在线体验

浏览器打开 https://jjjj111qq111-fusioncropnet-v6.hf.space 直接使用，无需安装。

---

## 项目结构

```
project1/
├── models/                  # 模型族 (V1→V6)
│   ├── fusion_net_v6.py     # ★ V6 旗舰
│   ├── fusion_net_v5pro.py  # V5Pro (MIL+HPO)
│   ├── fusion_net_v5_edl.py # V5EDL (证据学习)
│   ├── tsvit.py             # Baseline
│   └── ...
├── data/
│   ├── preprocessing/       # 统一预处理管道 (14 模块)
│   ├── cache/               # LRU 缓存系统
│   └── datasets/            # Dataset + DataLoader
├── utils/
│   ├── trainer.py           # 训练器 (AMP + resume)
│   ├── losses.py            # 损失函数 (含 EDL Loss)
│   ├── calibration.py       # 温度缩放 + ECE
│   ├── hpo.py               # Optuna 超参搜索
│   └── ...
├── scripts/
│   ├── train_fusion_edl.py  # 主训练脚本
│   ├── predict.py           # 推理脚本
│   ├── demo_v6.py           # V6 Demo
│   └── ...
├── api/                     # FastAPI 服务
├── frontend/                # Vue 地图大屏
├── sql/                     # SQLite 训练日志
├── tests/                   # 255 单元测试
├── docs/                    # 设计文档 + 审阅报告
├── app.py                   # Streamlit 入口
├── demo_app.py              # Gradio HF Spaces
├── Dockerfile               # Docker 部署
└── docker-compose.yml       # 一键自托管
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.0+, TorchVision, Timm |
| 遥感处理 | rasterio, GDAL, sentinelhub |
| 前端 | Streamlit, Gradio, Vue 3 |
| API | FastAPI + Uvicorn |
| 数据库 | SQLite + SQLAlchemy |
| DevOps | Docker, GitHub Actions, HF Spaces |
| 可视化 | matplotlib, ECharts, folium |
| 优化 | AMP, Optuna, LRU Cache, async I/O |

---

## 测试

```bash
pytest tests/ -v --cov
```

当前状态：**230 tests, 0 failures** (2026-05-25)。

---

## 引用

```bibtex
@software{fusioncropnet2026,
  title     = {FusionCropNet: Multi-Modal Remote Sensing Crop Classification},
  author    = {Jason Zhou},
  year      = {2026},
  url       = {https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification}
}
```

---

*Licensed under MIT. Built with PyTorch and determination.*
