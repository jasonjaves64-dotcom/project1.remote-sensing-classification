[**English**](README_EN.md) | **中文**

---

# FusionCropNet — 多模态遥感影像农作物精细分类系统

> **From pixels to parcels, with quantified uncertainty.**
> 从像素到地块，每一步都有置信度。

一个完整的深度学习遥感作物分类解决方案：三模态数据融合（Sentinel-2 光学 + Sentinel-1 SAR + DEM 地形）、层次化多尺度架构、证据深度学习 (EDL) 不确定性估计、从训练到部署的全链路工具。

[![GitHub stars](https://img.shields.io/github/stars/jasonjaves64-dotcom/project1.remote-sensing-classification?style=flat)](https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-orange)](https://jjjj111qq111-fusioncropnet-v6.hf.space)
[![Tests](https://img.shields.io/badge/tests-230%20passed-success)]()

---

## 目录

1. [最新更新](#最新更新-v611)
2. [模型卡片](#模型卡片)
3. [模型族谱](#模型族谱)
4. [系统架构](#系统架构)
5. [不确定性估计](#不确定性估计-edl)
6. [快速开始](#快速开始)
7. [API 文档](#api-文档)
8. [项目结构](#项目结构)
9. [测试](#测试)
10. [安全声明](#安全声明)
11. [引用](#引用)

---

## 最新更新: V6.1.1 (2026-05-25)

| 类别 | 内容 |
|------|------|
| **Bug 修复** | V5Pro Decoder 通道适配 (256→128), TSViT 输出层修复, api/main.py MODEL 初始化, SQL 参数化查询 |
| **安全加固** | docker-compose 密码环境变量化, .env 豁免提交, HuggingFace token 清除 |
| **前端修复** | Vue 大屏依赖补全 (pinia/vue-router/echarts), API 端口对接 |
| **部署优化** | Dockerfile 多阶段构建, requirements.txt 统一 (5→18 包), .dockerignore |
| **模型联调** | 5 个模型 API 全部 200 OK |

---

## 模型卡片

| 属性 | 值 |
|------|-----|
| **模型名称** | FusionCropNet V6 |
| **任务** | 像素级作物语义分割 (7 类: 小麦/玉米/水稻/大豆/棉花/蔬菜/其他) |
| **输入** | Sentinel-2 光学 (10波段×12时间步) + Sentinel-1 SAR (5通道×12时间步) + DEM (5特征) |
| **输出** | 分类图 + 不确定性热力图 (vacuity/dissonance) |
| **参数量** | 49.0M |
| **架构** | CNN编码器-解码器 + Transformer时序编码 + 多尺度跨模态注意力 |
| **不确定性** | Dirichlet 证据深度学习 (EDL) + MC-Dropout + TTA |
| **骨干网络** | ResNet50 / ConvNeXt-T / EfficientNet-B0/B4 (可插拔) |
| **训练框架** | PyTorch 2.0+ + AMP混合精度 + 梯度裁剪 + 断点续训 |
| **测试基准** | 230 tests, 0 failures |
| **许可证** | MIT |

---

## 模型家族

从双模态到三模态，从确定性到不确定性量化，6 代持续演进：

```
V1 ─── V4 ─── V5 ─── V5EDL ─── V5Pro ─── ★ V6 (当前)
双模态    +DEM    标准化    EDL     旗舰      下一代
```

| 版本 | 模态 | 核心创新 | 参数量 | 状态 |
|------|------|----------|:--:|:--:|
| V1 | 光学 + SAR | 双模态融合 + 单路时序 | — | 已归档 |
| V4 | +DEM | 三模态 + 双路时序 + MC-Dropout | — | 消融对照 |
| V5 | 同上 | 组件重构 + 4 bug修复 | 47.8M | 已精简至 150 行 |
| V5EDL | 同上 | Dirichlet 证据学习 + 不确定性分解 | 47.8M | 所有 EDL 版本基类 |
| V5Pro | 同上 | 可插拔骨干 + CARAFE + 多尺度融合 | 49.0M | 旗舰 |
| **V6** ★ | 同上 | **14 新组件, 层次化多尺度多任务** | 49.0M | **当前主力** |
| TSViT | 光学 | 纯 Transformer 基线 (时空 ViT) | — | 论文对照 |

---

## V6 架构演进 (2026-05-23)

**14 个新组件**，围绕"层次化多尺度 + 多任务学习"重构：

### 核心组件

| Block | 组件 | 功能 | 技术细节 |
|-------|------|------|----------|
| **Block 1** | TemporalLite | 轻量时序编码 | FFN替代Self-Attention, ~48× 加速 |
| **Block 2** | MultiScaleFusion | 层次化多尺度融合 | s1/s2 两个尺度联合 attention |
| **Block 3** | CARAFE Upsampler | 内容感知上采样 | 避免转置卷积的棋盘伪影 |
| **Block 4** | DEMDeepFuser | DEM 深度特征注入 | 三层注入: Early Fusion + FiLM + Decoder Skip |
| **Block 5** | BoundaryAware | 边界感知 | 边界检测辅助任务 + Dice loss |
| **Block 6** | MultiTaskHead | 多任务输出 | 分类 + LAI + 生长期 + 边界 + 场景 |
| **Block 7** | PretrainedEncoder | 遥感预训练 | SeCo 对比学习权重 + 域自适应 |
| **Block 8** | ActiveSampler | 主动学习 | 不确定度排序采样 |
| **Block 9** | SceneParser | 场景理解 | 场景类型 + 作物分布预测 |

### 训练增强

- **AMP 混合精度** — 显存降低 40%, 训练速度 2×
- **梯度裁剪 + LR Warmup** — 稳定千轮训练
- **断点续训** — 任意中断可无损恢复, optimizer/scheduler 状态全量保存
- **Modal Dropout** — 随机丢弃模态增强鲁棒性

### 数据管线

- **统一预处理管道** — 5 套管道合并为 1 套 (代码量 -86.1%)
- **LRU 数据缓存** — DataLoader 吞吐量 +85% (4.2× 缓存加速)
- **异步预加载** — GPU 零空闲

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      用户层                               │
│  Vue 地图大屏 │ Gradio HF Spaces │ FastAPI REST          │
├──────────────────────────────────────────────────────────┤
│                      推理引擎层                            │
│  EDL-Ensemble推理 │ 校准验证 (ECE/MCE/NLL/Brier) │ 不确定性可视化 │
├──────────────────────────────────────────────────────────┤
│                      训练引擎层                            │
│  TwoPhaseTrainer │ AMP 混合精度 │ 断点续训 │ TensorBoard    │
├──────────────────────────────────────────────────────────┤
│                      V6 模型核心                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Optical(S2) ──┐                                  │   │
│  │ SAR(S1)    ───┼──→ FFN 时序 ──→ 多尺度融合 ──→ MultiTask Head
│  │ DEM         ──┘     (48×加速)    (s1/s2联合)     │   │
│  │ DOY + Scene ────────────────────────────────────→   │   │
│  └──────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────┤
│                      数据层                               │
│  统一预处理管道 │ LRU 缓存(4.2×) │ 异步预加载 │ 自动数据验证 │
├──────────────────────────────────────────────────────────┤
│              输入: 光学(10波段×12步) + SAR(5ch×12步) + DEM(5特征)  │
│              输出: 分类图 + vacuity + dissonance + per-class var   │
└──────────────────────────────────────────────────────────┘
```

---

## 不确定性估计 (EDL)

系统基于 **Dirichlet 证据深度学习** (Evidential Deep Learning) 提供三类不确定性度量：

| 度量 | 英文 | 含义 | 应用 |
|------|------|------|------|
| **Vacuity** | 证据真空度 | 数据不确定性 — 模型"没见过"这种输入 | 识别云覆盖/数据缺失区域 |
| **Dissonance** | 认知冲突度 | 知识不确定性 — 模型"分不清"两个类别 | 标记类间边界模糊区域 → 主动学习 |
| **Class Variance** | 类间方差 | 单类不确定性 — 某个特定类别的不确定度 | 难样本挖掘 |

**推理时**通过 MC Dropout (10 passes) + TTA (水平翻转) 的多轮 alpha 融合降低单一推理的随机性。

---

## 快速开始

### 安装

```bash
git clone https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification.git
cd project1.remote-sensing-classification
python install.py
```

### Web 推理

```bash
# Vue 地图大屏 (推荐)
cd frontend && npm install && npm run dev

# Gradio 在线 Demo
python demo_v6.py
```

浏览器打开 http://localhost:5173, 框选 AOI → Run Classification。

### 命令行推理

```bash
# EDL 不确定性推理
python scripts/predict.py --model V6 --edl --input your_sequence.npy

# 校准验证
python scripts/predict.py --model V5EDL --edl --calibration --label labels.tif
```

### API 服务

```bash
API_KEY=your_key uvicorn api.main:app --port 8000
```

API 文档: http://localhost:8000/docs

### 在线体验

浏览器打开 https://jjjj111qq111-fusioncropnet-v6.hf.space — 零安装。

> **注意**：当前在线 Demo 使用合成随机数据演示管线。真实分类结果需要在真实遥感数据上训练的模型权重。训练与基准评估将在 V6.2 中提供。

---

## API 文档

### 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 + 模型加载状态 |
| POST | `/predict/{model}` | JSON推理 (模型: v5/v5edl/v5pro/v6/tsvit) |
| POST | `/predict/{model}/upload` | 文件上传推理 (.tif/.npy/.npz) |
| GET | `/stats` | 服务运行统计 |

### 预测请求示例

```json
POST /predict/v6
{
  "aoi": {"type": "Polygon", "coordinates": [[...]]}
}
```

### 预测响应

```json
{
  "dominant": "wheat",
  "confidence": 87.3,
  "time": 1.12,
  "distribution": {
    "wheat": 87.3, "soybean": 8.1, "corn": 3.2, "other": 1.4
  },
  "aux": {}
}
```

---

## 项目结构

```
project1/
├── models/                  # 模型族 (V1→V6 + TSViT基线)
│   ├── _base.py             # ★ 共享组件 (15个模块, 单一来源)
│   ├── fusion_net_v6.py     # ★ V6 旗舰 (基于V5EDL, 14新组件)
│   ├── fusion_net_v5_edl.py # V5EDL (EDL基类)
│   ├── fusion_net_v5pro.py  # V5Pro (可插拔骨干)
│   ├── tsvit.py             # 纯Transformer基线
│   ├── multi_task_heads.py  # 多任务头 (LAI/Growth/Boundary/Scene)
│   └── temporal_lite.py     # 轻量时序编码器
│
├── data/
│   ├── preprocessing/       # 统一预处理管道
│   │   ├── pipeline.py      #   主管道 (3模式: 修正/增强/融合)
│   │   ├── optical.py       #   光学预处理 (大气校正 + BRDF)
│   │   ├── sar.py           #   SAR预处理 (散斑滤波 + dB转换)
│   │   └── label.py         #   标签处理 (类别分布 + 掩膜)
│   ├── cache/               # LRU 三级缓存系统
│   └── datasets/            # Dataset + DataLoader
│
├── utils/
│   ├── trainer.py           # 训练器 (AMP + 断点续训)
│   ├── calibration.py       # 校准评估 (ECE/MCE/NLL/Brier/Spearman)
│   ├── losses.py            # 损失函数 (DiceFocal + EDLLoss + Tversky)
│   └── hpo.py               # Optuna 超参搜索
│
├── scripts/
│   ├── train_fusion_edl.py  # 主训练脚本 (Phase1+Phase2)
│   ├── predict.py           # 推理 + 校准 + TTA
│   ├── test_all_models_comparison.py  # 模型对比脚本
│   └── benchmark_dataloader.py        # DataLoader压测
│
├── api/                     # FastAPI 后端
├── frontend/                # Vue 3 地图大屏
├── tests/                   # 230 测试用例
├── Dockerfile               # 多阶段构建 (CPU torch)
├── docker-compose.yml       # 一键自托管 (API + MySQL)
└── .env.example             # 环境变量模板
```

---

## 测试

```bash
# 全量测试
pytest tests/ -v

# 仅 V6 测试
pytest tests/ -v -k "V6"

# 覆盖率报告
pytest tests/ --cov=models --cov-report=html
```

| 指标 | 数值 |
|------|:--:|
| 总测试数 | 230 |
| 通过率 | 100% (0 failures) |
| V6 专项 | 8 passed |
| 数据集 | 合成数据 (true data pending V6.2) |

---

## 安全声明

本项目是 **公开的开源仓库**。我们采取了以下措施保护敏感信息：

- `.env` 已加入 `.gitignore`，仅 `.env.example` 占位符提交
- `docker-compose.yml` 密码使用 `${ENV_VAR:-default}` 模式
- SQL 查询全部参数化
- API 密钥使用开发环境默认值（非生产凭证）
- 无密钥、Token、数据库连接串提交

**报告安全问题**: 请提交 GitHub Issue 或联系维护者。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.0+, TorchVision, Timm, einops |
| 遥感 | rasterio, GDAL, sentinelhub |
| 前端 | Vue 3 + Vite + Pinia + Leaflet + ECharts |
| 后端 | FastAPI + Uvicorn |
| 数据库 | MySQL 8.0 + MySQL Connector |
| DevOps | Docker, GitHub Actions, HF Spaces, Gradio |
| 可视化 | matplotlib, ECharts, folium |
| 优化 | AMP, Optuna, LRU Cache, async I/O |

---

## 引用

```bibtex
@software{fusioncropnet2026,
  title     = {FusionCropNet: Multi-Modal Remote Sensing Crop Classification
               with Evidential Deep Learning},
  author    = {Jason Zhou},
  year      = {2026},
  url       = {https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification}
}
```

---

*Licensed under MIT. Built with PyTorch and determination.*
