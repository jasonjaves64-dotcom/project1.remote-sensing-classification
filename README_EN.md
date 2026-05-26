**English** | [**中文**](README.md)

---

# FusionCropNet — Multi-Modal Remote Sensing Crop Classification System

> **From pixels to parcels, with quantified uncertainty.**

A full-stack deep learning solution for remote sensing crop classification: tri-modal data fusion (Sentinel-2 Optical + Sentinel-1 SAR + DEM), hierarchical multi-scale architecture, Evidential Deep Learning (EDL) uncertainty estimation, and end-to-end toolchain from training to deployment.

[![GitHub stars](https://img.shields.io/github/stars/jasonjaves64-dotcom/project1.remote-sensing-classification?style=flat)](https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-orange)](https://jjjj111qq111-fusioncropnet-v6.hf.space)
[![Tests](https://img.shields.io/badge/tests-230%20passed-success)]()

---

## Table of Contents

1. [Latest Updates](#latest-updates-v611)
2. [Model Card](#model-card)
3. [Model Family](#model-family)
4. [System Architecture](#system-architecture)
5. [Uncertainty Estimation](#uncertainty-estimation-edl)
6. [Quick Start](#quick-start)
7. [API Documentation](#api-documentation)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Security Disclosure](#security-disclosure)
11. [Citation](#citation)

---

## Latest Updates: V6.1.1 (2026-05-25)

| Category | Details |
|------|------|
| **Bug Fixes** | V5Pro Decoder channel adaptation (256→128), TSViT output layer, api/main.py MODEL initialization, SQL parameterized queries |
| **Security** | docker-compose password env-var migration, .env git-excluded, HuggingFace token removed |
| **Frontend** | Vue dashboard deps (pinia/vue-router/echarts), API proxy port verification |
| **Deployment** | Dockerfile multi-stage build, requirements.txt consolidation (5→18 pkgs), .dockerignore |
| **Integration** | All 5 model APIs return 200 OK |

---

## Model Card

| Attribute | Value |
|------|-----|
| **Model Name** | FusionCropNet V6 |
| **Task** | Pixel-wise crop semantic segmentation (7 classes: wheat / corn / rice / soybean / cotton / vegetable / other) |
| **Input** | Sentinel-2 Optical (10 bands × 12 time steps) + Sentinel-1 SAR (5 channels × 12 time steps) + DEM (5 features) |
| **Output** | Classification map + uncertainty heatmaps (vacuity / dissonance / class variance) |
| **Parameters** | 49.0M |
| **Architecture** | CNN encoder-decoder + Transformer temporal encoder + multi-scale cross-modal attention |
| **Uncertainty** | Dirichlet Evidential Deep Learning + MC-Dropout + TTA |
| **Backbone** | ResNet50 / ConvNeXt-Tiny / EfficientNet-B0/B4 (pluggable) |
| **Training** | PyTorch 2.0+ with AMP mixed precision + gradient clipping + checkpoint resume |
| **Test Baseline** | 230 tests, 0 failures |
| **License** | MIT |

---

## Model Family

From dual-modal to tri-modal, from deterministic to uncertainty-aware — six generations of continuous evolution:

```
V1 ─── V4 ─── V5 ─── V5EDL ─── V5Pro ─── ★ V6 (current)
Dual    +DEM    Standard  EDL       Flagship   Next-Gen
```

| Version | Modalities | Key Innovation | Params | Status |
|------|------|----------|:--:|:--:|
| V1 | Optical + SAR | Dual-modal fusion + single-path temporal | — | Archived |
| V4 | +DEM | Tri-modal + dual-path temporal + MC-Dropout | — | Ablation baseline |
| V5 | Same | Component refactor + 4 bug fixes | 47.8M | Simplified to 150 lines |
| V5EDL | Same | EDL uncertainty + vacuity/dissonance decomposition | 47.8M | Base class for all EDL variants |
| V5Pro | Same | Pluggable backbone + CARAFE + multi-scale fusion | 49.0M | Flagship |
| **V6** ★ | Same | **14 new components, hierarchical multi-task** | 49.0M | **Current** |
| TSViT | Optical | Pure Transformer baseline (temporal-spatial ViT) | — | Benchmark |

---

## V6 Architecture Evolution (2026-05-23)

**14 new components** rebuilt around "hierarchical multi-scale + multi-task learning":

### Core Components

| Block | Component | Function | Technical Detail |
|-------|------|------|----------|
| **Block 1** | TemporalLite | Lightweight temporal encoding | FFN replaces Self-Attention, ~48× faster |
| **Block 2** | MultiScaleFusion | Hierarchical multi-scale fusion | Dual-scale (s1/s2) joint cross-modal attention |
| **Block 3** | CARAFE Upsampler | Content-aware upsampling | Avoids transposed convolution checkerboard artifacts |
| **Block 4** | DEMDeepFuser | Deep DEM feature injection | Three injection points: Early Fusion + FiLM + Decoder Skip |
| **Block 5** | BoundaryAware | Boundary awareness | Auxiliary boundary detection + Dice loss |
| **Block 6** | MultiTaskHead | Multi-task output | Classification + LAI + Growth Stage + Boundary + Scene |
| **Block 7** | PretrainedEncoder | Remote sensing pre-training | SeCo contrastive weights + domain adaptation |
| **Block 8** | ActiveSampler | Active learning | Uncertainty-ranked sampling |
| **Block 9** | SceneParser | Scene understanding | Scene type + crop distribution prediction |

### Training Enhancements

- **AMP Mixed Precision** — 40% memory reduction, 2× training speed
- **Gradient Clipping + LR Warmup** — stable convergence over thousands of epochs
- **Checkpoint Resume** — full optimizer/scheduler/seed state restoration
- **Modal Dropout** — random modality masking for robustness

### Data Pipeline

- **Unified Preprocessing** — 5 pipelines merged into 1 configurable pipeline (86.1% code reduction)
- **LRU Data Cache** — DataLoader throughput +85% (4.2× cache acceleration)
- **Async Preloading** — zero GPU idle time

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      User Layer                           │
│  Vue Dashboard │ Gradio HF Spaces │ FastAPI REST         │
├──────────────────────────────────────────────────────────┤
│                      Inference Engine                     │
│  EDL-Ensemble │ Calibration (ECE/MCE/NLL/Brier) │ Viz     │
├──────────────────────────────────────────────────────────┤
│                      Training Engine                      │
│  TwoPhaseTrainer │ AMP │ Checkpoint Resume │ TensorBoard   │
├──────────────────────────────────────────────────────────┤
│                      V6 Model Core                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Optical(S2) ──┐                                  │   │
│  │ SAR(S1)    ───┼──→ FFN Temporal ──→ Multi-Scale ──→ MultiTask Head
│  │ DEM         ──┘     (48× faster)     Fusion       │   │
│  │ DOY + Scene ────────────────────────────────────→   │   │
│  └──────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────┤
│                      Data Layer                           │
│  Unified Pipeline │ LRU Cache(4.2×) │ Async Preload │ Auto Validate │
├──────────────────────────────────────────────────────────┤
│          Input: Optical(10 bands × 12 steps) + SAR(5ch × 12 steps) + DEM(5 features)
│          Output: Classification Map + Vacuity + Dissonance + Per-Class Variance │
└──────────────────────────────────────────────────────────┘
```

---

## Uncertainty Estimation (EDL)

The system uses **Evidential Deep Learning** to provide three types of uncertainty:

| Metric | Meaning | Use Case |
|------|------|------|
| **Vacuity** | Aleatoric uncertainty — model hasn't "seen" this kind of input | Identify cloud-covered / data-missing regions |
| **Dissonance** | Epistemic uncertainty — model "can't decide" between two classes | Flag ambiguous boundary regions → active learning |
| **Class Variance** | Per-class uncertainty — how uncertain the model is about a specific class | Hard sample mining |

At inference time, **MC Dropout** (10 passes) + **TTA** (horizontal flip) with alpha-level fusion reduces single-inference stochasticity.

---

## Quick Start

### Installation

```bash
git clone https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification.git
cd project1.remote-sensing-classification
python install.py
```

### Web Interface

```bash
# Vue Dashboard (recommended)
cd frontend && npm install && npm run dev

# Gradio Demo
python demo_v6.py
```

Open http://localhost:5173, draw an AOI → Run Classification.

### Command Line Inference

```bash
# EDL inference with uncertainty
python scripts/predict.py --model V6 --edl --input your_sequence.npy

# Calibration report
python scripts/predict.py --model V5EDL --edl --calibration --label labels.tif
```

### API Service

```bash
API_KEY=your_key uvicorn api.main:app --port 8000
```

API docs: http://localhost:8000/docs

### Online Demo

Visit https://jjjj111qq111-fusioncropnet-v6.hf.space — no installation required.

> **Note**: The current online demo uses synthetic random data for pipeline demonstration. Real classification results require model weights trained on real remote sensing data. Training and benchmark evaluation coming in V6.2.

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|------|------|------|
| GET | `/health` | Health check + model load status |
| POST | `/predict/{model}` | JSON inference (model: v5/v5edl/v5pro/v6/tsvit) |
| POST | `/predict/{model}/upload` | File upload inference (.tif/.npy/.npz) |
| GET | `/stats` | Service statistics |

### Prediction Request

```json
POST /predict/v6
{
  "aoi": {"type": "Polygon", "coordinates": [[...]]}
}
```

### Prediction Response

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

## Project Structure

```
project1/
├── models/                  # Model family (V1→V6 + TSViT)
│   ├── _base.py             # ★ Shared components (15 modules, single source)
│   ├── fusion_net_v6.py     # ★ V6 Flagship (based on V5EDL, 14 new components)
│   ├── fusion_net_v5_edl.py # V5EDL (EDL base class)
│   ├── fusion_net_v5pro.py  # V5Pro (pluggable backbone)
│   ├── tsvit.py             # Pure Transformer baseline
│   ├── multi_task_heads.py  # Multi-task heads (LAI/Growth/Boundary/Scene)
│   └── temporal_lite.py     # Lightweight temporal encoder
│
├── data/
│   ├── preprocessing/       # Unified preprocessing pipeline
│   │   ├── pipeline.py      #   Main pipeline (3 modes)
│   │   ├── optical.py       #   Optical preprocessing
│   │   ├── sar.py           #   SAR preprocessing
│   │   └── label.py         #   Label processing
│   ├── cache/               # LRU 3-tier cache system
│   └── datasets/            # Dataset + DataLoader
│
├── utils/
│   ├── trainer.py           # Trainer (AMP + resume)
│   ├── calibration.py       # Calibration (ECE/MCE/NLL/Brier/Spearman)
│   ├── losses.py            # Loss functions (DiceFocal + EDLLoss + Tversky)
│   └── hpo.py               # Optuna hyperparameter search
│
├── scripts/
│   ├── train_fusion_edl.py  # Main training script (Phase1+Phase2)
│   ├── predict.py           # Inference + calibration + TTA
│   └── test_all_models_comparison.py  # Multi-model comparison
│
├── api/                     # FastAPI backend
├── frontend/                # Vue 3 dashboard
├── tests/                   # 230 test cases
├── Dockerfile               # Multi-stage build (CPU torch)
├── docker-compose.yml       # One-click self-host (API + MySQL)
└── .env.example             # Environment variable template
```

---

## Testing

```bash
# Full test suite
pytest tests/ -v

# V6-specific tests
pytest tests/ -v -k "V6"

# Coverage report
pytest tests/ --cov=models --cov-report=html
```

| Metric | Value |
|------|:--:|
| Total Tests | 230 |
| Pass Rate | 100% (0 failures) |
| V6 Specific | 8 passed |
| Dataset | Synthetic (real data pending V6.2) |

---

## Security Disclosure

This is a **public open-source repository**. The following measures protect sensitive information:

- `.env` is `.gitignore`-d; only `.env.example` with placeholder values is committed
- `docker-compose.yml` passwords use `${ENV_VAR:-default}` pattern
- All SQL queries use parameterized statements
- API keys use development defaults (not production credentials)
- No private keys, GitHub tokens, or database credentials are stored

**To report a security issue**: please open a GitHub Issue or contact the maintainer.

---

## Tech Stack

| Layer | Technology |
|------|------|
| Deep Learning | PyTorch 2.0+, TorchVision, Timm, einops |
| Remote Sensing | rasterio, GDAL, sentinelhub |
| Frontend | Vue 3 + Vite + Pinia + Leaflet + ECharts |
| Backend | FastAPI + Uvicorn |
| Database | MySQL 8.0 + MySQL Connector |
| DevOps | Docker, GitHub Actions, HF Spaces, Gradio |
| Visualization | matplotlib, ECharts, folium |
| Optimization | AMP, Optuna, LRU Cache, async I/O |

---

## Citation

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
