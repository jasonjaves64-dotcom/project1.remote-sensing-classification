**English** | [**中文**](README.md)

---

# FusionCropNet — Multi-Modal Remote Sensing Crop Classification

> 🎓 A student's learning project in remote sensing and deep learning. Work in progress — feedback welcome.

This is my attempt at applying deep learning to crop classification using multi-modal remote sensing data. It explores fusing Sentinel-2 optical time series, Sentinel-1 SAR, and DEM data for pixel-level crop mapping, along with Evidential Deep Learning (EDL) for uncertainty estimation. I built this to learn — expect rough edges and incomplete experiments.

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-orange)](https://jjjj111qq111-fusioncropnet-v6.hf.space)

---

## ⚠️ Disclaimer

**This is a student side project.** It was built primarily as a learning exercise. The code may have imperfections, and experimental conclusions should be taken as tentative. Suggestions and corrections are very welcome — I'm here to learn.

The current pipeline uses synthetic data for validation. Training and evaluation on real remote sensing data is planned but not yet complete.

---

## Contents

1. [Model Overview](#model-overview)
2. [Version History](#version-history)
3. [System Overview](#system-overview)
4. [Uncertainty Estimation (EDL)](#uncertainty-estimation-edl)
5. [Quick Start](#quick-start)
6. [Project Structure](#project-structure)
7. [Security](#security)

---

## Model Overview

| Item | Description |
|------|------|
| **Task** | Pixel-level crop classification (7 classes: wheat / corn / rice / soybean / cotton / vegetable / other) |
| **Input** | Sentinel-2 Optical (10 bands × 12 time steps) + Sentinel-1 SAR (5 channels × 12 time steps) + DEM (5 features) |
| **Output** | Classification map + uncertainty heatmaps |
| **Parameters** | ~49M |
| **Backbone** | ResNet50 / ConvNeXt-T / EfficientNet-B0/B4 (pluggable) |
| **Uncertainty** | Dirichlet EDL + MC-Dropout + TTA |
| **Data** | Currently synthetic; real data training is in progress |

> 📝 The model architecture draws inspiration from several papers in remote sensing and computer vision, with my own combinations and experiments. See [Acknowledgements](#acknowledgements).

---

## Version History

Over time I've iterated through several versions, each trying out new ideas:

```
V1 ─── V4 ─── V5 ─── V5EDL ─── V5Pro ─── V6 (current)
```

| Version | What I tried | Main changes |
|------|----------|----------|
| V1 | Getting started | Optical+SAR dual-modal fusion + single-path temporal encoder |
| V4 | Adding DEM | Tri-modal + dual-path temporal + Dropout for uncertainty |
| V5 | Code cleanup | Refactored components, fixed early design issues |
| V5EDL | Uncertainty | Added Dirichlet EDL, vacuity/dissonance outputs |
| V5Pro | Polishing | Pluggable backbone + CARAFE upsampling + multi-scale fusion |
| V6 | More experiments | Hierarchical multi-scale + multi-task + temporal encoding improvements |

> Many V6 components are still experimental. See `docs/` for design notes.

---

## System Overview

The project roughly follows a training-to-deployment pipeline (parts are still under construction):

```
User Layer:      Vue Dashboard / Gradio Demo / FastAPI
Inference Layer: Model inference + calibration + uncertainty viz
Training Layer:  TwoPhase trainer + AMP + checkpoint resume
Model Layer:     FusionCropNet (CNN + Transformer temporal + multi-scale fusion)
Data Layer:      Preprocessing pipeline + LRU cache + async loading
```

---

## Uncertainty Estimation (EDL)

I experimented with Dirichlet-based **Evidential Deep Learning**, which produces three uncertainty measures:

| Metric | Meaning | Potential use |
|------|------|----------|
| Vacuity | Model is uncertain about this input — might be an unfamiliar pattern | Flag cloud-covered or missing data areas |
| Dissonance | Model "can't decide" between two classes | Identify ambiguous class boundaries |
| Class Variance | Per-class uncertainty level | Find hard samples |

MC Dropout + TTA are used at inference to reduce single-pass stochasticity. These are techniques I learned from papers and tried applying to my own data.

---

## Quick Start

> ⚠️ Currently uses synthetic data for demonstration. Real classification requires weights trained on real data.

### Setup

```bash
git clone https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification.git
cd project1.remote-sensing-classification
python install.py
```

### Online Demo

Visit https://jjjj111qq111-fusioncropnet-v6.hf.space — try it out without installing anything (uses synthetic data for pipeline demo).

### Local Usage

```bash
# Gradio demo
python demo_app.py

# CLI inference
python scripts/predict.py --model V6 --input your_data.npy
```

### API (experimental)

```bash
uvicorn api.main:app --port 8000
# Docs at http://localhost:8000/docs
```

---

## Project Structure

```
project1/
├── models/              # Model code (V1 through V6 evolution)
│   ├── _base.py         # Shared components
│   ├── fusion_net_v5_edl.py  # EDL-based version
│   ├── fusion_net_v6.py      # Latest experiments
│   └── ...
├── data/                # Preprocessing and data loading
├── utils/               # Training, calibration, loss functions
├── api/                 # FastAPI backend
├── frontend/            # Vue dashboard
├── tests/               # Test suite
├── scripts/             # Training and inference scripts
└── docs/                # Design notes and documentation
```

---

## Security

This is a public repository. Safeguards in place:
- `.env` is gitignored; only `.env.example` with placeholders is committed
- Database passwords use environment variables
- SQL queries are parameterized
- No real keys, tokens, or production credentials are stored

If you find a security issue, please open an Issue — thank you.

---

## Acknowledgements

This project was inspired by and learned from (non-exhaustive):

- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
- Sentinel-2 / Sentinel-1 processing workflows adapted from Copernicus documentation
- Remote sensing pre-training: SeCo (Seasonal Contrast)
- Model components reference: TorchVision and timm libraries

Thank you to the open source community 🙏

---

*MIT License. A student's remote sensing learning project — feedback and advice are always appreciated.*
