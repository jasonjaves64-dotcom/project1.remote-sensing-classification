**English** | [**中文**](README.md)

---

# FusionCropNet — Multi-Modal Remote Sensing Crop Classification

> 🎓 A student's learning project in remote sensing and deep learning. Work in progress — feedback welcome.

This is my attempt at applying deep learning to crop classification using multi-modal remote sensing data. It explores fusing Sentinel-2 optical time series, Sentinel-1 SAR, and DEM data for pixel-level crop mapping, along with Evidential Deep Learning (EDL) for uncertainty estimation. Built to learn — expect rough edges and incomplete experiments.

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-orange)](https://jjjj111qq111-fusioncropnet-v6.hf.space)

---

## ⚠️ Disclaimer

**This is a student side project** built primarily as a learning exercise. Code may have imperfections; experimental conclusions should be taken as tentative. Suggestions and corrections are very welcome — I'm here to learn.

Currently using synthetic data for pipeline validation and ablation experiments. Real data training is pending V6.2.

---

## Ablation Study

To validate design decisions, a systematic **6-category ablation study** (50+ configs, 16 component ablations) was conducted on synthetic data. Full report: [`docs/V6_EXPERIMENTS_REPORT.md`](docs/V6_EXPERIMENTS_REPORT.md).

### Modality Ablation

| Config | mIoU | vs Full |
|------|------|:--:|
| **Full (Opt+SAR+DEM)** | **0.0888** | — |
| Optical Only | 0.0798 | −10.1% |
| Opt+SAR (no DEM) | 0.0757 | −14.8% |

> Tri-modal fusion outperforms any subset. DEM adds 14.8% through cross-modal synergy.

### Fusion Ablation

| Removed | mIoU | Drop |
|------|------|:--:|
| V6 Full | 0.0888 | — |
| **− Early Fusion** | **0.0758** | **−14.6%** |
| − Late Fusion | 0.0841 | −5.3% |

> Early Fusion is the single most critical component.

### V6 Component Contribution

| Step | mIoU | Gain |
|------|------|:--:|
| V5EDL (baseline) | 0.0769 | — |
| + Early Fusion | 0.0861 | **+12.0%** |
| V6 Full | 0.0888 | +15.5% |

> V6 improves **+15.5%** over V5EDL.

![Modality Ablation](v6_experiments_output/figures/fig1a_modality_ablation.png)
![Fusion Ablation](v6_experiments_output/figures/fig2_fusion_ablation.png)
![Component Ablation](v6_experiments_output/figures/fig5_component_ablation.png)
![Summary Dashboard](v6_experiments_output/figures/fig0_summary_dashboard.png)

> ⚠️ **Note**: All experiments on synthetic data. Low absolute mIoU values (~0.09) are from synthetic random labels. **Relative differences** (percentage gains/losses) are the reliable signal. Real data validation pending V6.2.

---

## Quick Start

```bash
git clone https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification.git
cd project1.remote-sensing-classification
python install.py
```

Online demo: https://jjjj111qq111-fusioncropnet-v6.hf.space

---

## Acknowledgements

Inspired by Sensoy et al. (EDL, NeurIPS 2018), SeCo, Copernicus docs, TorchVision, and the open source community.

---

*MIT License. A student's remote sensing learning project — feedback always appreciated.*
