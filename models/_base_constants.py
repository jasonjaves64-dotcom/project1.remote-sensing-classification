"""_base_constants.py — Canonical dimension constants for the FusionCropNet family.

All magic numbers (64, 128, 256, 512, etc.) are defined ONCE here.
Individual modules import from this file instead of hardcoding values.

When changing backbone or resolution, update the constants here — all
downstream components pick up the change automatically.
"""

# ── DEM Encoder ─────────────────────────────────────────────────────────
DEM_CH = 128                # DEM feature embedding dimension

# ── Backbone / Feature Pyramid ───────────────────────────────────────────
FEAT_DIM = 512              # Optical backbone output channels (H/4 scale)
FEAT_DIM_HALF = 256         # FP2 output (H/2 scale) = FEAT_DIM // 2

# ── SAR Encoder ──────────────────────────────────────────────────────────
SAR_S1_CH = 64              # SAR stage 1 output (H scale)
SAR_S2_CH = 128             # SAR stage 2 output (H/2 scale)
SAR_S3_CH = 512             # SAR stage 3 output (H/4 scale) = FEAT_DIM

# ── Decoder ──────────────────────────────────────────────────────────────
PRE_HEAD_CH = 64            # Pre-head feature channels (decoder output)
SKIP_PROJ_CH = 64           # Skip connection projection channels

# ── Early Fusion ─────────────────────────────────────────────────────────
EARLY_FUSION_IN_CH = 20     # 10 (opt) + 5 (sar) + 5 (dem) = 20 channels
EARLY_FUSION_CH = 128       # Early fusion output channels

# ── Multi-task Heads ─────────────────────────────────────────────────────
LAI_HIDDEN = 32             # LAI regression head hidden dim
GROWTH_HIDDEN = 32          # Growth stage head hidden dim
BOUNDARY_HIDDEN = 32        # Boundary detection head hidden dim
SCENE_HIDDEN = 256          # LightSceneHead hidden dim

# ── Cross-Modal Attention ────────────────────────────────────────────────
CMA_N_HEADS_H = 1           # H scale cross-modal attention heads
CMA_N_HEADS_H2 = 4          # H/2 scale
CMA_N_HEADS_H4 = 8          # H/4 scale (full CrossModalAttention)

# ── Temporal ─────────────────────────────────────────────────────────────
TEMPORAL_KERNEL_SIZE = 3    # TemporalLite convolution kernel (1.5 month window)
N_FREQS_DEFAULT = 4         # Fourier DOY encoding frequencies

# ── Model Defaults ───────────────────────────────────────────────────────
NUM_CLASSES_DEFAULT = 7
OPT_CH_DEFAULT = 10
SAR_CH_DEFAULT = 5
DEM_CH_IN_DEFAULT = 5
MAX_OBS_DEFAULT = 24
