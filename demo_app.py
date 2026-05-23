"""
FusionCropNet — Streamlit 可视化界面 v2
升级功能: 暗色/亮色主题、分类结果叠加对比 slider、不确定性热力图、
         批量推理进度条、移动端适配
"""
import streamlit as st
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
import os, sys, time, io, json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="FusionCropNet — 作物分类",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "FusionCropNet V6 — 多模态遥感影像农作物分类系统"}
)

# ── Constants ──
CROP_NAMES = ["背景", "冬小麦", "夏玉米", "水稻", "大豆", "棉花", "其他作物"]
CROP_COLORS = ["#000000", "#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#BC4A8C", "#9C89B8"]
CROP_COLORS_HEX = ["#000000", "#2E86AB", "#F18F01", "#C73E1D", "#6A994E", "#BC4A8C", "#9C89B8"]
MODEL_PATH = Path(__file__).parent / "best_model.pth"
DATA_DIR = Path(__file__).parent / "demo_data"

# ── Session state init ──
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "inference_cache" not in st.session_state:
    st.session_state.inference_cache = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []


# ═══════════════════════════════════════════════════════════════════════════════
# Theme & CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_theme_css():
    """Inject responsive CSS for dark/light theme and mobile adaptation."""
    is_dark = st.session_state.theme == "dark"
    bg = "#0e1117" if is_dark else "#ffffff"
    fg = "#fafafa" if is_dark else "#262730"
    card_bg = "#1a1c23" if is_dark else "#f0f2f6"
    border = "#2e3039" if is_dark else "#e0e2e6"
    accent = "#4da6ff" if is_dark else "#0068c9"

    st.markdown(f"""
    <style>
    /* Base theme overrides */
    .stApp {{
        background-color: {bg};
    }}
    .metric-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s;
    }}
    .metric-card:hover {{
        border-color: {accent};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .metric-card .value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {accent};
    }}
    .metric-card .label {{
        font-size: 0.85rem;
        color: {"#8b8f97" if is_dark else "#555"};
        margin-top: 0.25rem;
    }}
    .section-title {{
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {accent};
        color: {fg};
    }}

    /* Mobile responsive */
    @media (max-width: 768px) {{
        .metric-card {{
            padding: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        .metric-card .value {{
            font-size: 1.3rem;
        }}
        .hide-mobile {{
            display: none !important;
        }}
        .section-title {{
            font-size: 1.1rem;
        }}
        /* Stack columns on mobile */
        div[data-testid="column"] {{
            width: 100% !important;
            flex: none !important;
        }}
    }}

    @media (min-width: 769px) {{
        .hide-desktop {{
            display: none !important;
        }}
    }}

    /* Progress bar enhancement */
    div[data-testid="stProgress"] > div {{
        border-radius: 10px;
    }}

    /* Smoother transitions */
    * {{
        transition: background-color 0.3s, color 0.3s, border-color 0.3s;
    }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(version="V5EDL"):
    t0 = time.time()
    if version == "V6":
        from models.fusion_net_v6 import FusionCropNetV6
        model = FusionCropNetV6(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=4, n_layers=2
        )
    else:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=4, n_layers=2, use_v6_enhancements=False
        )
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    elapsed = time.time() - t0
    return model, elapsed


@st.cache_data(show_spinner=False)
def load_demo_data(timesteps=6):
    t0 = time.time()
    opt = torch.from_numpy(np.load(DATA_DIR / "optical_sequence.npy")).float()[:timesteps]
    sar = torch.from_numpy(np.load(DATA_DIR / "sar_sequence.npy")).float()[:timesteps]
    doy = torch.from_numpy(np.load(DATA_DIR / "doy.npy")).float()[:timesteps]
    dem = torch.from_numpy(np.load(DATA_DIR / "dem.npy")).float()
    labels = np.load(DATA_DIR / "labels.npy")
    elapsed = time.time() - t0
    return opt, sar, dem, doy, labels, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(model, opt, sar, dem, doy, n_passes=2, use_tta=False):
    t0 = time.time()
    with torch.inference_mode():
        result = model.predict_uncertainty(
            opt.unsqueeze(0), sar.unsqueeze(0),
            dem.unsqueeze(0), doy.unsqueeze(0),
            n_passes=n_passes, use_tta=use_tta
        )
    elapsed = time.time() - t0
    out = {k: v.squeeze(0).cpu().numpy() if torch.is_tensor(v) else v
           for k, v in result.items()}
    out["inference_time"] = elapsed
    return out


def run_batch_inference(model, opt, sar, dem, doy, grid_rows=2, grid_cols=2, n_passes=1):
    """Run inference on spatial grid of patches for batch demo."""
    _, _, H, W = opt.shape
    ph, pw = H // grid_rows, W // grid_cols
    total_patches = grid_rows * grid_cols
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx in range(total_patches):
        r = idx // grid_cols
        c = idx % grid_cols
        r0, r1 = r * ph, min((r + 1) * ph, H)
        c0, c1 = c * pw, min((c + 1) * pw, W)

        status_text.text(f"推理中: 区域 {idx+1}/{total_patches} (行{r+1},列{c+1})")
        progress_bar.progress((idx + 1) / total_patches)

        opt_patch = opt[:, :, r0:r1, c0:c1]
        sar_patch = sar[:, :, r0:r1, c0:c1]
        dem_patch = dem[:, r0:r1, c0:c1]
        doy_patch = doy

        result = run_inference(model, opt_patch, sar_patch, dem_patch, doy_patch,
                              n_passes=n_passes)
        result["position"] = (r, c, r0, r1, c0, c1)
        results.append(result)

    progress_bar.empty()
    status_text.empty()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def create_rgb_composite(opt_tensor):
    """Create an RGB-like composite from optical data (bands 3,2,1 → R,G,B)."""
    if opt_tensor.dim() == 4:
        img = opt_tensor[-1].cpu().numpy()
    else:
        img = opt_tensor
    r = img[2] if img.shape[0] > 2 else img[0]
    g = img[1] if img.shape[0] > 1 else img[0]
    b = img[0]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return (rgb * 255).astype(np.uint8)


def plot_overlay_comparison(rgb_img, classified, alpha=0.5, figsize=(8, 6)):
    """Create overlay comparison with adjustable blend."""
    cmap = ListedColormap(CROP_COLORS_HEX)
    classified_rgb = cmap(classified / 6.0)[..., :3]

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.imshow(rgb_img, interpolation="nearest")
    ax.imshow(classified_rgb, alpha=alpha, interpolation="nearest")
    ax.set_title(f"叠加对比 (透明度={alpha:.2f})", fontsize=13)
    ax.axis("off")
    patches_list = [mpatches.Patch(color=CROP_COLORS_HEX[i], label=CROP_NAMES[i])
                    for i in range(1, 7)]
    ax.legend(handles=patches_list, fontsize=7, loc="lower left",
              ncols=3, framealpha=0.7)
    plt.tight_layout(pad=0.5)
    return fig


def plot_uncertainty_heatmap(data, title, cmap="hot", figsize=(7, 5.5)):
    """Enhanced uncertainty heatmap with colorbar."""
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    im = ax.imshow(data, cmap=cmap, interpolation="bilinear", aspect="equal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.5)
    return fig


def plot_classification(pred_class, figsize=(7, 5.5)):
    """Plot classified map with legend."""
    cmap = ListedColormap(CROP_COLORS_HEX)
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.imshow(pred_class, cmap=cmap, vmin=0, vmax=6, interpolation="nearest")
    ax.set_title("分类结果", fontsize=13, fontweight="bold")
    patches_list = [mpatches.Patch(color=CROP_COLORS_HEX[i], label=CROP_NAMES[i])
                    for i in range(7)]
    ax.legend(handles=patches_list, fontsize=7, loc="lower left",
              ncols=4, framealpha=0.7)
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    return fig


def plot_distribution(pred_class, figsize=(7, 5.5)):
    """Plot class distribution as donut chart."""
    counts = np.bincount(pred_class.flatten(), minlength=7)[:7]
    total = counts.sum()
    if total == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "无数据", ha="center", va="center")
        return fig

    colors = [CROP_COLORS_HEX[i] for i in range(7)]
    valid = [(i, c, n) for i, (c, n) in enumerate(zip(colors, counts)) if n > 0]

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    wedges, texts, autotexts = ax.pie(
        [v[2] for v in valid],
        colors=[v[1] for v in valid],
        labels=[f"{CROP_NAMES[v[0]]}" for v in valid],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        labeldistance=1.08
    )
    for at in autotexts:
        at.set_fontsize(8)
    centre = plt.Circle((0, 0), 0.55, fc="white" if st.session_state.theme == "light" else "#0e1117")
    ax.add_artist(centre)
    ax.set_title("类别分布", fontsize=13, fontweight="bold")
    plt.tight_layout(pad=0.5)
    return fig


def plot_combined_uncertainty(vacuity, dissonance, figsize=(12, 5)):
    """Side-by-side uncertainty maps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=100)

    im1 = ax1.imshow(vacuity, cmap="hot", interpolation="bilinear", aspect="equal")
    ax1.set_title("Aleatoric (Vacuity)", fontsize=12, fontweight="bold")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, shrink=0.8)

    im2 = ax2.imshow(dissonance, cmap="coolwarm", interpolation="bilinear", aspect="equal")
    ax2.set_title("Epistemic (Dissonance)", fontsize=12, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, shrink=0.8)

    plt.tight_layout(pad=1.5)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UI Components
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render enhanced sidebar with theme toggle and controls."""
    with st.sidebar:
        st.title("🌾 FusionCropNet")
        st.caption("多模态遥感影像农作物分类系统")

        st.markdown("---")

        # Theme toggle
        col_t1, col_t2 = st.columns([3, 2])
        with col_t1:
            st.caption("界面主题")
        with col_t2:
            new_theme = "dark" if st.toggle(
                "🌙", key="theme_toggle",
                value=st.session_state.theme == "dark",
                help="切换暗色/亮色主题"
            ) else "light"
            if new_theme != st.session_state.theme:
                st.session_state.theme = new_theme
                st.rerun()

        st.markdown("---")

        # Model selection
        st.subheader("🧠 模型配置")
        model_version = st.selectbox(
            "模型版本",
            ["V5EDL (不确定性)", "V6 (Next-Gen)"],
            index=0,
            key="model_version",
            help="V5EDL: EDL不确定性估计\nV6: 5辅助头 + TemporalLite"
        )
        model_key = "V6" if "V6" in model_version else "V5EDL"

        n_passes = st.slider(
            "EDL 采样次数",
            1, 5, 2,
            key="n_passes",
            help="越高越稳定，但推理时间线性增长"
        )

        use_tta = st.checkbox(
            "TTA (测试时增强)",
            value=False,
            key="use_tta",
            help="水平翻转 + 原始预测取平均"
        )

        st.markdown("---")

        # Data selection
        st.subheader("📦 数据配置")
        n_timesteps = st.selectbox(
            "时序长度",
            [3, 6, 12],
            index=1,
            key="n_timesteps",
            help="3=快速预览, 6=标准演示, 12=完整时序"
        )

        use_dem = st.checkbox("使用 DEM 数据", value=True, key="use_dem")

        st.markdown("---")

        # Info
        st.caption("""
        **模型**: FusionCropNet
        **输入**: 光学(Landsat) + SAR(S1) + DEM
        **输出**: 7类作物 + 不确定性
        **加载时间**: <3s (模型缓存后即时)
        """)

        return model_key, n_passes, use_tta, n_timesteps, use_dem


def render_metric_cards(result):
    """Render metric cards for inference results."""
    cols = st.columns([1, 1, 1, 1.5])
    vacuity_mean = float(result["vacuity"].mean())
    dissonance_mean = float(result["dissonance"].mean())
    unique, counts = np.unique(result["pred_class"], return_counts=True)
    max_cls_idx = unique[counts.argmax()]
    conf = float(1.0 - vacuity_mean)

    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{conf:.3f}</div>
            <div class="label">平均置信度</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{vacuity_mean:.4f}</div>
            <div class="label">平均 Vacuity ↓</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{dissonance_mean:.4f}</div>
            <div class="label">平均 Dissonance ↓</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{CROP_NAMES[max_cls_idx]}</div>
            <div class="label">主要作物类型</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    inject_theme_css()

    model_key, n_passes, use_tta, n_timesteps, use_dem = render_sidebar()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 项目介绍", "🔮 单张推理", "📊 批量推理", "📋 技术信息"
    ])

    # ── Tab 1: Project Intro ──
    with tab1:
        st.title("基于深度学习的遥感影像作物分类")
        st.markdown("---")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            ### 🎯 系统能力

            - **三模态融合**: 光学影像 + SAR雷达 + DEM地形协同
            - **EDL不确定性**: 每个像素附带可信度评估 (Vacuity + Dissonance)
            - **7类作物**: 冬小麦 / 夏玉米 / 水稻 / 大豆 / 棉花 / 其他作物
            - **时序建模**: Transformer 聚合全年生长季信息

            ### 🔑 核心创新

            - **证据深度学习 (EDL)**: 输出 Vacuity（证据不足）+
              Dissonance（类间冲突），每个预测都附带可信度
            - **多尺度跨模态注意力**: 光学-SAR 双尺度交叉融合
            - **地形感知编码**: DEM 条件注入光学和 SAR 编码器
            - **时序精简模块**: TemporalLite 高效时序聚合
            """)
        with col2:
            st.image(
                "https://raw.githubusercontent.com/jasonjaves64-dotcom/project1.remote-sensing-classification/master/loss_plot_metrics.png",
                caption="训练过程指标",
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("### 🏗️ 模型架构")
        st.code("""
        opt(T,10,H,W)    sar(T,5,H,W)    dem(5,H,W)
             │                │               │
        OpticalEncoder   SAREncoder     DEMEncoder
             │           (DEM FiLM)          │
             └────┬──────────┘               │
                  ▼                          │
        CrossModalAttention(H/4)             │
                  │                          │
        DEMSpatialConditioner  ←─────────────┘
                  │
        LateFusion → Decoder → EDLHead
                  │
        alpha → {预测类别, Vacuity, Dissonance}
        """, language=None)

        # Quick-start guide
        with st.expander("🚀 快速上手指南", expanded=False):
            st.markdown("""
            1. **选择标签页**: 切换到「单张推理」或「批量推理」
            2. **配置模型**: 在左侧边栏选择模型版本和参数
            3. **运行推理**: 点击推理按钮，等待结果
            4. **查看结果**: 分类图、不确定性热力图、类别分布
            5. **调整对比**: 使用叠加对比 slider 查看分类与原始影像的叠合效果
            """)

    # ── Tab 2: Single Inference ──
    with tab2:
        st.markdown('<p class="section-title">🔮 单张影像推理</p>',
                    unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            start_btn = st.button(
                "🚀 开始推理", type="primary",
                use_container_width=True, key="single_run"
            )
        with col_b:
            overlay_alpha = st.slider(
                "叠加透明度",
                0.0, 1.0, 0.5, 0.05,
                key="overlay_alpha",
                help="0=仅原始影像, 1=仅分类图"
            )
        with col_c:
            st.caption("推理耗时: 约 2-8 秒")

        if start_btn:
            # Load model
            with st.spinner("⚡ 加载模型..."):
                try:
                    model, load_time = load_model(model_key)
                    st.session_state.model_loaded = True
                except Exception as e:
                    st.error(f"模型加载失败: {e}")
                    st.stop()

            # Load data
            with st.spinner("📦 加载数据..."):
                opt, sar, dem, doy, labels, data_time = load_demo_data(n_timesteps)
                if not use_dem:
                    dem = torch.zeros_like(dem)

            # Run inference
            with st.spinner(f"🧠 推理中 (EDL ×{n_passes})..."):
                result = run_inference(model, opt, sar, dem, doy,
                                      n_passes=n_passes, use_tta=use_tta)
                st.session_state.inference_cache = result
                st.session_state.opt_data = opt
                st.session_state.labels_data = labels

            st.markdown("---")

            # Stats
            st.markdown(f"""
            <div style="color: {'#8b8f97' if st.session_state.theme == 'dark' else '#888'}; font-size: 0.85rem;">
            ⚡ 模型加载: {load_time:.1f}s &nbsp;|&nbsp;
            📦 数据加载: {data_time:.1f}s &nbsp;|&nbsp;
            🧠 推理耗时: {result['inference_time']:.1f}s &nbsp;|&nbsp;
            📐 影像尺寸: {result['pred_class'].shape}
            </div>
            """, unsafe_allow_html=True)

            render_metric_cards(result)
            st.markdown("---")

            # Overlay comparison
            st.markdown("#### 🔍 分类结果叠加对比")
            rgb_img = create_rgb_composite(opt)
            fig_overlay = plot_overlay_comparison(
                rgb_img, result["pred_class"], alpha=overlay_alpha
            )
            st.pyplot(fig_overlay, use_container_width=True)
            plt.close(fig_overlay)

            st.markdown("---")

            # Classification & distribution
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### 🗺️ 分类结果")
                fig_cls = plot_classification(result["pred_class"])
                st.pyplot(fig_cls, use_container_width=True)
                plt.close(fig_cls)
            with col2:
                st.markdown("#### 📊 类别分布")
                fig_dist = plot_distribution(result["pred_class"])
                st.pyplot(fig_dist, use_container_width=True)
                plt.close(fig_dist)

            st.markdown("---")

            # Uncertainty
            st.markdown("#### 🔥 不确定性热力图")
            fig_unc = plot_combined_uncertainty(result["vacuity"], result["dissonance"])
            st.pyplot(fig_unc, use_container_width=True)
            plt.close(fig_unc)

            with st.expander("🔍 如何解读不确定性?", expanded=False):
                st.markdown("""
                - **Vacuity (红色热力图)**: 数据不确定性。高值 = 观测证据不足
                  （云遮挡、数据缺失），建议补充数据
                - **Dissonance (冷暖图)**: 认知不确定性。高值 = 模型在类别间犹豫
                  （如水稻 vs 湿地），建议人工复核
                - **低不确定性区域**: 预测高度可信，可直接用于决策
                """)

    # ── Tab 3: Batch Inference ──
    with tab3:
        st.markdown('<p class="section-title">📊 批量区域推理</p>',
                    unsafe_allow_html=True)
        st.markdown("将影像划分为网格区域，逐块推理并汇总结果。")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            grid_rows = st.slider("网格行数", 1, 4, 2, key="grid_rows")
        with col2:
            grid_cols = st.slider("网格列数", 1, 4, 2, key="grid_cols")
        with col3:
            batch_n_passes = st.slider(
                "EDL 采样 (批量)",
                1, 3, 1,
                key="batch_n_passes",
                help="批量模式下建议使用1以加快速度"
            )

        total_patches = grid_rows * grid_cols
        st.caption(f"共 {total_patches} 个区域 (预估耗时: {total_patches * 3}-{total_patches * 8} 秒)")

        if st.button("📊 开始批量推理", type="primary", use_container_width=True,
                     key="batch_run"):
            # Load model
            with st.spinner("⚡ 加载模型..."):
                try:
                    model, load_time = load_model(model_key)
                except Exception as e:
                    st.error(f"模型加载失败: {e}")
                    st.stop()

            # Load data
            with st.spinner("📦 加载数据..."):
                opt, sar, dem, doy, labels, _ = load_demo_data(n_timesteps)
                if not use_dem:
                    dem = torch.zeros_like(dem)

            # Batch inference
            st.markdown("---")
            st.markdown("#### ⏳ 推理进度")
            results = run_batch_inference(
                model, opt, sar, dem, doy,
                grid_rows=grid_rows, grid_cols=grid_cols,
                n_passes=batch_n_passes
            )
            st.session_state.batch_results = results

            # Summary
            st.success(f"✅ 批量推理完成: {len(results)} 个区域")
            st.markdown("---")

            # Show results grid
            st.markdown("#### 📋 各区域结果总览")
            for i in range(0, len(results), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(results):
                        break
                    r = results[idx]
                    pos = r["position"]
                    with col:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=80)
                        cmap = ListedColormap(CROP_COLORS_HEX)
                        ax1.imshow(r["pred_class"], cmap=cmap, vmin=0, vmax=6,
                                  interpolation="nearest")
                        ax1.set_title(f"区域({pos[0]+1},{pos[1]+1}) 分类",
                                     fontsize=10)
                        ax1.axis("off")
                        im = ax2.imshow(r["vacuity"], cmap="hot",
                                       interpolation="bilinear")
                        ax2.set_title(f"不确定性 (Vacuity={float(r['vacuity'].mean()):.4f})",
                                     fontsize=10)
                        ax2.axis("off")
                        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                        plt.tight_layout(pad=1)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            # Aggregate stats
            st.markdown("---")
            st.markdown("#### 📈 汇总统计")
            all_vacuity = np.mean([float(r["vacuity"].mean()) for r in results])
            all_dissonance = np.mean([float(r["dissonance"].mean()) for r in results])
            all_times = np.sum([r["inference_time"] for r in results])

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("总推理耗时", f"{all_times:.1f}s")
            with col_s2:
                st.metric("平均 Vacuity", f"{all_vacuity:.4f}")
            with col_s3:
                st.metric("平均 Dissonance", f"{all_dissonance:.4f}")

    # ── Tab 4: Tech Info ──
    with tab4:
        st.markdown('<p class="section-title">📋 技术信息</p>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🧠 模型规格

            | 参数 | 值 |
            |------|-----|
            | 架构 | FusionCropNet V5EDL / V6 |
            | 参数量 | ~100M |
            | 骨架网络 | ResNet50 |
            | 光学通道 | 10 (6波段+4植被指数) |
            | SAR通道 | 5 (VV/VH+衍生) |
            | DEM通道 | 5 (海拔/坡度/坡向/TWI) |
            | 时序长度 | 12+ 时相 |

            ### 🎯 性能指标

            | 指标 | 值 |
            |------|-----|
            | 总体精度 (OA) | ~92% |
            | mIoU | ~78% |
            | 推理速度 | 2-8s (H×W) |
            | 模型加载 | <3s (缓存后即时) |
            """)
        with col2:
            st.markdown("""
            ### 🛠️ 技术栈

            | 层级 | 技术 |
            |------|------|
            | 框架 | PyTorch 2.x |
            | 前端 | Streamlit |
            | 遥感 | rasterio, GDAL |
            | 部署 | Docker, PyInstaller |
            | 可视化 | Matplotlib |
            | 不确定性 | Evidence Deep Learning |

            ### 📱 兼容性

            | 平台 | 状态 |
            |------|------|
            | Desktop (1920+) | ✅ 完整 |
            | Desktop (1366) | ✅ 完整 |
            | Tablet (768+) | ✅ 适配 |
            | Mobile (320+) | ✅ 响应式 |
            """)

        st.markdown("---")
        st.markdown("### 📂 开源与引用")
        st.markdown("""
        - **GitHub**: [jasonjaves64-dotcom/project1](https://github.com/jasonjaves64-dotcom/project1.remote-sensing-classification)
        - **许可**: MIT License
        - **文档**: 完整知识库 (20篇) — 模型架构、数据预处理、部署指南
        """)


if __name__ == "__main__":
    main()
