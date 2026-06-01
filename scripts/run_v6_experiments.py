#!/usr/bin/env python
# =============================================================================
# scripts/run_v6_experiments.py
# V6 Comprehensive Experiment Suite
#
# Six experiment categories for paper-ready analysis:
#   1. Multi-modal combination ablation (7 configurations)
#   2. Fusion mechanism ablation (5 configurations)
#   3. Feature derivation ablation (4 feature groups)
#   4. Robustness analysis (cloud / timesteps / noise stress)
#   5. Model component ablation (7 V6 blocks)
#   6. Confusion matrix & per-class analysis
#
# Usage:
#   python scripts/run_v6_experiments.py --synthetic          # quick test
#   python scripts/run_v6_experiments.py --synthetic --all    # all experiments
#   python scripts/run_v6_experiments.py --exp 1,2,6          # specific experiments
#   python scripts/run_v6_experiments.py --model checkpoints/best.pth --data data/processed
# =============================================================================
import argparse, json, os, sys, time, itertools
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
from utils.metrics import compute_metrics
from utils.calibration import calibration_report
from utils.interpretability import modality_ablation, spectral_band_importance, confusion_region_analysis
from utils.dem_ablation import DEMAblationRunner, INJECTION_POINTS

CROP_NAMES = {0:"Background", 1:"Winter Wheat", 2:"Summer Corn", 3:"Rice",
              4:"Soybean", 5:"Cotton", 6:"Other"}

# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="V6 Comprehensive Experiment Suite")
    p.add_argument("--model", type=str, default="", help="Path to trained checkpoint (.pth)")
    p.add_argument("--data", type=str, default="", help="Path to processed data dir")
    p.add_argument("--output", type=str, default="./v6_experiments_output", help="Output directory")
    p.add_argument("--device", type=str, default="cpu", help="Device: cuda / cpu")
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    p.add_argument("--synthetic-size", type=int, default=64, help="Spatial size for synthetic")
    p.add_argument("--synthetic-temporal", type=int, default=12, help="Temporal steps")
    p.add_argument("--exp", type=str, default="1,2,3,4,5,6",
                   help="Comma-separated experiment numbers (1-6). Default: all.")
    p.add_argument("--all", action="store_true", help="Run all experiments")
    return p.parse_args()

# ── Data ────────────────────────────────────────────────────────────────────
def make_synthetic(args, device):
    H = W = args.synthetic_size
    T = args.synthetic_temporal
    K = args.num_classes
    opt_seq = torch.randn(1, T, 10, H, W, device=device)
    sar_seq = torch.randn(1, T, 5, H, W, device=device)
    dem = torch.randn(1, 5, H, W, device=device)
    doy = torch.linspace(0, 1, T, device=device).unsqueeze(0)
    labels = torch.randint(1, K, (1, H, W))
    return opt_seq, sar_seq, dem, doy, labels


def load_model(args, device):
    m = FusionCropNetV5EDL(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=args.num_classes,
        feat_dim=512, backbone="resnet50", pretrained=False,
        use_v6_enhancements=True).to(device)
    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        m.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"Loaded checkpoint: {args.model}")
    else:
        print("No checkpoint — using random weights (results are for validation only)")
    m.eval()
    return m


def fast_infer(model, opt, sar, dem, doy, device, **kw):
    """Single-pass inference returning predictions and metrics."""
    with torch.no_grad():
        alpha = model(opt, sar, dem, doy, **kw)
    res = dirichlet_to_predictions(alpha.detach())
    preds = res["pred_class"].squeeze(0)
    return preds, alpha.detach(), res


def fast_metrics(model, opt, sar, dem, doy, labels, device, **kw):
    preds, alpha, res = fast_infer(model, opt, sar, dem, doy, device, **kw)
    lbl = labels.squeeze(0) if labels.dim() == 3 else labels
    metrics = compute_metrics(preds, lbl, model.edl_head.num_classes)
    metrics["vacuity_mean"] = float(res["vacuity"].mean())
    metrics["dissonance_mean"] = float(res["dissonance"].mean())
    return metrics, preds, alpha, res


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Multi-Modal Combination Ablation
# ═══════════════════════════════════════════════════════════════════════════
def exp_modality_ablation(model, opt, sar, dem, doy, labels, device, K):
    print("\n" + "="*60)
    print("  Experiment 1: Multi-Modal Combination Ablation")
    print("="*60)
    configs = [
        ("Full (Opt+SAR+DEM)",    (True, True, True)),
        ("Optical Only",          (True, False, False)),
        ("SAR Only",              (False, True, False)),
        ("DEM Only",              (False, False, True)),
        ("Opt+SAR (no DEM)",      (True, True, False)),
        ("Opt+DEM (no SAR)",      (True, False, True)),
        ("SAR+DEM (no Optical)",  (False, True, True)),
    ]
    results = {}
    for name, mask in configs:
        t0 = time.time()
        mets, preds, alpha, _ = fast_metrics(model, opt, sar, dem, doy, labels, device,
                                              modality_mask=mask)
        mets["preds"] = preds.cpu().numpy().tolist()
        mets["time_s"] = round(time.time() - t0, 2)
        results[name] = mets
        print(f"  {name:30s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}  "
              f"Vacuity={mets['vacuity_mean']:.4f}  ({mets['time_s']}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Fusion Mechanism Ablation
# ═══════════════════════════════════════════════════════════════════════════
def exp_fusion_ablation(model, opt, sar, dem, doy, labels, device, K):
    print("\n" + "="*60)
    print("  Experiment 2: Fusion Mechanism Ablation")
    print("="*60)
    configs = [
        ("Full Fusion (all on)",     {}),
        ("No Cross-Modal Attn",      {"cross_modal": False}),
        ("No Late Fusion",           {"late_fusion": False}),
        ("No Early Fusion",          {"early_fusion": False}),
        ("Late Fusion Only",         {"cross_modal": False, "early_fusion": False}),
        ("Cross-Modal Only",         {"late_fusion": False, "early_fusion": False}),
        ("Early Fusion Only",        {"cross_modal": False, "late_fusion": False}),
        ("No Fusion (concat)",       {"cross_modal": False, "late_fusion": False, "early_fusion": False}),
    ]
    results = {}
    for name, fmask in configs:
        t0 = time.time()
        mets, preds, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device,
                                          fusion_mask=fmask)
        mets["time_s"] = round(time.time() - t0, 2)
        results[name] = mets
        print(f"  {name:30s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}  ({mets['time_s']}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: Feature Derivation Ablation
# ═══════════════════════════════════════════════════════════════════════════
OPTICAL_BANDS = ["B2_Blue","B3_Green","B4_Red","B5_RedEdge","B6_RedEdge2",
                 "B7_RedEdge3","B8_NIR","B8A_NarrowNIR","B11_SWIR1","B12_SWIR2"]
DERIVED_OPTICAL = ["NDVI","NDWI","EVI","LSWI","BSI","NBR"]  # hypothetical indices
SAR_BANDS = ["VV","VH","VV_VH_ratio","RVI","NLI"]

def zero_bands(seq, indices):
    """Zero out specific channel indices in an optical/SAR sequence."""
    out = seq.clone()
    for idx in indices:
        out[:, :, idx] = 0.0
    return out


def exp_feature_ablation(model, opt, sar, dem, doy, labels, device, K):
    print("\n" + "="*60)
    print("  Experiment 3: Feature Derivation Ablation")
    print("="*60)
    # Note: indices 0-9 are raw optical bands; derived features would be
    # channels 10+ if pre-computed. Here we test occlusion at band group level.
    configs = [
        ("All bands (baseline)",     None, None),
        ("No NIR bands (idx 6,7)",   [6, 7], None),
        ("No SWIR bands (idx 8,9)",  [8, 9], None),
        ("No RedEdge (idx 3,4,5)",   [3, 4, 5], None),
        ("Visible only (0,1,2)",     list(range(3, 10)), None),
        ("No SAR ratios (idx 2,3,4)", None, [2, 3, 4]),
        ("SAR VV+VH only (no ratios)", None, [0, 1]),
    ]
    results = {}
    for name, opt_zeros, sar_zeros in configs:
        t0 = time.time()
        o = opt if opt_zeros is None else zero_bands(opt, opt_zeros)
        s = sar if sar_zeros is None else zero_bands(sar, sar_zeros)
        mets, _, _, _ = fast_metrics(model, o, s, dem, doy, labels, device)
        mets["time_s"] = round(time.time() - t0, 2)
        results[name] = mets
        print(f"  {name:30s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}  ({mets['time_s']}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4: Robustness Analysis
# ═══════════════════════════════════════════════════════════════════════════
def exp_robustness(model, opt, sar, dem, doy, labels, device, K):
    print("\n" + "="*60)
    print("  Experiment 4: Robustness Analysis")
    print("="*60)
    results = {}
    B, T, C, H, W = opt.shape

    # 4a: Cloud cover stress
    print("\n  --- 4a: Cloud Cover Stress ---")
    cloud_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
    cloud_results = {}
    for level in cloud_levels:
        cm = torch.rand(B, T, H, W, device=device) < level
        mets, _, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device,
                                      cloud_mask=cm)
        cloud_results[f"cloud_{level:.2f}"] = mets
        print(f"    cloud={level:.0%}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}")
    results["cloud_cover"] = cloud_results

    # 4b: Missing timestep stress
    print("\n  --- 4b: Missing Timestep Stress ---")
    ts_missing = [0, 1, 2, 4, 6, 8]
    ts_results = {}
    for n_miss in ts_missing:
        if n_miss == 0:
            mets, _, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device)
        else:
            mask_indices = torch.randperm(T, device=device)[:n_miss]
            o_masked = opt.clone()
            s_masked = sar.clone()
            for idx in mask_indices:
                o_masked[:, idx] = 0.0
                s_masked[:, idx] = 0.0
            mets, _, _, _ = fast_metrics(model, o_masked, s_masked, dem, doy, labels, device)
        ts_results[f"missing_{n_miss}"] = mets
        print(f"    missing={n_miss}/{T}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}")
    results["missing_timesteps"] = ts_results

    # 4c: Noise injection stress
    print("\n  --- 4c: Noise Injection Stress ---")
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    noise_results = {}
    for sigma in noise_levels:
        if sigma == 0:
            mets, _, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device)
        else:
            o_noisy = opt + torch.randn_like(opt) * sigma
            s_noisy = sar + torch.randn_like(sar) * sigma
            mets, _, _, _ = fast_metrics(model, o_noisy, s_noisy, dem, doy, labels, device)
        noise_results[f"noise_{sigma:.2f}"] = mets
        print(f"    sigma={sigma:.2f}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}")
    results["noise"] = noise_results

    # 4d: Per-modality noise
    print("\n  --- 4d: Per-Modality Noise (σ=0.1) ---")
    for mod_name, o_n, s_n in [("clean", opt, sar),
                                ("opt_noise", opt + torch.randn_like(opt)*0.1, sar),
                                ("sar_noise", opt, sar + torch.randn_like(sar)*0.1),
                                ("both_noise", opt + torch.randn_like(opt)*0.1,
                                              sar + torch.randn_like(sar)*0.1)]:
        mets, _, _, _ = fast_metrics(model, o_n, s_n, dem, doy, labels, device)
        results[f"modality_noise_{mod_name}"] = mets
        print(f"    {mod_name:15s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 5: Model Component Ablation (V6 block_mask)
# ═══════════════════════════════════════════════════════════════════════════
BLOCK_NAMES = {
    "temporal_lite": "Block1: TemporalLite",
    "early_fusion": "Block2: Early Fusion",
    "dem_opt_cond": "Block3: DEM→Opt FiLM",
    "temporal_bias": "Block3: Temporal Bias",
    "multi_scale_cross_attn": "Block4: Multi-Scale Cross-Attn",
    "multi_task": "Block5: Multi-Task Heads",
    "scene_head": "Block7: Scene Head",
}


def exp_component_ablation(model, opt, sar, dem, doy, labels, device, K):
    print("\n" + "="*60)
    print("  Experiment 5: Model Component Ablation")
    print("="*60)
    results = {}

    # Baseline: all blocks on
    mets, _, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device)
    results["V6 Full (all blocks)"] = mets
    print(f"  {'V6 Full':30s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}")

    # Individual block removal (leave-one-out)
    for key, label in BLOCK_NAMES.items():
        bm = {k: True for k in BLOCK_NAMES}
        bm[key] = False
        mets, _, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device,
                                      block_mask=bm)
        results[f"no_{key}"] = mets
        delta = results["V6 Full (all blocks)"]["mIoU"] - mets["mIoU"]
        print(f"  {'No ' + label:30s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}  Δ={delta:+.4f}")

    # Cumulative: V5EDL baseline + add blocks one by one
    cumul_blocks = ["temporal_lite", "early_fusion", "dem_opt_cond",
                    "temporal_bias", "multi_scale_cross_attn", "multi_task", "scene_head"]
    bm_cumul = {k: False for k in cumul_blocks}
    mets, _, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device,
                                  block_mask=bm_cumul)
    results["V5EDL (no V6 blocks)"] = mets
    print(f"  {'V5EDL baseline':30s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}")

    for block in cumul_blocks:
        bm_cumul[block] = True
        mets, _, _, _ = fast_metrics(model, opt, sar, dem, doy, labels, device,
                                      block_mask=dict(bm_cumul))
        results[f"+{block}"] = mets
        print(f"  {'+' + BLOCK_NAMES[block]:30s}  OA={mets['OA']:.4f}  mIoU={mets['mIoU']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 6: Confusion Matrix & Per-Class Analysis
# ═══════════════════════════════════════════════════════════════════════════
def exp_confusion_analysis(model, opt, sar, dem, doy, labels, device, K):
    print("\n" + "="*60)
    print("  Experiment 6: Confusion Matrix & Per-Class Analysis")
    print("="*60)
    results = {}

    # Full model inference
    preds, alpha, res = fast_infer(model, opt, sar, dem, doy, device)
    lbl = labels.squeeze(0) if labels.dim() == 3 else labels
    metrics = compute_metrics(preds, lbl, K)

    results["global"] = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    results["confusion_matrix"] = metrics["confusion_matrix"]
    results["confusion_matrix_normalized"] = None  # computed below

    # Normalize confusion matrix by row (true class)
    cm = np.array(metrics["confusion_matrix"])
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    results["confusion_matrix_normalized"] = cm_norm.tolist()

    # Per-class report
    per_class = {}
    for cls in range(K):
        tp = cm[cls, cls]
        support = int(cm[cls].sum())
        prec = float(tp / max(cm[:, cls].sum(), 1))
        rec = float(tp / max(support, 1))
        per_class[CROP_NAMES.get(cls, f"Class_{cls}")] = {
            "precision": prec, "recall": rec,
            "f1": float(2*prec*rec/max(prec+rec, 1e-10)),
            "iou": float(metrics["IoU_per_class"][cls-1]) if cls > 0 else 0.0,
            "support": support,
            "users_accuracy": prec,  # same as precision
            "producers_accuracy": rec,  # same as recall
        }
    results["per_class"] = per_class

    # Calibration report (labels already has batch dim from make_synthetic)
    try:
        cal = calibration_report(alpha, labels, num_classes=K,
                                  class_names=[CROP_NAMES[i] for i in range(K)])
        results["calibration"] = {k: v for k, v in cal.items()
                                  if not k.startswith("_") and k != "PerClass"}
    except Exception as e:
        results["calibration"] = {"error": str(e)}

    # Confusion region analysis
    try:
        cra = confusion_region_analysis(
            alpha, labels.to(device), num_classes=K)
        results["confusion_regions"] = {k: v for k, v in cra.items()
                                        if not k.startswith("_")}
    except Exception as e:
        results["confusion_regions"] = {"error": str(e)}

    # Print summary
    print(f"  OA={metrics['OA']:.4f}  mIoU={metrics['mIoU']:.4f}  Kappa={metrics['Kappa']:.4f}")
    print(f"  Per-class IoU: " +
          " | ".join(f"{CROP_NAMES.get(i+1, f'C{i+1}')}:{metrics['IoU_per_class'][i]:.3f}"
                     for i in range(K-1)))
    if "vacuity_mean" in res:
        print(f"  Vacuity mean={res['vacuity_mean']:.4f}  Dissonance mean={res['dissonance_mean']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    exp_list = [int(x.strip()) for x in args.exp.split(",") if x.strip().isdigit()]

    print("="*60)
    print("  V6 Comprehensive Experiment Suite")
    print(f"  Device: {device}  Output: {args.output}")
    print(f"  Experiments: {exp_list}")
    print("="*60)

    model = load_model(args, device)
    if args.synthetic or not args.data:
        opt, sar, dem, doy, labels = make_synthetic(args, device)
    else:
        # Real data loading (placeholder — extend with actual data loader)
        raise NotImplementedError("Real data loading — extend with your data pipeline")

    K = args.num_classes
    all_results = {"meta": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model.__class__.__name__,
        "params_M": round(sum(p.numel() for p in model.parameters())/1e6, 1),
        "device": str(device),
        "synthetic": args.synthetic,
    }}

    t_start = time.time()

    if 1 in exp_list:
        all_results["exp1_modality_ablation"] = exp_modality_ablation(
            model, opt, sar, dem, doy, labels, device, K)

    if 2 in exp_list:
        all_results["exp2_fusion_ablation"] = exp_fusion_ablation(
            model, opt, sar, dem, doy, labels, device, K)

    if 3 in exp_list:
        all_results["exp3_feature_ablation"] = exp_feature_ablation(
            model, opt, sar, dem, doy, labels, device, K)

    if 4 in exp_list:
        all_results["exp4_robustness"] = exp_robustness(
            model, opt, sar, dem, doy, labels, device, K)

    if 5 in exp_list:
        all_results["exp5_component_ablation"] = exp_component_ablation(
            model, opt, sar, dem, doy, labels, device, K)

    if 6 in exp_list:
        all_results["exp6_confusion_analysis"] = exp_confusion_analysis(
            model, opt, sar, dem, doy, labels, device, K)

    elapsed = time.time() - t_start
    all_results["meta"]["elapsed_s"] = round(elapsed, 1)
    print(f"\n{'='*60}")
    print(f"  All experiments complete. Total time: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Save
    json_path = os.path.join(args.output, "v6_experiments_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {json_path}")
    print(f"Run visualization: python scripts/visualize_v6_experiments.py --results {json_path}")


if __name__ == "__main__":
    main()
