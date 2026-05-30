# =============================================================================
# utils/dem_ablation.py
# DEM Ablation Experiment Framework
#
# Quantifies the contribution of each DEM injection point by selectively
# disabling them and measuring the impact on classification accuracy.
#
# Injection points (6 total, 3 shared V5EDL/V6 + 3 V6-only):
#   V5EDL/V6 shared:
#     sar_film     — SAR encoder Stage 1+2 FiLM modulation
#     spatial_cond — DEMSpatialConditioner on cross-modal fused features
#     decoder_skip — Decoder DEM skip connection (additive residual)
#   V6-only:
#     early_fusion — ModalNormalize + early_fusion conv (raw DEM)
#     opt_cond     — DEMOpticalConditioner dual-scale FiLM on optical features
#     temporal_bias— DEM temporal bias added to optical + SAR token sequences
# =============================================================================
import json
import time
import numpy as np
import torch
from pathlib import Path

INJECTION_POINTS = {
    "sar_film": {
        "name": "SAR FiLM",
        "description": "SAR encoder Stage 1+2 FiLM modulation by DEM features",
        "group": "v5_shared",
        "scale": "H, H/2",
    },
    "spatial_cond": {
        "name": "Spatial Conditioner",
        "description": "DEMSpatialConditioner gated FiLM on cross-modal fused features",
        "group": "v5_shared",
        "scale": "H/4",
    },
    "decoder_skip": {
        "name": "Decoder Skip",
        "description": "Decoder DEM skip connection (1x1 proj + additive residual)",
        "group": "v5_shared",
        "scale": "H/8",
    },
    "early_fusion": {
        "name": "Early Fusion",
        "description": "ModalNormalize + early_fusion conv with raw 5ch DEM",
        "group": "v6_only",
        "scale": "H",
    },
    "opt_cond": {
        "name": "Optical Conditioner",
        "description": "DEMOpticalConditioner dual-scale FiLM on optical backbone features",
        "group": "v6_only",
        "scale": "H/4, H/2",
    },
    "temporal_bias": {
        "name": "Temporal Bias",
        "description": "DEM-derived additive bias on optical + SAR temporal tokens",
        "group": "v6_only",
        "scale": "H/4",
    },
}

ABLATION_GROUPS = {
    "v5_shared": ["sar_film", "spatial_cond", "decoder_skip"],
    "v6_only": ["early_fusion", "opt_cond", "temporal_bias"],
    "encoder": ["sar_film", "opt_cond"],
    "fusion": ["spatial_cond", "early_fusion"],
    "decoder": ["decoder_skip", "temporal_bias"],
}

CROP_CLASSES = {
    0: "Background", 1: "Winter Wheat", 2: "Summer Corn",
    3: "Rice", 4: "Soybean", 5: "Cotton", 6: "Other",
}


def compute_metrics(preds, labels, num_classes=7):
    """Compute OA, mIoU, Kappa, per-class IoU from numpy or torch arrays."""
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    valid = (labels > 0) & (labels < 255)
    p = preds[valid]
    t = labels[valid]
    if len(p) == 0:
        return {"OA": 0.0, "mIoU": 0.0, "Kappa": 0.0, "IoU_per_class": []}

    oa = float((p == t).astype(np.float64).mean())
    iou_list = []
    for cls in range(1, num_classes):
        tp = ((p == cls) & (t == cls)).sum()
        fp = ((p == cls) & (t != cls)).sum()
        fn = ((p != cls) & (t == cls)).sum()
        iou_list.append(float(tp / (tp + fp + fn + 1e-6)))

    n = len(p)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for ti, pi in zip(t, p):
        if 0 <= ti < num_classes and 0 <= pi < num_classes:
            cm[ti, pi] += 1
    po = cm.diagonal().sum() / (n + 1e-6)
    pe = (cm.sum(0) * cm.sum(1)).sum() / (n ** 2 + 1e-6)
    kappa = (po - pe) / (1 - pe + 1e-6)

    return {
        "OA": float(oa), "mIoU": float(np.mean(iou_list)) if iou_list else 0.0,
        "Kappa": float(kappa), "IoU_per_class": iou_list,
    }


class DEMAblationRunner:
    """Runs systematic DEM ablation experiments on a FusionCropNet model.

    Tests each injection point individually and in groups, measuring
    the impact on OA, mIoU, and per-class IoU.

    Usage:
        runner = DEMAblationRunner(model, device="cuda")
        results = runner.run_individual(opt, sar, dem, doy, labels)
        results = runner.run_grouped(opt, sar, dem, doy, labels)
        report = runner.generate_report(results)
    """

    def __init__(self, model, device="cuda", num_classes=7):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_classes = num_classes

    def _run_single_config(self, opt_seq, sar_seq, dem, doy, labels,
                           dem_ablation, cloud_mask=None, valid_count=None):
        """Run inference with a specific DEM ablation configuration."""
        self.model.eval()
        with torch.no_grad():
            alpha = self.model(
                opt_seq, sar_seq, dem, doy,
                cloud_mask=cloud_mask, valid_count=valid_count,
                dem_ablation=dem_ablation,
            )
            if isinstance(alpha, tuple):
                alpha = alpha[0]
        probs = (alpha / alpha.sum(dim=1, keepdim=True)).cpu().numpy()
        preds = probs.argmax(axis=1)
        if labels.ndim == 3:
            labels_np = labels
        else:
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

        metrics = compute_metrics(preds[0] if preds.ndim == 3 else preds,
                                  labels_np[0] if labels_np.ndim == 3 else labels_np,
                                  self.num_classes)
        metrics["_preds"] = preds
        metrics["_probs"] = probs
        return metrics

    def run_baseline(self, opt_seq, sar_seq, dem, doy, labels,
                     cloud_mask=None, valid_count=None):
        """Run baseline inference with all DEM injections enabled."""
        return self._run_single_config(
            opt_seq, sar_seq, dem, doy, labels,
            dem_ablation={}, cloud_mask=cloud_mask, valid_count=valid_count)

    def run_individual(self, opt_seq, sar_seq, dem, doy, labels,
                       cloud_mask=None, valid_count=None):
        """Run individual ablation: disable each injection point one at a time.

        Returns dict with baseline + per-point results and deltas.
        """
        results = {"_type": "individual", "_num_classes": self.num_classes}

        t0 = time.time()
        print("  [baseline] all DEM injections enabled...")
        baseline = self.run_baseline(opt_seq, sar_seq, dem, doy, labels,
                                     cloud_mask, valid_count)
        results["baseline"] = {k: v for k, v in baseline.items()
                               if not k.startswith("_")}
        print(f"    OA={baseline['OA']:.4f}  mIoU={baseline['mIoU']:.4f}  "
              f"({time.time() - t0:.1f}s)")

        for key, info in INJECTION_POINTS.items():
            t1 = time.time()
            cfg = {k: k != key for k in INJECTION_POINTS}
            desc = f"[no {key}] {info['name']}"
            print(f"  {desc}...")
            result = self._run_single_config(
                opt_seq, sar_seq, dem, doy, labels,
                dem_ablation=cfg, cloud_mask=cloud_mask, valid_count=valid_count)
            stored = {k: v for k, v in result.items() if not k.startswith("_")}
            stored["name"] = info["name"]
            stored["group"] = info["group"]
            delta_miou = baseline["mIoU"] - result["mIoU"]
            delta_oa = baseline["OA"] - result["OA"]
            stored["delta_mIoU"] = float(delta_miou)
            stored["delta_OA"] = float(delta_oa)
            stored["delta_rel_mIoU"] = float(delta_miou / max(baseline["mIoU"], 1e-6))
            results[key] = stored
            print(f"    OA={result['OA']:.4f}  mIoU={result['mIoU']:.4f}  "
                  f"ΔmIoU={-delta_miou:+.4f}  ({time.time() - t1:.1f}s)")

        return results

    def run_grouped(self, opt_seq, sar_seq, dem, doy, labels,
                    cloud_mask=None, valid_count=None):
        """Run grouped ablation: disable groups of injection points.

        Tests: v5_shared only, v6_only only, encoder group, fusion group,
        decoder group, and all disabled.
        """
        results = {"_type": "grouped", "_num_classes": self.num_classes}

        t0 = time.time()
        baseline = self.run_baseline(opt_seq, sar_seq, dem, doy, labels,
                                     cloud_mask, valid_count)
        results["baseline"] = {k: v for k, v in baseline.items()
                               if not k.startswith("_")}

        group_configs = {
            "no_v5_shared": {
                "group_name": "V5 Shared DEM disabled",
                "disable": ABLATION_GROUPS["v5_shared"],
                "description": "Disable all V5EDL-shared DEM paths (SAR FiLM + Spatial Cond + Decoder Skip)",
            },
            "no_v6_only": {
                "group_name": "V6 DEM enhancements disabled",
                "disable": ABLATION_GROUPS["v6_only"],
                "description": "Disable all V6-only DEM paths (Early Fusion + Opt Cond + Temporal Bias)",
            },
            "no_encoder_dem": {
                "group_name": "Encoder DEM disabled",
                "disable": ABLATION_GROUPS["encoder"],
                "description": "Disable DEM in encoder paths (SAR FiLM + Opt Cond)",
            },
            "no_fusion_dem": {
                "group_name": "Fusion DEM disabled",
                "disable": ABLATION_GROUPS["fusion"],
                "description": "Disable DEM in fusion paths (Spatial Cond + Early Fusion)",
            },
            "no_decoder_dem": {
                "group_name": "Decoder DEM disabled",
                "disable": ABLATION_GROUPS["decoder"],
                "description": "Disable DEM in decoder paths (Decoder Skip + Temporal Bias)",
            },
            "no_dem_all": {
                "group_name": "ALL DEM disabled",
                "disable": list(INJECTION_POINTS.keys()),
                "description": "Completely remove DEM from the model",
            },
        }

        for key, ginfo in group_configs.items():
            t1 = time.time()
            cfg = {k: k not in ginfo["disable"] for k in INJECTION_POINTS}
            print(f"  [{key}] {ginfo['group_name']}...")
            result = self._run_single_config(
                opt_seq, sar_seq, dem, doy, labels,
                dem_ablation=cfg, cloud_mask=cloud_mask, valid_count=valid_count)
            stored = {k: v for k, v in result.items() if not k.startswith("_")}
            stored["name"] = ginfo["group_name"]
            stored["description"] = ginfo["description"]
            stored["disabled_points"] = ginfo["disable"]
            delta_miou = baseline["mIoU"] - result["mIoU"]
            delta_oa = baseline["OA"] - result["OA"]
            stored["delta_mIoU"] = float(delta_miou)
            stored["delta_OA"] = float(delta_oa)
            stored["delta_rel_mIoU"] = float(delta_miou / max(baseline["mIoU"], 1e-6))
            results[key] = stored
            print(f"    OA={result['OA']:.4f}  mIoU={result['mIoU']:.4f}  "
                  f"ΔmIoU={-delta_miou:+.4f}  ({time.time() - t1:.1f}s)")

        return results

    def run_full(self, opt_seq, sar_seq, dem, doy, labels,
                 cloud_mask=None, valid_count=None):
        """Run both individual and grouped ablation in one pass.

        Shares the baseline run between both analyses.
        """
        print("=" * 60)
        print("  DEM Ablation Experiment Suite")
        print("=" * 60)

        print("\n--- Phase 1: Individual Injection Point Ablation ---")
        individual = self.run_individual(opt_seq, sar_seq, dem, doy, labels,
                                         cloud_mask, valid_count)

        print("\n--- Phase 2: Grouped Ablation ---")
        grouped = self.run_grouped(opt_seq, sar_seq, dem, doy, labels,
                                   cloud_mask, valid_count)

        return {
            "individual": individual,
            "grouped": grouped,
            "injection_points": INJECTION_POINTS,
        }

    def generate_report(self, results, output_dir="./ablation_output"):
        """Generate a comprehensive ablation report with rankings."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = []

        def _h(s, level=1):
            report.append(f"{'#' * level} {s}\n")

        _h("DEM Ablation Report", 1)
        _h(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", 2)

        # ── Individual ablation ──
        indiv = results.get("individual", {})
        if indiv:
            _h("Individual Injection Point Ablation", 2)
            bl = indiv.get("baseline", {})
            report.append(f"**Baseline (all DEM enabled):** "
                          f"OA={bl.get('OA', 0):.4f}, "
                          f"mIoU={bl.get('mIoU', 0):.4f}, "
                          f"Kappa={bl.get('Kappa', 0):.4f}\n")

            report.append("| Rank | Injection Point | Group | OA | mIoU | ΔmIoU | ΔRel% |")
            report.append("|------|----------------|-------|----|------|-------|-------|")

            points = [(k, v) for k, v in indiv.items()
                      if k not in ("_type", "_num_classes", "baseline")]
            points.sort(key=lambda x: x[1].get("delta_mIoU", 0), reverse=True)

            for rank, (key, info) in enumerate(points, 1):
                report.append(
                    f"| {rank} | {info.get('name', key)} | {info.get('group', '')} "
                    f"| {info.get('OA', 0):.4f} | {info.get('mIoU', 0):.4f} "
                    f"| {info.get('delta_mIoU', 0):+.4f} "
                    f"| {info.get('delta_rel_mIoU', 0)*100:+.2f}% |")

            report.append("")

            # Per-class breakdown for top-3 most impactful
            report.append("### Per-Class IoU Impact (top-3 most impactful points)\n")
            top3 = points[:3]
            report.append("| Class | Baseline | " +
                          " | ".join(info.get('name', k)[:12] for k, info in top3) + " |")
            report.append("|-------|----------|" +
                          "|".join("------" for _ in top3) + "|")

            bl_iou = bl.get("IoU_per_class", [])
            for ci, cname in CROP_CLASSES.items():
                if ci == 0:
                    continue
                bl_val = bl_iou[ci - 1] if ci - 1 < len(bl_iou) else 0
                row = f"| {cname} | {bl_val:.4f} |"
                for _, info in top3:
                    iou_list = info.get("IoU_per_class", [])
                    val = iou_list[ci - 1] if ci - 1 < len(iou_list) else 0
                    row += f" {val:.4f} |"
                report.append(row)
            report.append("")

        # ── Grouped ablation ──
        grouped = results.get("grouped", {})
        if grouped:
            _h("Grouped Ablation", 2)
            bl = grouped.get("baseline", {})
            report.append(f"**Baseline:** OA={bl.get('OA', 0):.4f}, "
                          f"mIoU={bl.get('mIoU', 0):.4f}\n")

            report.append("| Configuration | OA | mIoU | ΔmIoU | ΔRel% |")
            report.append("|---------------|----|------|-------|-------|")

            groups = [(k, v) for k, v in grouped.items()
                      if k not in ("_type", "_num_classes", "baseline")]
            groups.sort(key=lambda x: x[1].get("delta_mIoU", 0), reverse=True)

            for key, info in groups:
                report.append(
                    f"| {info.get('name', key)} | {info.get('OA', 0):.4f} "
                    f"| {info.get('mIoU', 0):.4f} "
                    f"| {info.get('delta_mIoU', 0):+.4f} "
                    f"| {info.get('delta_rel_mIoU', 0)*100:+.2f}% |")

            report.append("")

        # ── Summary ──
        _h("Summary & Recommendations", 2)

        if indiv:
            points_all = [(k, v) for k, v in indiv.items()
                          if k not in ("_type", "_num_classes", "baseline")]
            points_all.sort(key=lambda x: x[1].get("delta_mIoU", 0), reverse=True)

            report.append("**Injection points ranked by impact (ΔmIoU):**\n")
            for rank, (key, info) in enumerate(points_all, 1):
                report.append(
                    f"{rank}. **{info.get('name', key)}** "
                    f"(ΔmIoU={info.get('delta_mIoU', 0):+.4f}, "
                    f"ΔRel={info.get('delta_rel_mIoU', 0)*100:+.2f}%)")

            # Identify low-contribution points
            low_impact = [(k, v) for k, v in points_all
                          if abs(v.get("delta_mIoU", 0)) < 0.005]
            if low_impact:
                report.append(
                    f"\n**Potentially removable:** {len(low_impact)} injection point(s) "
                    f"with ΔmIoU < 0.005: " +
                    ", ".join(f"*{v.get('name', k)}*" for k, v in low_impact))

        report_text = "\n".join(report)

        # Save report
        report_path = output_dir / "dem_ablation_report.md"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

        # Save JSON
        json_path = output_dir / "dem_ablation_results.json"
        serializable = {}
        for section in ["individual", "grouped"]:
            if section in results:
                serializable[section] = {}
                for k, v in results[section].items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, dict):
                        serializable[section][k] = {
                            kk: vv for kk, vv in v.items()
                            if not kk.startswith("_")
                        }
        json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"JSON results saved to: {json_path}")

        return report_text


def print_ablation_summary(results):
    """Print a concise ablation summary to the console."""
    indiv = results.get("individual", {})
    bl = indiv.get("baseline", {})

    print("\n" + "=" * 70)
    print("  DEM Ablation Summary")
    print("=" * 70)
    print(f"  Baseline (all DEM):  OA={bl.get('OA', 0):.4f}  "
          f"mIoU={bl.get('mIoU', 0):.4f}  Kappa={bl.get('Kappa', 0):.4f}")
    print("-" * 70)
    print(f"  {'Injection Point':<24s} {'OA':>7s} {'mIoU':>7s} {'ΔmIoU':>8s} {'ΔRel%':>8s}")
    print("-" * 70)

    points = [(k, v) for k, v in indiv.items()
              if k not in ("_type", "_num_classes", "baseline")]
    points.sort(key=lambda x: x[1].get("delta_mIoU", 0), reverse=True)

    for key, info in points:
        print(f"  {info.get('name', key):<24s} "
              f"{info.get('OA', 0):7.4f} {info.get('mIoU', 0):7.4f} "
              f"{info.get('delta_mIoU', 0):+8.4f} {info.get('delta_rel_mIoU', 0)*100:+8.2f}%")

    print("=" * 70)
    if points:
        most = points[0]
        least = points[-1]
        print(f"  Most impactful:   {most[1].get('name', most[0])} "
              f"(ΔmIoU={most[1].get('delta_mIoU', 0):+.4f})")
        print(f"  Least impactful:  {least[1].get('name', least[0])} "
              f"(ΔmIoU={least[1].get('delta_mIoU', 0):+.4f})")
    print("=" * 70 + "\n")
