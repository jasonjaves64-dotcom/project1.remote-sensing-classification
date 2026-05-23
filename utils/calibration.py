# =============================================================================
# utils/calibration.py
# EDL Uncertainty Calibration Validation Module
#
# Provides:
#   1. Expected Calibration Error (ECE) — binned & continuous variants
#   2. Reliability diagrams
#   3. Confidence histograms & sharpness metrics
#   4. Negative Log-Likelihood (NLL) & Brier Score for Dirichlet outputs
#   5. OOD / misclassification detection via uncertainty thresholding
#   6. Uncertainty-error correlation (Spearman ρ, AUROC, PR-AUC)
#   7. Class-wise calibration breakdown
# =============================================================================
import numpy as np
from collections import defaultdict

def _safe_divide(a, b, eps=1e-10):
    return a / (b + eps)


# ---------------------------------------------------------------------------
# Core calibration metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(confidences, accuracies, n_bins=15):
    """ECE—binned variant (Guo et al., 2017).

    Args:
        confidences: (N,) max predicted probability per sample [0, 1]
        accuracies:  (N,) binary correctness per sample {0, 1}
        n_bins:      number of equal-width bins

    Returns:
        ece: float, weighted average of |conf - acc| per bin
        bin_details: list of dicts with per-bin statistics
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    total = len(confidences)
    bin_details = []
    for b in range(n_bins):
        mask = bin_ids == b
        n_b = mask.sum()
        if n_b == 0:
            bin_details.append({"bin": b, "n": 0, "conf": 0, "acc": 0, "gap": 0})
            continue
        conf_b = confidences[mask].mean()
        acc_b = accuracies[mask].mean()
        gap = abs(conf_b - acc_b)
        ece += (n_b / total) * gap
        bin_details.append({
            "bin": b, "n": int(n_b), "conf": float(conf_b),
            "acc": float(acc_b), "gap": float(gap),
            "range": (float(bins[b]), float(bins[b+1])),
        })
    return float(ece), bin_details


def adaptive_ece(confidences, accuracies, n_bins=15):
    """Adaptive ECE—equal-mass binning (more robust for skewed confidence)."""
    order = np.argsort(confidences)
    conf_s = confidences[order]
    acc_s = accuracies[order]
    N = len(conf_s)
    ece = 0.0
    bin_details = []
    for b in range(n_bins):
        lo = int(round(b * N / n_bins))
        hi = int(round((b + 1) * N / n_bins))
        if hi <= lo:
            bin_details.append({"bin": b, "n": 0, "conf": 0, "acc": 0, "gap": 0})
            continue
        n_b = hi - lo
        conf_b = conf_s[lo:hi].mean()
        acc_b = acc_s[lo:hi].mean()
        gap = abs(conf_b - acc_b)
        ece += (n_b / N) * gap
        bin_details.append({
            "bin": b, "n": int(n_b), "conf": float(conf_b),
            "acc": float(acc_b), "gap": float(gap),
            "range": (float(conf_s[lo]), float(conf_s[hi-1])),
        })
    return float(ece), bin_details


def maximum_calibration_error(confidences, accuracies, n_bins=15):
    """MCE—maximum gap across bins (worst-case calibration error)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    mce = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        gap = abs(confidences[mask].mean() - accuracies[mask].mean())
        mce = max(mce, gap)
    return float(mce)


def sharpness(confidences):
    """Sharpness: Var(conf), higher = sharper (more concentrated) predictions."""
    return float(np.var(confidences))


def dispersion(alpha, targets, num_classes, ignore_index=255):
    """Dispersion: mean of 1/(1 + sum(alpha)), measures evidence strength.

    Low dispersion → high evidence → confident. High → uncertain.
    """
    valid = targets != ignore_index
    alpha_v = alpha[valid]
    S = alpha_v.sum(axis=-1)
    return float(np.mean(1.0 / (1.0 + S)))


def negative_log_likelihood_dirichlet(alpha, targets, num_classes, ignore_index=255):
    """NLL under Dirichlet: -E_q[log p(y|x)].

    Uses: -log(alpha_y/S) + KL(post||prior) decomposed form.
    """
    valid = targets != ignore_index
    alpha_v = alpha[valid]
    tgt_v = targets[valid]
    S = alpha_v.sum(axis=-1)
    log_probs = np.log(alpha_v / S[:, None] + 1e-10)
    nll = -log_probs[np.arange(len(tgt_v)), tgt_v].mean()
    return float(nll)


def brier_score_dirichlet(alpha, targets, num_classes, ignore_index=255):
    """Brier score adapted for Dirichlet: E[(p - one_hot)^2].

    p_k = alpha_k / S, S = sum(alpha)
    """
    valid = targets != ignore_index
    alpha_v = alpha[valid]
    tgt_v = targets[valid]
    S = alpha_v.sum(axis=-1, keepdims=True)
    probs = alpha_v / S
    one_hot = np.eye(num_classes)[tgt_v]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return float(brier)


# ---------------------------------------------------------------------------
# Uncertainty-error correlation
# ---------------------------------------------------------------------------
def uncertainty_error_correlation(uncertainties, correctness, metric="spearman"):
    """Correlation between uncertainty and prediction error.

    Args:
        uncertainties: (N,) e.g. vacuity or dissonance per pixel
        correctness:   (N,) 0=error, 1=correct
        metric:        "spearman" or "pearson"

    Returns:
        r: correlation coefficient (higher = uncertainty tracks errors better)
        p_value: two-sided p-value
    """
    from scipy import stats
    if metric == "spearman":
        r, p = stats.spearmanr(uncertainties, 1.0 - correctness)
    else:
        r, p = stats.pearsonr(uncertainties, correctness)
    return float(r), float(p)


def uncertainty_auroc(uncertainties, correctness):
    """AUROC for error detection: treat uncertainty as detection score,
    correctness as label (1=correct, 0=error).
    High AUROC → uncertainty discriminates errors from correct predictions.
    """
    from sklearn.metrics import roc_auc_score
    if len(np.unique(correctness)) < 2:
        return 0.5
    score = roc_auc_score(correctness, -uncertainties)
    return float(score)


def uncertainty_pr_auc(uncertainties, correctness):
    """Precision-Recall AUC for error detection."""
    from sklearn.metrics import average_precision_score
    if len(np.unique(correctness)) < 2:
        return 1.0 / max(len(correctness), 1)
    score = average_precision_score(1 - correctness, uncertainties)
    return float(score)


# ---------------------------------------------------------------------------
# OOD / misclassification detection
# ---------------------------------------------------------------------------
def ood_detection_metrics(uncertainties, correctness, percentile=90):
    """Evaluate uncertainty as OOD / misclassification detector.

    Strategy: flag top-(100-percentile)% most uncertain predictions as "errors".
    Compute precision, recall, F1 of this flag against real errors.

    Args:
        uncertainties: (N,)
        correctness:   (N,) 0=error, 1=correct
        percentile:    threshold percentile for flagging

    Returns:
        dict with precision, recall, f1, flag_rate
    """
    threshold = np.percentile(uncertainties, percentile)
    flagged = uncertainties >= threshold
    errors = 1 - correctness

    tp = (flagged & errors.astype(bool)).sum()
    fp = (flagged & ~errors.astype(bool)).sum()
    fn = (~flagged & errors.astype(bool)).sum()

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flag_rate": float(flagged.mean()),
        "threshold": float(threshold),
    }


def uncertainty_rejection_curve(uncertainties, correctness, n_points=20):
    """Build an accuracy-rejection curve (oracle ordering).

    Pixels are sorted by uncertainty; accuracy is computed on the
    most-confident subset at each retention ratio.
    """
    order = np.argsort(uncertainties)  # ascending
    correct_sorted = correctness[order]
    N = len(correct_sorted)
    ratios = np.linspace(0.1, 1.0, n_points)
    curve = []
    for r in ratios:
        k = int(N * r)
        if k == 0:
            continue
        acc = correct_sorted[:k].mean()
        curve.append({"retention": float(r), "accuracy": float(acc)})
    return curve


# ---------------------------------------------------------------------------
# Full calibration report
# ---------------------------------------------------------------------------
def calibration_report(alpha, targets, num_classes, class_names=None,
                       ignore_index=255, n_bins=15):
    """Generate comprehensive calibration report for EDL outputs.

    Args:
        alpha:      (N, K) or (B, K, H, W) Dirichlet evidence
        targets:    (N,) or (B, H, W) ground truth labels
        num_classes: int
        class_names: optional list of class name strings
        ignore_index: label value to ignore
        n_bins:     bins for ECE

    Returns:
        dict with all calibration metrics, per-class breakdown, and raw arrays
    """
    # Flatten
    if alpha.ndim == 4:
        B, K, H, W = alpha.shape
        alpha = alpha.transpose(0, 2, 3, 1).reshape(-1, K)
    if targets.ndim == 3:
        targets = targets.reshape(-1)
    elif targets.ndim == 2:
        targets = targets.reshape(-1)

    valid = targets != ignore_index
    alpha_v = alpha[valid]
    tgt_v = targets[valid]

    K = num_classes
    S = alpha_v.sum(axis=-1, keepdims=True)
    probs = alpha_v / S
    conf = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == tgt_v).astype(np.float32)

    vacuity = K / S.squeeze(-1)
    dissonance = 1.0 - (probs * probs).sum(axis=-1)

    # ECE variants
    ece, bin_details = expected_calibration_error(conf, correct, n_bins)
    adap_ece, adap_bins = adaptive_ece(conf, correct, n_bins)
    mce = maximum_calibration_error(conf, correct, n_bins)

    # Sharpness & dispersion
    sharp = sharpness(conf)
    disp = dispersion(alpha_v, tgt_v, K, ignore_index=0)

    # Scoring rules
    nll = negative_log_likelihood_dirichlet(alpha_v, tgt_v, K, ignore_index=0)
    brier = brier_score_dirichlet(alpha_v, tgt_v, K, ignore_index=0)

    # Accuracy
    oa = float(correct.mean())

    # Uncertainty-error correlation
    spear_r, spear_p = uncertainty_error_correlation(vacuity, correct, "spearman")
    diss_r, diss_p = uncertainty_error_correlation(dissonance, correct, "spearman")
    auroc = uncertainty_auroc(vacuity, correct)
    prauc = uncertainty_pr_auc(vacuity, correct)

    # OOD detection at multiple thresholds
    ood_metrics = {}
    for pct in [80, 90, 95]:
        ood_metrics[f"p{pct}"] = ood_detection_metrics(vacuity, correct, pct)

    # Rejection curve
    rej_curve = uncertainty_rejection_curve(vacuity, correct)

    # Per-class calibration
    per_class = {}
    if class_names is None:
        class_names = {k: f"Class_{k}" for k in range(K)}
    for k in range(K):
        cls_mask = tgt_v == k
        if cls_mask.sum() < 10:
            per_class[k] = {"name": class_names.get(k, str(k)), "n": int(cls_mask.sum()),
                           "ece": None, "acc": None, "mean_conf": None, "mean_vacuity": None}
            continue
        cls_conf = conf[cls_mask]
        cls_corr = correct[cls_mask]
        cls_ece, _ = expected_calibration_error(cls_conf, cls_corr, min(n_bins, cls_mask.sum()//5))
        per_class[k] = {
            "name": class_names.get(k, str(k)),
            "n": int(cls_mask.sum()),
            "ece": float(cls_ece),
            "acc": float(cls_corr.mean()),
            "mean_conf": float(cls_conf.mean()),
            "mean_vacuity": float(vacuity[cls_mask].mean()),
        }

    return {
        "OA": oa,
        "ECE": ece,
        "AdaptiveECE": adap_ece,
        "MCE": mce,
        "NLL": nll,
        "Brier": brier,
        "Sharpness": sharp,
        "Dispersion": disp,
        "SpearmanR_vacuity": spear_r,
        "SpearmanR_dissonance": diss_r,
        "AUROC_error_detection": auroc,
        "PR_AUC_error_detection": prauc,
        "OOD_detection": ood_metrics,
        "RejectionCurve": rej_curve,
        "ECE_bins": bin_details,
        "PerClass": per_class,
        "_raw": {
            "confidences": conf,
            "correctness": correct,
            "predictions": preds,
            "vacuity": vacuity,
            "dissonance": dissonance,
            "probs": probs,
        },
    }


def print_calibration_report(report):
    """Pretty-print a calibration report."""
    print("=" * 62)
    print("  EDL Uncertainty Calibration Report")
    print("=" * 62)
    print(f"  OA:               {report['OA']:.4f}")
    print(f"  ECE:              {report['ECE']:.4f}")
    print(f"  Adaptive ECE:     {report['AdaptiveECE']:.4f}")
    print(f"  MCE:              {report['MCE']:.4f}")
    print(f"  NLL (Dirichlet):  {report['NLL']:.4f}")
    print(f"  Brier:            {report['Brier']:.4f}")
    print(f"  Sharpness:        {report['Sharpness']:.4f}")
    print(f"  Dispersion:       {report['Dispersion']:.4f}")
    print(f"  Spearman ρ (vac): {report['SpearmanR_vacuity']:.4f}")
    print(f"  AUROC (err det):  {report['AUROC_error_detection']:.4f}")
    print(f"  PR-AUC (err det): {report['PR_AUC_error_detection']:.4f}")
    print("-" * 62)
    print("  Per-class ECE:")
    for k, v in report["PerClass"].items():
        if v["ece"] is not None:
            print(f"    {v['name']:14s}: ECE={v['ece']:.4f}  Acc={v['acc']:.4f}  "
                  f"Conf={v['mean_conf']:.4f}  n={v['n']}")
    print("-" * 62)
    for k, v in report["OOD_detection"].items():
        print(f"  OOD@{k}: P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f}  flag={v['flag_rate']:.3f}")
    print("=" * 62)
