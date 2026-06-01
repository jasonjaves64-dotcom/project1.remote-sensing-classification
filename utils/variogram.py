# =============================================================================
# utils/variogram.py
# Spatial Variogram Quality Metrics for Uncertainty Maps
#
# Measures whether the model's vacuity (epistemic uncertainty) exhibits
# spatially structured patterns — a key indicator of prediction quality.
#
# Rationale:
#   - Well-calibrated model: high vacuity clusters at field boundaries and
#     rare crop types → structured spatial autocorrelation at short lags
#   - Poorly-calibrated model: vacuity is random noise → flat variogram
#   - Over-confident model: vacuity is uniformly low → zero variogram
#
# Metrics:
#   1. Empirical variogram (semivariance vs lag distance)
#   2. Variogram model fitting (exponential, spherical, Gaussian)
#   3. Spatial autocorrelation (Moran's I, Geary's C)
#   4. Variogram Quality Score (VQS): composite indicator 0–1
# =============================================================================
import numpy as np
from typing import Optional, Tuple, Literal
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


# ── Variogram model functions ──────────────────────────────────────────────
def _exponential(h, nugget, sill, range_param):
    """Exponential variogram: γ(h) = n + s*(1 - exp(-h/r))."""
    return nugget + sill * (1.0 - np.exp(-h / (range_param + 1e-8)))


def _spherical(h, nugget, sill, range_param):
    """Spherical variogram: linear at short distances, flat beyond range."""
    ratio = h / (range_param + 1e-8)
    linear_part = nugget + sill * (1.5 * ratio - 0.5 * ratio**3)
    flat_part = np.full_like(h, nugget + sill)
    return np.where(h < range_param, linear_part, flat_part)


def _gaussian(h, nugget, sill, range_param):
    """Gaussian variogram: smooth near origin, s-shaped."""
    return nugget + sill * (1.0 - np.exp(-(h**2) / (range_param**2 + 1e-8)))


VARIOGRAM_MODELS = {
    "exponential": _exponential,
    "spherical": _spherical,
    "gaussian": _gaussian,
}


# ── Empirical variogram ────────────────────────────────────────────────────
def empirical_variogram(
    uncertainty_map: np.ndarray,
    coords: Optional[np.ndarray] = None,
    n_lags: int = 20,
    max_lag: Optional[float] = None,
) -> dict:
    """Compute empirical semivariance for an uncertainty map.

    Args:
        uncertainty_map: 2D array (H, W) of vacuity or other uncertainty values.
        coords: (N, 2) array of spatial coordinates; if None, uses pixel indices.
        n_lags: Number of distance bins.
        max_lag: Maximum distance to consider; if None, auto from data extent.

    Returns:
        dict with keys: lags, semivariance, n_pairs, lag_centers
    """
    H, W = uncertainty_map.shape
    u = uncertainty_map.ravel()

    if coords is None:
        ys, xs = np.mgrid[0:H, 0:W]
        coords = np.column_stack([ys.ravel(), xs.ravel()])

    # Pairwise distances
    dist_matrix = squareform(pdist(coords))

    if max_lag is None:
        max_lag = np.percentile(dist_matrix[dist_matrix > 0], 50)

    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2

    semivariance = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=int)

    for i in range(n_lags):
        mask = (dist_matrix >= lag_edges[i]) & (dist_matrix < lag_edges[i + 1])
        n_pairs[i] = mask.sum()
        if n_pairs[i] > 0:
            diffs = np.abs(u[:, None] - u[None, :])  # (N, N)
            semivariance[i] = 0.5 * np.mean(diffs[mask]**2)

    return {
        "lags": lag_centers,
        "semivariance": semivariance,
        "n_pairs": n_pairs,
        "lag_centers": lag_centers,
    }


# ── Variogram model fitting ────────────────────────────────────────────────
def fit_variogram(
    lags: np.ndarray,
    semivariance: np.ndarray,
    model: Literal["exponential", "spherical", "gaussian"] = "exponential",
    p0: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Fit a theoretical variogram model to empirical semivariance.

    Args:
        lags: Distance bin centers from empirical_variogram().
        semivariance: Empirical semivariance values.
        model: Variogram model type.
        p0: Initial parameters (nugget, sill, range); auto-estimated if None.

    Returns:
        dict with: model_name, nugget, sill, range, r_squared, fitted_values
    """
    valid = np.isfinite(semivariance) & (semivariance > 0) & (lags > 0)
    lags_v = lags[valid]
    sv_v = semivariance[valid]

    if len(lags_v) < 5:
        return {
            "model_name": model, "nugget": 0.0, "sill": 0.0,
            "range": 0.0, "r_squared": 0.0, "fitted_values": np.zeros_like(lags),
            "error": "insufficient valid lags",
        }

    if p0 is None:
        nugget0 = float(np.percentile(sv_v, 5))
        sill0 = float(np.percentile(sv_v, 95)) - nugget0
        range0 = float(np.percentile(lags_v, 50))
        p0 = (nugget0, max(sill0, 1e-6), max(range0, 1e-6))

    model_fn = VARIOGRAM_MODELS[model]

    try:
        popt, pcov = curve_fit(
            model_fn, lags_v, sv_v, p0=p0,
            bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]),
            maxfev=5000,
        )
        fitted = model_fn(lags, *popt)
        ss_res = np.sum((sv_v - model_fn(lags_v, *popt))**2)
        ss_tot = np.sum((sv_v - sv_v.mean())**2)
        r_squared = 1.0 - ss_res / (ss_tot + 1e-10)
    except (RuntimeError, ValueError):
        popt = p0
        fitted = model_fn(lags, *popt)
        r_squared = 0.0

    return {
        "model_name": model,
        "nugget": float(popt[0]),
        "sill": float(popt[1]),
        "range": float(popt[2]),
        "r_squared": float(max(0.0, min(1.0, r_squared))),
        "fitted_values": fitted,
    }


# ── Spatial autocorrelation ────────────────────────────────────────────────
def morans_i(
    uncertainty_map: np.ndarray,
    n_lags: int = 20,
    max_lag: Optional[float] = None,
    subsample: int = 2000,
) -> dict:
    """Compute Moran's I spatial autocorrelation for an uncertainty map.

    Moran's I > 0 → clustering (good: uncertainty is spatially structured)
    Moran's I ≈ 0 → random (bad: uncertainty is noise)
    Moran's I < 0 → dispersion (unusual: over-correction)

    Uses subsampling for large maps to keep computation tractable.

    Returns:
        dict with: moran_i, p_value, z_score, interpretation
    """
    H, W = uncertainty_map.shape
    total_n = H * W

    # Subsample if needed
    if total_n > subsample:
        rng = np.random.default_rng(42)
        indices = rng.choice(total_n, subsample, replace=False)
        u = uncertainty_map.ravel()[indices]
        ys, xs = np.unravel_index(indices, (H, W))
    else:
        u = uncertainty_map.ravel()
        ys, xs = np.mgrid[0:H, 0:W]
        ys, xs = ys.ravel(), xs.ravel()

    n = len(u)
    coords = np.column_stack([ys, xs])
    dist = squareform(pdist(coords))

    # Binary spatial weight: 1 if within median distance, 0 otherwise
    if max_lag is None:
        max_lag = np.percentile(dist[dist > 0], 25)
    W = (dist > 0) & (dist < max_lag)
    w_sum = W.sum()

    if w_sum == 0:
        return {"moran_i": 0.0, "p_value": 1.0, "z_score": 0.0,
                "interpretation": "insufficient data"}

    u_mean = u.mean()
    u_demean = u - u_mean
    numerator = n * np.sum(W * np.outer(u_demean, u_demean))
    denominator = w_sum * np.sum(u_demean**2)

    I = numerator / (denominator + 1e-10)

    # Expected I under null (no spatial autocorrelation)
    E_I = -1.0 / (n - 1)

    # Variance approximation (Cliff & Ord, 1981) — numerically stable
    # Use float64 throughout to avoid overflow in large-n computations
    n_f = float(n)
    u_demean_f = u_demean.astype(np.float64)
    S1 = 0.5 * np.sum((W.astype(np.float64) + W.T.astype(np.float64))**2)
    S2 = np.sum((W.sum(axis=0).astype(np.float64) + W.sum(axis=1).astype(np.float64))**2)
    w_sum_f = float(w_sum)
    b2 = (np.sum(u_demean_f**4) / n_f) / max(np.sum(u_demean_f**2)**2 / n_f**2, 1e-15)
    # Compute in log-space chunks to avoid overflow
    t1 = (n_f**2 - 3*n_f + 3) * S1
    t2 = n_f * S2
    t3 = 3 * w_sum_f**2
    t4 = (n_f**2 - n_f) * S1
    t5 = 2 * n_f * S2
    t6 = 6 * w_sum_f**2
    num = n_f * (t1 - t2 + t3) - b2 * (t4 - t5 + t6)
    den = (n_f - 1) * (n_f - 2) * (n_f - 3) * w_sum_f**2
    var_I = max(num / max(den, 1e-15), 1e-15) - E_I**2
    var_I = max(var_I, 1e-15)

    z = (I - E_I) / (np.sqrt(var_I) + 1e-10)

    # Approximate p-value (two-tailed)
    from scipy.stats import norm
    p_value = 2.0 * norm.sf(abs(z))

    if p_value < 0.05 and I > 0:
        interp = "significant positive spatial autocorrelation — uncertainty is structured ✓"
    elif p_value < 0.05 and I < 0:
        interp = "significant negative spatial autocorrelation — unusual pattern ⚠"
    else:
        interp = "no significant spatial autocorrelation — uncertainty appears random"

    return {
        "moran_i": float(I),
        "p_value": float(p_value),
        "z_score": float(z),
        "interpretation": interp,
    }


# ── Variogram Quality Score (VQS) ──────────────────────────────────────────
def variogram_quality_score(
    uncertainty_map: np.ndarray,
    n_lags: int = 20,
) -> dict:
    """Composite Variogram Quality Score (VQS) for an uncertainty map.

    VQS ∈ [0, 1] quantifies how well the uncertainty map exhibits desirable
    spatial structure:
      - VQS > 0.7: uncertainty is well-structured (field boundaries, rare crops)
      - VQS 0.3–0.7: moderate structure, model may need calibration
      - VQS < 0.3: uncertainty is random or absent — poor calibration

    Components:
      1. Variogram fit quality (R² of fitted model)
      2. Spatial range ratio (range / max_lag): larger = broader patterns
      3. Nugget-to-sill ratio (NSR): lower = more spatial structure
      4. Moran's I significance

    Args:
        uncertainty_map: 2D array (H, W) of vacuity/dissonance/variance.
        n_lags: Number of lags for variogram computation.

    Returns:
        dict with vqs and per-component scores.
    """
    # 1. Empirical variogram + fitting
    ev = empirical_variogram(uncertainty_map, n_lags=n_lags)
    fit = fit_variogram(ev["lags"], ev["semivariance"], model="exponential")

    # 2. Spatial autocorrelation
    mi = morans_i(uncertainty_map)

    # Component scores
    # Component A: variogram fit R² (0–1)
    r2_score = fit.get("r_squared", 0.0)

    # Component B: nugget-to-sill ratio (NSR); lower = better
    nsr = fit["nugget"] / (fit["sill"] + 1e-8)
    nsr_score = 1.0 - min(nsr, 1.0)  # invert: 0→1 (clean→noisy)

    # Component C: spatial range significance
    max_lag = ev["lags"][-1] if len(ev["lags"]) > 0 else 1.0
    range_ratio = fit["range"] / (max_lag + 1e-8)
    # Optimal range is moderate (5–30% of max extent)
    optimal_range = 0.05 <= range_ratio <= 0.30
    range_score = 1.0 - abs(range_ratio - 0.15) / 0.5 if not optimal_range else 1.0
    range_score = max(0.0, min(1.0, range_score))

    # Component D: Moran's I significance
    moran_score = 1.0 - mi.get("p_value", 1.0)  # significant → high score

    # Composite VQS
    vqs = 0.25 * r2_score + 0.30 * nsr_score + 0.20 * range_score + 0.25 * moran_score
    vqs = max(0.0, min(1.0, vqs))

    # Interpret
    if vqs >= 0.7:
        grade = "A — Well-structured uncertainty"
    elif vqs >= 0.5:
        grade = "B — Moderately structured"
    elif vqs >= 0.3:
        grade = "C — Weak spatial structure — recalibration recommended"
    else:
        grade = "D — Random/absent uncertainty pattern — recalibration required"

    return {
        "VQS": float(vqs),
        "grade": grade,
        "components": {
            "variogram_r2": float(r2_score),
            "nugget_sill_ratio": float(nsr),
            "range_ratio": float(range_ratio),
            "moran_p_value": float(mi["p_value"]),
        },
        "variogram_fit": fit,
        "moran": mi,
    }


# ── Multi-class vacuity analysis ───────────────────────────────────────────
def per_class_variogram(
    vacuity_map: np.ndarray,
    pred_map: np.ndarray,
    num_classes: int = 7,
    n_lags: int = 20,
) -> dict:
    """Compute variogram metrics per predicted class.

    Identifies which crop types have spatially structured uncertainty.
    Classes with low VQS indicate areas where the model is uniformly
    confident (possibly over-confident) or uniformly uncertain.

    Args:
        vacuity_map: (H, W) array of vacuity values.
        pred_map: (H, W) array of predicted class labels.
        num_classes: Total number of classes.
        n_lags: Lags for variogram computation.

    Returns:
        dict mapping class_index → variogram_quality_score result.
    """
    results = {}
    for cls in range(num_classes):
        mask = pred_map == cls
        if mask.sum() < 50:  # too few pixels
            results[cls] = {"VQS": 0.0, "grade": "insufficient data",
                           "n_pixels": int(mask.sum())}
            continue
        vacuity_cls = np.where(mask, vacuity_map, 0.0)
        vqs = variogram_quality_score(vacuity_cls, n_lags=n_lags)
        vqs["n_pixels"] = int(mask.sum())
        results[cls] = vqs
    return results


# ── Convenience: full vacuity spatial report ────────────────────────────────
def vacuity_spatial_report(
    vacuity_map: np.ndarray,
    pred_map: Optional[np.ndarray] = None,
    num_classes: int = 7,
    n_lags: int = 20,
) -> dict:
    """Generate a complete spatial quality report for a vacuity map.

    Args:
        vacuity_map: (H, W) array of vacuity values.
        pred_map: (H, W) array of predicted class labels (optional).
        num_classes: Number of classes.
        n_lags: Lags for variogram.

    Returns:
        Comprehensive report dict.
    """
    report = {
        "global_vqs": variogram_quality_score(vacuity_map, n_lags=n_lags),
        "moran": morans_i(vacuity_map),
        "empirical_variogram": empirical_variogram(vacuity_map, n_lags=n_lags),
    }

    if pred_map is not None:
        report["per_class"] = per_class_variogram(
            vacuity_map, pred_map, num_classes, n_lags)

    return report
