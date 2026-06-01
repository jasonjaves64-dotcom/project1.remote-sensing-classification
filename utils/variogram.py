# =============================================================================
# utils/variogram.py
# Spatial quality assessment: Variogram Quality Score (VQS) and Moran's I
# =============================================================================
"""
Variogram-based spatial prediction quality metrics.

VQS (Variogram Quality Score) quantifies how well prediction residuals
conform to the expected spatial structure.  A score near 1 indicates
that residuals are spatially uncorrelated (ideal); a low score flags
remaining spatial autocorrelation that the model failed to capture.

Core functions
--------------
compute_vqs(errors, coords) -> dict
morans_i(errors, weights_matrix) -> float
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Optional scipy import — fall back to manual least-squares fitting
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# =============================================================================
# Exponential variogram model
# =============================================================================
def _exponential_model(
    h: np.ndarray, nugget: float, sill: float, range_param: float
) -> np.ndarray:
    """Exponential variogram: gamma(h) = nugget + sill * (1 - exp(-h / range))."""
    return nugget + sill * (1.0 - np.exp(-h / range_param))


# =============================================================================
# Experimental variogram
# =============================================================================
def _experimental_variogram(
    errors: np.ndarray, coords: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Bin pairwise squared-differences by distance lag.

    Parameters
    ----------
    errors : (N,) float
        Prediction residuals (pred - true) or (true - pred).
    coords : (N, 2) float
        Spatial coordinates in projected CRS (metres recommended).
    n_bins : int
        Number of distance bins.

    Returns
    -------
    h_bins : (n_bins,) float — bin-centre distances
    gamma_exp : (n_bins,) float — empirical semivariance per bin
    """
    n = len(errors)
    # Pairwise distance matrix (upper triangle only to save memory)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Extract upper-triangle (i < j) indices
    i_upper, j_upper = np.triu_indices(n, k=1)
    distances = dist[i_upper, j_upper]

    # Semivariance for each pair
    semivar = 0.5 * (errors[i_upper] - errors[j_upper]) ** 2

    # Bin edges (uniform spacing from 0 to max distance)
    max_dist = distances.max() if len(distances) > 0 else 1.0
    if max_dist <= 1e-8:
        h_bins = np.linspace(0, 1, n_bins + 1)[1:]
        return h_bins, np.zeros(n_bins)

    edges = np.linspace(0, max_dist, n_bins + 1)
    h_bins = 0.5 * (edges[:-1] + edges[1:])  # bin centres
    gamma_exp = np.zeros(n_bins)

    for k in range(n_bins):
        mask = (distances >= edges[k]) & (distances < edges[k + 1])
        if mask.sum() > 0:
            gamma_exp[k] = semivar[mask].mean()

    return h_bins, gamma_exp


# =============================================================================
# Variogram fitting
# =============================================================================
def _fit_variogram(
    h_bins: np.ndarray, gamma_exp: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit exponential variogram model to experimental semivariance.

    Returns
    -------
    popt : (3,) ndarray — [nugget, sill, range]
    r2 : float — coefficient of determination of the fit
    """
    # Initial guesses
    p0 = np.array([0.0, gamma_exp.max() if gamma_exp.max() > 0 else 1.0,
                   h_bins.max() * 0.33 if h_bins.max() > 0 else 1.0])

    if _HAS_SCIPY:
        try:
            popt, _ = curve_fit(
                _exponential_model, h_bins, gamma_exp,
                p0=p0, maxfev=10000,
                bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]),
            )
        except (RuntimeError, ValueError):
            popt = p0
    else:
        # Manual Levenberg-Marquardt-style least squares via grid + refinement
        popt = _manual_fit(h_bins, gamma_exp, p0)

    # Goodness-of-fit
    pred = _exponential_model(h_bins, *popt)
    ss_res = np.sum((gamma_exp - pred) ** 2)
    ss_tot = np.sum((gamma_exp - gamma_exp.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return popt, float(r2)


def _manual_fit(
    h: np.ndarray, y: np.ndarray, p0: np.ndarray
) -> np.ndarray:
    """Grid-search + local-refinement least-squares fit (scipy-free fallback)."""
    best_params = p0.copy()
    best_loss = np.inf

    # Coarse grid around initial guess
    nugget_candidates = np.linspace(0, p0[1] * 0.3, 8)
    sill_candidates = np.linspace(p0[1] * 0.5, p0[1] * 1.5, 8)
    range_candidates = np.linspace(p0[2] * 0.1, p0[2] * 2.0, 8)

    for n0 in nugget_candidates:
        for s0 in sill_candidates:
            for r0 in range_candidates:
                if r0 <= 0 or s0 < 0 or n0 < 0:
                    continue
                pred = _exponential_model(h, n0, s0, r0)
                loss = np.sum((y - pred) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    best_params = np.array([n0, s0, r0])

    # Local refinement via small random perturbations
    rng = np.random.default_rng(42)
    for _ in range(200):
        noise = (rng.random(3) - 0.5) * 0.1 * np.abs(best_params)
        candidate = best_params + noise
        candidate = np.maximum(candidate, [0, 0, 1e-6])
        pred = _exponential_model(h, *candidate)
        loss = np.sum((y - pred) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_params = candidate

    return best_params


# =============================================================================
# Spatial weights matrix helpers
# =============================================================================
def _build_weights_matrix(
    coords: np.ndarray,
    threshold: float | None = None,
    k_nearest: int = 8,
) -> np.ndarray:
    """Build spatial weights matrix from coordinates.

    Uses inverse-distance weighting with a distance threshold or
    k-nearest-neighbour connectivity.

    Parameters
    ----------
    coords : (N, 2) float
    threshold : float or None
        Distance cutoff.  If None, auto-computed as the median of the
        k_nearest-th nearest-neighbour distances.
    k_nearest : int
        Used only when threshold is None.

    Returns
    -------
    W : (N, N) float — row-standardised weights matrix
    """
    n = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    if threshold is None:
        # Use median of k-nearest-neighbour distances as threshold
        sorted_dist = np.sort(dist, axis=1)
        threshold = float(np.median(sorted_dist[:, min(k_nearest, n - 1)]))

    W = np.where((dist > 0) & (dist <= threshold), 1.0 / (dist + 1e-8), 0.0)

    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums

    return W


# =============================================================================
# Moran's I
# =============================================================================
def morans_i(errors: np.ndarray, weights_matrix: np.ndarray) -> float:
    """Moran's I spatial autocorrelation index.

    Parameters
    ----------
    errors : (N,) float — prediction residuals.
    weights_matrix : (N, N) float — row-standardised spatial weights.

    Returns
    -------
    I : float — Moran's I in [-1, 1].  Values near 0 indicate spatial
        randomness; positive values indicate clustering of similar errors;
        negative values indicate dispersion.
    """
    n = len(errors)
    z = errors - errors.mean()
    W = weights_matrix

    # Numerator: weighted covariance
    num = np.sum(W * np.outer(z, z))

    # Denominator: total variance
    denom = np.sum(z ** 2)

    # Sum of weights
    W_sum = W.sum()

    if denom < 1e-12 or W_sum < 1e-12:
        return 0.0

    I = (n / W_sum) * (num / denom)
    return float(I)


# =============================================================================
# VQS — Variogram Quality Score
# =============================================================================
def compute_vqs(
    errors: np.ndarray,
    coords: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compute the Variogram Quality Score and related spatial diagnostics.

    Parameters
    ----------
    errors : (N,) float
        Prediction residuals (e.g. ``y_true - y_pred``).
    coords : (N, 2) float
        Spatial coordinates (x, y) in a projected CRS (metres).
    n_bins : int
        Number of distance bins for the experimental variogram.

    Returns
    -------
    dict with keys:
        vqs_score : float
            Variogram Quality Score in [0, 1].  1 = perfect (no spatial
            structure in residuals).
        morans_i : float
            Moran's I of the residuals.
        variogram_params : dict
            ``nugget``, ``sill``, ``range``, ``r2`` — fitted exponential
            model parameters and goodness-of-fit.
        h_bins : ndarray
            Distance bin centres (metres).
        gamma_exp : ndarray
            Experimental semivariance values.
        gamma_fit : ndarray
            Fitted model semivariance at bin centres.
    """
    errors = np.asarray(errors, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if len(errors) != coords.shape[0]:
        raise ValueError(
            f"errors ({len(errors)}) and coords ({coords.shape[0]}) "
            f"must have the same length"
        )

    # --- experimental variogram ---
    h_bins, gamma_exp = _experimental_variogram(errors, coords, n_bins=n_bins)

    # --- fit exponential model ---
    popt, r2 = _fit_variogram(h_bins, gamma_exp)
    nugget, sill, range_param = float(popt[0]), float(popt[1]), float(popt[2])
    gamma_fit = _exponential_model(h_bins, nugget, sill, range_param)

    # --- VQS score ---
    gamma_range = gamma_exp.max() - gamma_exp.min()
    if gamma_range < 1e-8:
        vqs_score = 1.0
    else:
        rmse = np.sqrt(np.mean((gamma_exp - gamma_fit) ** 2))
        nrmse = rmse / gamma_range
        vqs_score = max(0.0, min(1.0, 1.0 - float(nrmse)))

    # --- Moran's I ---
    W = _build_weights_matrix(coords)
    mi = morans_i(errors, W)

    return {
        "vqs_score": float(vqs_score),
        "morans_i": mi,
        "variogram_params": {
            "nugget": nugget,
            "sill": sill,
            "range": range_param,
            "r2": r2,
        },
        "h_bins": h_bins,
        "gamma_exp": gamma_exp,
        "gamma_fit": gamma_fit,
    }


# =============================================================================
# Quick-look test (run with: python utils/variogram.py)
# =============================================================================
if __name__ == "__main__":
    # Generate synthetic residuals with known spatial structure
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(0, 500, n)
    y = rng.uniform(0, 500, n)
    coords = np.column_stack([x, y])

    # Spatially correlated errors: smooth trend + local noise (range ~80 m)
    errors = (0.5 * np.sin(x / 80.0) * np.cos(y / 80.0)
              + rng.normal(0, 0.15, n))

    result = compute_vqs(errors, coords, n_bins=12)

    print("=== VQS Quick Test ===")
    print(f"  VQS score : {result['vqs_score']:.4f}")
    print(f"  Moran's I : {result['morans_i']:.4f}")
    print(f"  Nugget    : {result['variogram_params']['nugget']:.4f}")
    print(f"  Sill      : {result['variogram_params']['sill']:.4f}")
    print(f"  Range     : {result['variogram_params']['range']:.1f} m")
    print(f"  Fit R^2   : {result['variogram_params']['r2']:.4f}")

    # Sanity: pure white noise should give high VQS
    wn = rng.normal(0, 1, n)
    result_wn = compute_vqs(wn, coords, n_bins=12)
    print(f"\n  White-noise VQS : {result_wn['vqs_score']:.4f}")
    print(f"  White-noise MI  : {result_wn['morans_i']:.4f}")
