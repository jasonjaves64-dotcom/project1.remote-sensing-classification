"""Effective sample size estimation for temporal sequences.

Implements Module 4 (Statistical Learning Theory) utilities:
  - Autocorrelation-based T_eff estimation
  - Closed-form hyperparameter recommendations (M*, r*, lambda*)

Reference: project1-数学理论基础-四大模块.md, Section 4.2
"""
import math
import torch


def estimate_effective_sample_size(
    x: torch.Tensor, max_lag: int = None
) -> float:
    """Estimate effective sample size accounting for temporal autocorrelation.

    Computes T_eff = T / (1 + 2 * sum_{k=1}^T beta(k)) where beta(k) is the
    lag-k autocorrelation of the time series.

    Under geometric mixing (beta(k) <= C * exp(-gamma * k)), T_eff reflects
    the number of *independent* temporal observations — directly impacting
    the generalization bound (Theorem 2, Module 4).

    Args:
        x: (N, T, D) or (T,) temporal sequence
        max_lag: optional max autocorrelation lag (default: min(T//4, 24))

    Returns:
        T_eff: effective sample size (>= 1.0, <= T)
    """
    if x.dim() == 3:
        # Average over batch and channel dims for robust autocorrelation
        x_1d = x.mean(dim=(0, 2))  # (T,)
    elif x.dim() == 2:
        x_1d = x.mean(dim=1)  # (T,)
    else:
        x_1d = x

    T = x_1d.shape[0]
    if T <= 1:
        return 1.0

    if max_lag is None:
        max_lag = min(T // 4, 24)
    max_lag = min(max_lag, T - 1)

    # Center the series
    x_centered = x_1d - x_1d.mean()

    # Compute autocorrelation via FFT for all lags
    x_padded = torch.cat([x_centered, torch.zeros(T, device=x.device)])
    fx = torch.fft.rfft(x_padded)
    acf_full = torch.fft.irfft(fx * fx.conj())[:T]
    acf_full = acf_full / (acf_full[0] + 1e-8)  # normalize to rho(0) = 1

    # Sum positive autocorrelations up to max_lag
    sum_beta = 0.0
    for k in range(1, max_lag + 1):
        beta_k = acf_full[k].item()
        if beta_k <= 0:
            break  # geometric decay: once it crosses zero, contribution is negligible
        sum_beta += beta_k

    T_eff = T / (1.0 + 2.0 * sum_beta)
    return max(1.0, min(float(T), T_eff))


def recommend_hyperparams(
    T_eff: float, K: int, C_in: int, C_out: int
) -> dict:
    """Closed-form hyperparameter recommendations from the temporal generalization bound.

    Based on Module 4, Theorem 2: the generalization bound is minimized when:
      M* = round(sqrt(T_eff / (K * C_in * C_out)))
      r* = max(4, C_in / sqrt(T_eff))
      lambda* = 0.1 / E[||x||^2]

    These closed-form values differ from exhaustive search optimum by <0.8%
    (Section 4.2 experimental validation), reducing tuning from ~12h to <3min.

    Args:
        T_eff: effective sample size from estimate_effective_sample_size()
        K: convolution kernel size (e.g., 3 for TemporalLite)
        C_in: input channels
        C_out: output channels

    Returns:
        dict with keys 'M_star' (int), 'r_star' (int), 'lambda_star' (float)
    """
    M_star = max(1, round(math.sqrt(T_eff / max(K * C_in * C_out, 1))))
    r_star = max(4, round(C_in / max(math.sqrt(T_eff), 1.0)))
    # Conservative lambda: typical E[||x||^2] ~= C_in for normalized features
    lambda_star = 0.1 / max(C_in, 1.0)

    return {
        'M_star': M_star,
        'r_star': r_star,
        'lambda_star': lambda_star,
    }


def estimate_mixing_coefficient(x: torch.Tensor) -> float:
    """Estimate the geometric mixing rate gamma from autocorrelation decay.

    Fits beta(k) ≈ C * exp(-gamma * k) to the first few significant lags.

    Args:
        x: (N, T, D) or (T,) temporal sequence

    Returns:
        gamma: estimated mixing rate (higher = faster decorrelation)
    """
    if x.dim() == 3:
        x_1d = x.mean(dim=(0, 2))
    elif x.dim() == 2:
        x_1d = x.mean(dim=1)
    else:
        x_1d = x

    T = x_1d.shape[0]
    if T <= 2:
        return 1.0

    x_centered = x_1d - x_1d.mean()
    x_padded = torch.cat([x_centered, torch.zeros(T, device=x.device)])
    fx = torch.fft.rfft(x_padded)
    acf = torch.fft.irfft(fx * fx.conj())[:T]
    acf = acf / (acf[0] + 1e-8)

    # Fit log(beta(k)) ~ -gamma * k for the first K lags where beta > 0.01
    lags = []
    log_betas = []
    for k in range(1, min(T // 2, 24)):
        beta = acf[k].item()
        if beta > 0.01:
            lags.append(k)
            log_betas.append(math.log(max(beta, 1e-8)))
        else:
            break

    if len(lags) < 2:
        return 0.5  # default for weak temporal structure

    # Simple OLS: gamma = -slope of log(beta) vs lag
    n = len(lags)
    mean_k = sum(lags) / n
    mean_log = sum(log_betas) / n
    num = sum((lags[i] - mean_k) * (log_betas[i] - mean_log) for i in range(n))
    den = sum((lags[i] - mean_k) ** 2 for i in range(n))
    gamma = -num / max(den, 1e-8)
    return max(0.01, min(5.0, gamma))
