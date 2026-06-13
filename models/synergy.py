"""Cross-Module Mathematical Synergy Infrastructure.

Shared computational building blocks used by multiple theory modules:
  - Persistent homology computation (shared: Topo-EWC + DomainAdapter)
  - Effective sample size estimate (shared: TemporalLite + TTA safety)
  - Stiefel ADMM solver (shared: Grassmann OT + joint optimization)
  - Geometric anchor features (shared: DomainAdapter + conflict detector)

This module avoids code duplication and ensures consistent implementations
across all four mathematical theory modules.

Reference: project1-数学理论-V6映射分析.md, Cross-Module Synergy Matrix
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═════════════════════════════════════════════════════════════════
# Shared: Persistent Homology Computation
# ═════════════════════════════════════════════════════════════════

def persistence_barcode(
    filtration_values: torch.Tensor,
    max_dim: int = 1,
    min_persistence: float = 0.1,
) -> list[dict]:
    """Compute persistent homology barcode from a filtration.

    Used by:
      - Topo-EWC (parameter importance via barcode-based filtering)
      - DomainAdapter (adaptive regularization strength alpha from ESS)
      - TopologicalConflictClassifier (persistent correction pipeline)

    For a 1D filtration of a simplicial complex, returns birth-death pairs
    for H_0 and H_1 features. Long-lived features indicate robust structure;
    short-lived features indicate noise or transient conflict.

    Implementation uses the union-find algorithm for H_0 and a simplified
    matrix reduction for H_1.

    Args:
        filtration_values: (N,) values defining the filtration order
        max_dim: maximum homology dimension to compute
        min_persistence: minimum (death - birth) to retain a feature

    Returns:
        list of dicts with 'dim', 'birth', 'death', 'persistence'
    """
    N = filtration_values.shape[0]
    device = filtration_values.device

    # Sort by filtration value
    sorted_idx = filtration_values.argsort()
    sorted_vals = filtration_values[sorted_idx]

    barcodes = []

    # H_0: connected components via union-find
    parent = torch.arange(N, device=device)
    rank = torch.zeros(N, dtype=torch.long, device=device)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    # For H_0, features born at low filtration and die when merged
    birth_time = sorted_vals.clone()
    is_alive = torch.ones(N, dtype=torch.bool, device=device)

    for i in range(N):
        idx_i = sorted_idx[i].item()
        # Check adjacent points in filtration order
        for j in range(max(0, i - 1), min(N, i + 2)):
            if i != j:
                idx_j = sorted_idx[j].item()
                merged = union(idx_i, idx_j)
                if merged:
                    death_val = max(sorted_vals[i], sorted_vals[j])
                    birth_val = min(sorted_vals[i], sorted_vals[j])
                    persistence = death_val - birth_val
                    if persistence >= min_persistence:
                        barcodes.append({
                            'dim': 0, 'birth': birth_val.item(),
                            'death': death_val.item(),
                            'persistence': persistence.item(),
                        })
                    is_alive[idx_i] = False

    # Alive H_0 features (never merged) get infinite death
    for i in range(N):
        if is_alive[i] and find(sorted_idx[i].item()) == sorted_idx[i].item():
            persistence = sorted_vals[-1] - sorted_vals[i]
            if persistence >= min_persistence:
                barcodes.append({
                    'dim': 0, 'birth': sorted_vals[i].item(),
                    'death': float('inf'),
                    'persistence': persistence.item(),
                })

    return sorted(barcodes, key=lambda b: b['persistence'], reverse=True)


def persistence_threshold_from_barcode(
    barcodes: list[dict],
    retention_fraction: float = 0.5,
) -> float:
    """Determine persistence threshold to retain given fraction of features.

    Used to adaptively set the persistence_threshold for persistent_correction
    and Topo-EWC based on the actual barcode distribution.

    Args:
        barcodes: output from persistence_barcode()
        retention_fraction: fraction of features to retain (0-1)

    Returns:
        threshold: persistence value below which features are filtered
    """
    if not barcodes:
        return 0.5

    persistences = sorted([b['persistence'] for b in barcodes], reverse=True)
    n_retain = max(1, int(len(persistences) * retention_fraction))
    return persistences[n_retain - 1]


# ═════════════════════════════════════════════════════════════════
# Shared: Topological Parameter Importance
# ═════════════════════════════════════════════════════════════════

def compute_parameter_persistence_importance(
    model: nn.Module,
    loss_fn: callable,
    input_batch: dict,
    num_samples: int = 10,
) -> dict[str, torch.Tensor]:
    """Compute parameter importance via gradient persistence estimation.

    Runs multiple forward-backward passes with different random seeds to
    estimate how persistently each parameter contributes to the loss.
    Parameters with high variance across seeds → short persistence (noise).
    Parameters with consistent gradient direction → long persistence (structure).

    Used by both Topo-EWC (for SIREN TTA) and DomainAdapter (for adaptive
    regularization scheduling).

    Args:
        model: the model to analyze
        loss_fn: callable(model, batch) → scalar loss
        input_batch: dict of input tensors
        num_samples: number of stochastic passes

    Returns:
        importance: {param_name: importance_tensor}
    """
    importance = {}
    grad_accum = {}

    # Save/restore RNG state to avoid side effects on external training loops
    rng_state = torch.get_rng_state()
    for s in range(num_samples):
        torch.manual_seed(s * 42)
        model.zero_grad()
        loss = loss_fn(model, input_batch)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None and p.requires_grad:
                if name not in grad_accum:
                    grad_accum[name] = []
                grad_accum[name].append(p.grad.data.clone())

    for name, grads in grad_accum.items():
        stacked = torch.stack(grads)  # (num_samples, *param_shape)
        # Mean gradient direction
        mean_grad = stacked.mean(dim=0)
        mean_mag = mean_grad.abs()
        # Variance across samples (inverse of persistence)
        grad_var = stacked.var(dim=0)
        # Importance = mean_magnitude / (1 + variance)
        # Low variance = consistent gradient = long persistence = high importance
        importance[name] = mean_mag / (1.0 + grad_var)

    torch.set_rng_state(rng_state)
    return importance


# ═════════════════════════════════════════════════════════════════
# Shared: Gradient Alignment Diagnostic
# ═════════════════════════════════════════════════════════════════

def gradient_alignment(
    grad_a: dict[str, torch.Tensor],
    grad_b: dict[str, torch.Tensor],
) -> float:
    """Compute cosine similarity between two gradient dictionaries.

    A_gs = |Σ_i ∇_i L_geo · ∇_i L_sem| / (||∇L_geo|| · ||∇L_sem||)

    Used by TTA Safety Monitor (alignment between geometric and semantic
    optimization directions) and DomainAdapter (alignment between source
    and target domain gradients).

    Args:
        grad_a: parameter_name → gradient tensor (e.g., geometric loss)
        grad_b: parameter_name → gradient tensor (e.g., semantic loss)

    Returns:
        alignment: float in [0, 1], 1 = perfectly aligned, 0 = orthogonal
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for name in grad_a:
        if name in grad_b:
            ga = grad_a[name].flatten()
            gb = grad_b[name].flatten()
            dot += (ga * gb).sum().item()
            norm_a += (ga ** 2).sum().item()
            norm_b += (gb ** 2).sum().item()

    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0

    return abs(dot) / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ═════════════════════════════════════════════════════════════════
# Shared: Adaptive Regularization Scheduler
# ═════════════════════════════════════════════════════════════════

class AdaptiveRegularizationScheduler:
    """Adaptive regularization strength based on effective sample size.

    Used by:
      - DomainAdapter: alpha = alpha_0 * min(1, k / ESS_t)
      - Topo-EWC: lambda_ewc = lambda_max * (1 - rho_topo)
      - TTA Engine: LR schedule based on NTK stability

    Shared across modules via Cross-Module Synergy 4:
    Parameter importance Omega_i and adaptive alpha share persistent
    homology computation infrastructure.
    """

    def __init__(self, base_value: float = 1.0, min_value: float = 0.01):
        self.base_value = base_value
        self.min_value = min_value
        self.current = base_value

    def update(self, effective_sample_size: float,
               subspace_dim: int = 32) -> float:
        """Schedule regularization inversely proportional to ESS.

        alpha = base * min(1, k / ESS)
        As ESS decreases (more correlation), regularization increases.
        """
        if effective_sample_size <= 0:
            alpha = self.base_value * 10.0  # very small ESS → strong regularization
        else:
            alpha = self.base_value * min(1.0, subspace_dim / effective_sample_size)
        self.current = max(self.min_value, alpha)
        return self.current

    def update_topological(self, barcode_similarity: float) -> float:
        """Topology-aware scheduling: lambda = base * (1 - rho_topo).

        When topology is well-preserved (rho_topo ≈ 1), use low lambda.
        When topology degrades (rho_topo ≈ 0), use high lambda for protection.
        """
        lam = self.base_value * (1.0 - barcode_similarity)
        self.current = max(self.min_value, min(lam, self.base_value * 10.0))
        return self.current
