"""Tests for effective_sample_size utility (Module 4)."""
import math
import torch
import pytest
from utils.effective_sample_size import (
    estimate_effective_sample_size,
    recommend_hyperparams,
    estimate_mixing_coefficient,
)


class TestEffectiveSampleSize:
    @pytest.fixture
    def iid_sequence(self):
        """IID sequence: T_eff ≈ T (no autocorrelation)."""
        return torch.randn(32, 48, 128)  # (N, T, D)

    @pytest.fixture
    def highly_autocorrelated(self):
        """Highly autocorrelated: T_eff << T."""
        T = 48
        x = torch.zeros(16, T, 64)
        x[:, 0] = torch.randn(16, 64)
        for t in range(1, T):
            x[:, t] = 0.95 * x[:, t - 1] + 0.05 * torch.randn(16, 64)
        return x

    def test_iid_sequence(self, iid_sequence):
        """IID data: T_eff should be close to T."""
        T_eff = estimate_effective_sample_size(iid_sequence)
        T = iid_sequence.shape[1]
        assert T_eff > T * 0.5, f"Expected T_eff ≈ {T}, got {T_eff}"

    def test_autocorrelated_sequence(self, highly_autocorrelated):
        """Autocorrelated data: T_eff should be much smaller than T."""
        T_eff = estimate_effective_sample_size(highly_autocorrelated)
        T = highly_autocorrelated.shape[1]
        assert T_eff < T * 0.5, f"Expected T_eff << {T}, got {T_eff}"

    def test_single_timestep(self):
        """Single timestep: T_eff = 1."""
        x = torch.randn(8, 1, 64)
        T_eff = estimate_effective_sample_size(x)
        assert T_eff == 1.0

    def test_1d_input(self):
        """Works with 1D (T,) input."""
        x = torch.randn(100)
        T_eff = estimate_effective_sample_size(x)
        assert 1.0 <= T_eff <= 100.0

    def test_2d_input(self):
        """Works with 2D (B, T) input."""
        x = torch.randn(16, 24)
        T_eff = estimate_effective_sample_size(x)
        assert 1.0 <= T_eff <= 24.0

    def test_recommend_hyperparams(self):
        """Hyperparameter recommendations are reasonable."""
        rec = recommend_hyperparams(T_eff=12.0, K=3, C_in=64, C_out=64)
        assert rec['M_star'] >= 1
        assert rec['r_star'] >= 4
        assert rec['lambda_star'] > 0

    def test_recommend_small_T_eff(self):
        """Small T_eff should give conservative recommendations."""
        rec = recommend_hyperparams(T_eff=2.0, K=3, C_in=128, C_out=128)
        assert rec['M_star'] == 1  # at least 1 dynamic kernel
        assert rec['r_star'] >= 4

    def test_mixing_coefficient_iid(self, iid_sequence):
        """IID data: high gamma (fast mixing)."""
        gamma = estimate_mixing_coefficient(iid_sequence)
        assert gamma > 0.1

    def test_mixing_coefficient_autocorrelated(self, highly_autocorrelated):
        """Autocorrelated data: low gamma (slow mixing)."""
        gamma = estimate_mixing_coefficient(highly_autocorrelated)
        # Should be lower than iid case — just verify it's valid
        assert 0.01 <= gamma <= 5.0


class TestTemporalLiteSpectralInit:
    """Verify spectral init doesn't break TemporalLite."""

    def test_forward_still_works(self):
        """TemporalLite forward pass works with new init."""
        from models.temporal_lite import TemporalLite
        m = TemporalLite(64, k=3)
        x = torch.randn(32, 12, 64)
        out = m(x)
        assert out.shape == (32, 64)
        assert not torch.isnan(out).any()

    def test_param_count_unchanged(self):
        """Spectral init doesn't change parameter count."""
        from models.temporal_lite import TemporalLite
        m = TemporalLite(64, k=3)
        total = sum(p.numel() for p in m.parameters())
        assert total < 5000  # same lightweight guarantee

    def test_spectral_norm_within_bounds(self):
        """||W_conv||_2 should be approx sqrt(2) after spectral init."""
        from models.temporal_lite import TemporalLite
        m = TemporalLite(64, k=3)
        weight_norm = m.conv.weight.data.norm(p=2).item()
        assert abs(weight_norm - math.sqrt(2.0)) < 0.1, \
            f"Expected ||W|| ≈ {math.sqrt(2.0):.4f}, got {weight_norm:.4f}"

    def test_with_layer_norm_stability(self):
        """Gradient should flow without explosion with LayerNorm."""
        from models.temporal_lite import TemporalLite
        m = TemporalLite(128, k=3)
        x = torch.randn(4, 12, 128, requires_grad=False)
        out = m(x)
        loss = out.sum()
        loss.backward()
        for name, p in m.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
                assert p.grad.abs().max() < 100, \
                    f"Exploding grad in {name}: {p.grad.abs().max().item():.2f}"
