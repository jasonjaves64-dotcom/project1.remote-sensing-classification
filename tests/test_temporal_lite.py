"""Tests for TemporalLite module."""
import torch
import pytest
from models.temporal_lite import TemporalLite


class TestTemporalLite:
    @pytest.fixture
    def module_64(self):
        return TemporalLite(d_model=64, k=3)

    @pytest.fixture
    def module_128(self):
        return TemporalLite(d_model=128, k=5)

    def test_forward_shape(self, module_64):
        """Output shape: (B*HW, T, D) -> (B*HW, D)"""
        B, T, D = 32, 12, 64
        x = torch.randn(B, T, D)
        out = module_64(x)
        assert out.shape == (B, D)

    def test_forward_shape_128(self, module_128):
        B, T, D = 16, 24, 128
        x = torch.randn(B, T, D)
        out = module_128(x)
        assert out.shape == (B, D)

    def test_deterministic_in_eval(self, module_64):
        """Same input -> same output in eval mode."""
        module_64.eval()
        x = torch.randn(64, 12, 64)
        out1 = module_64(x)
        out2 = module_64(x)
        assert torch.allclose(out1, out2)

    def test_no_nan(self, module_64):
        """Output contains no NaN or Inf."""
        x = torch.randn(128, 12, 64)
        out = module_64(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gate_is_learnable(self, module_64):
        """gate parameter receives gradient."""
        x = torch.randn(8, 12, 64)
        out = module_64(x)
        loss = out.sum()
        loss.backward()
        assert module_64.gate.grad is not None

    def test_handles_variable_T(self, module_64):
        """Works with different sequence lengths."""
        for T in [6, 12, 24]:
            x = torch.randn(16, T, 64)
            out = module_64(x)
            assert out.shape == (16, 64)

    def test_handles_k3_and_k5(self):
        """Both kernel sizes work."""
        m3 = TemporalLite(64, k=3)
        m5 = TemporalLite(64, k=5)
        x = torch.randn(8, 12, 64)
        assert m3(x).shape == (8, 64)
        assert m5(x).shape == (8, 64)

    def test_param_count(self, module_64):
        """Extremely lightweight: < 5K params for d_model=64."""
        total = sum(p.numel() for p in module_64.parameters())
        assert total < 5000, f"Expected <5K params, got {total}"

    def test_k1_fallback(self):
        """k=1 works (no padding needed, basic temporal mix)."""
        m = TemporalLite(32, k=1)
        x = torch.randn(4, 6, 32)
        out = m(x)
        assert out.shape == (4, 32)
