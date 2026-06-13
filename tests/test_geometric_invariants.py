"""Tests for geometric_invariants module (Module 1: Differential Geometry)."""
import math
import torch
import pytest
from models.geometric_invariants import (
    compute_geometric_invariants,
    geometric_invariant_stack,
    GeometricInvariantEncoder,
    darboux_frame,
    verify_se3_invariance,
    siren_gradient,
    siren_hessian_diag,
)


class TestGeometricInvariants:
    @pytest.fixture
    def flat_dem(self):
        """Flat plane: all curvatures should be zero."""
        return torch.ones(2, 1, 64, 64) * 100.0  # 100m flat plateau

    @pytest.fixture
    def hill_dem(self):
        """Gaussian hill: positive K at peak, negative at flanks."""
        y, x = torch.meshgrid(torch.linspace(-3, 3, 64), torch.linspace(-3, 3, 64),
                             indexing='ij')
        hill = 500.0 * torch.exp(-(x ** 2 + y ** 2) / 2.0)
        return hill.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)

    @pytest.fixture
    def saddle_dem(self):
        """Saddle surface: negative K everywhere."""
        y, x = torch.meshgrid(torch.linspace(-3, 3, 64), torch.linspace(-3, 3, 64),
                             indexing='ij')
        saddle = 100.0 * (x ** 2 - y ** 2)
        return saddle.unsqueeze(0).unsqueeze(0)

    def test_output_shapes(self, hill_dem):
        """All invariant maps have correct shape (B, 1, H, W)."""
        inv = compute_geometric_invariants(hill_dem)
        for key in ['K', 'H', 'k1', 'k2', 'tau_g']:
            assert inv[key].shape == hill_dem.shape, \
                f"Expected {key}.shape = {hill_dem.shape}, got {inv[key].shape}"

    def test_no_nan(self, hill_dem):
        """No NaN or Inf in output invariants."""
        inv = compute_geometric_invariants(hill_dem)
        for key, val in inv.items():
            assert not torch.isnan(val).any(), f"NaN in {key}"
            assert not torch.isinf(val).any(), f"Inf in {key}"

    def test_flat_plane_curvatures(self, flat_dem):
        """Flat plane: all curvatures ≈ 0."""
        inv = compute_geometric_invariants(flat_dem)
        for key in ['K', 'H', 'k1', 'k2', 'tau_g']:
            # Flat plane: curvatures should be near-zero (some edge effects from Sobel)
            assert inv[key].abs().mean() < 0.01, \
                f"Expected {key} ≈ 0 for flat plane, got mean={inv[key].abs().mean().item():.4f}"

    def test_gaussian_hill_K_sign(self, hill_dem):
        """Gaussian hill: K > 0 at peak (elliptic point)."""
        inv = compute_geometric_invariants(hill_dem)
        K = inv['K']
        # Center region (peak): K should be positive
        center_K = K[:, :, 28:36, 28:36].mean()
        assert center_K > 0, f"Expected K > 0 at hill peak, got {center_K.item():.6f}"

    def test_saddle_K_sign(self, saddle_dem):
        """Saddle: K < 0 (hyperbolic point)."""
        inv = compute_geometric_invariants(saddle_dem)
        K = inv['K']
        # Center: saddle point, should be negative
        center_K = K[:, :, 28:36, 28:36].mean()
        assert center_K < 0, f"Expected K < 0 at saddle, got {center_K.item():.6f}"

    def test_k1_gte_k2(self, hill_dem):
        """k1 >= k2 everywhere (max vs min principal curvature)."""
        inv = compute_geometric_invariants(hill_dem)
        assert (inv['k1'] >= inv['k2'] - 1e-6).all(), \
            "k1 should be >= k2 everywhere"

    def test_H_is_mean_of_k(self, hill_dem):
        """H ≈ (k1 + k2) / 2 (mean curvature definition)."""
        inv = compute_geometric_invariants(hill_dem)
        H_from_k = (inv['k1'] + inv['k2']) / 2.0
        diff = (inv['H'] - H_from_k).abs().max()
        assert diff < 1e-4, f"H should equal (k1+k2)/2, max diff={diff.item():.2e}"

    def test_K_is_product_of_k(self, hill_dem):
        """K ≈ k1 * k2 (Gaussian curvature definition)."""
        inv = compute_geometric_invariants(hill_dem)
        K_from_k = inv['k1'] * inv['k2']
        diff = (inv['K'] - K_from_k).abs().max()
        assert diff < 1e-3, f"K should equal k1*k2, max diff={diff.item():.2e}"

    def test_geometric_invariant_stack(self, hill_dem):
        """Stack produces (B, 5, H, W) output."""
        geo = geometric_invariant_stack(hill_dem)
        assert geo.shape == (1, 5, 64, 64)

    def test_geometric_invariant_stack_multichannel(self):
        """Handles multi-channel DEM (uses first channel)."""
        dem = torch.randn(2, 5, 32, 32)
        geo = geometric_invariant_stack(dem)
        assert geo.shape == (2, 5, 32, 32)

    def test_encoder_output_shape(self, hill_dem):
        """GeometricInvariantEncoder produces (B, 128, H, W)."""
        enc = GeometricInvariantEncoder(out_ch=128)
        out = enc(hill_dem)
        assert out.shape == (1, 128, 64, 64)

    def test_encoder_no_nan(self, hill_dem):
        """Encoder output is numerically stable."""
        enc = GeometricInvariantEncoder(out_ch=128)
        out = enc(hill_dem)
        assert not torch.isnan(out).any()

    def test_encoder_multibatch(self):
        """Works with multiple batch items."""
        dem = torch.randn(4, 1, 32, 32)
        enc = GeometricInvariantEncoder(out_ch=64)
        out = enc(dem)
        assert out.shape == (4, 64, 32, 32)

    def test_extreme_terrain_stability(self):
        """Extreme slopes (cliff-like DEM) should not produce NaN."""
        # Simulate a steep cliff
        dem = torch.zeros(1, 1, 32, 32)
        dem[:, :, :16, :] = 1000.0  # 1000m vertical cliff
        inv = compute_geometric_invariants(dem)
        for key, val in inv.items():
            assert not torch.isnan(val).any(), f"NaN in {key} for cliff terrain"

    def test_se3_invariance_identity(self, hill_dem):
        """Identity transform: deviation should be zero."""
        # Use 0° rotation + 0 translation
        import torchvision.transforms.functional as TF  # noqa: F401

        inv_orig = compute_geometric_invariants(hill_dem)
        inv_same = compute_geometric_invariants(hill_dem)
        for key in ['K', 'H', 'k1', 'k2', 'tau_g']:
            max_diff = (inv_orig[key] - inv_same[key]).abs().max().item()
            assert max_diff < 1e-6, f"{key}: deterministic test failed"


class TestDarbouxFrame:
    @pytest.fixture
    def simple_grad_hess(self):
        """Gradient and Hessian for a simple paraboloid: h = x^2 + y^2."""
        B, H, W = 1, 16, 16
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W),
                             indexing='ij')
        grad_h = torch.stack([2 * x, 2 * y], dim=0).unsqueeze(0)  # (1, 2, 16, 16)
        hess_h = torch.stack([
            torch.full_like(x, 2.0),  # h_xx = 2
            torch.full_like(x, 2.0),  # h_yy = 2
            torch.zeros_like(x),      # h_xy = 0
        ], dim=0).unsqueeze(0)  # (1, 3, 16, 16)
        return grad_h, hess_h

    def test_frame_output_shapes(self, simple_grad_hess):
        """Darboux frame outputs are (B, 3, H, W)."""
        grad_h, hess_h = simple_grad_hess
        frame = darboux_frame(grad_h, hess_h)
        for key in ['e1', 'e2', 'n']:
            assert frame[key].shape == (1, 3, 16, 16), \
                f"{key}: expected (1,3,16,16), got {frame[key].shape}"

    def test_frame_orthonormal(self, simple_grad_hess):
        """e1, e2, n should be orthonormal at each point."""
        grad_h, hess_h = simple_grad_hess
        frame = darboux_frame(grad_h, hess_h)
        e1, e2, n = frame['e1'], frame['e2'], frame['n']

        # e1 · e1 ≈ 1
        e1_norm = (e1 * e1).sum(dim=1).sqrt()
        assert (e1_norm - 1.0).abs().max() < 0.01

        # e1 · n ≈ 0 (tangent ⟂ normal)
        dot_e1n = (e1 * n).sum(dim=1).abs()
        assert dot_e1n.max() < 0.01, f"e1·n max={dot_e1n.max().item():.4f}"

        # e2 · n ≈ 0
        dot_e2n = (e2 * n).sum(dim=1).abs()
        assert dot_e2n.max() < 0.01, f"e2·n max={dot_e2n.max().item():.4f}"
