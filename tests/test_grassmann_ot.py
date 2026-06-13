"""Tests for grassmann_ot module (Module 2: Optimal Transport)."""
import torch
import pytest
from models.grassmann_ot import (
    sliced_gw_distance,
    grassmann_geodesic_distance,
    grassmann_basis,
    sliced_gw_barycentric_map,
    geometric_anchor_distance,
    JointOTGrassmannAligner,
)


class TestSlicedGW:
    def test_distance_nonnegative(self):
        """SGW distance is non-negative."""
        X = torch.randn(100, 64)
        Y = torch.randn(100, 64)
        d = sliced_gw_distance(X, Y, n_projections=16)
        assert d.item() >= 0

    def test_self_distance_zero(self):
        """SGW(X, X) ≈ 0."""
        X = torch.randn(100, 64)
        d = sliced_gw_distance(X, X, n_projections=32)
        assert d.item() < 0.5  # Approx zero for self-comparison

    def test_different_dimensions(self):
        """Works with different feature dimensions."""
        X = torch.randn(100, 32)
        Y = torch.randn(100, 64)
        d = sliced_gw_distance(X, Y, n_projections=16)
        assert d.item() >= 0

    def test_different_sizes(self):
        """Works with different sample sizes."""
        X = torch.randn(80, 64)
        Y = torch.randn(120, 64)
        d = sliced_gw_distance(X, Y, n_projections=16)
        assert d.item() >= 0

    def test_barycentric_map_shape(self):
        """Barycentric mapping preserves shape."""
        X = torch.randn(64, 128)
        Y = torch.randn(64, 128)
        mapped = sliced_gw_barycentric_map(X, Y, n_projections=16)
        assert mapped.shape == X.shape

    def test_barycentric_map_no_nan(self):
        """Barycentric mapping doesn't produce NaN."""
        X = torch.randn(32, 64)
        Y = torch.randn(32, 64)
        mapped = sliced_gw_barycentric_map(X, Y, n_projections=16)
        assert not torch.isnan(mapped).any()


class TestGrassmann:
    def test_geodesic_self_distance_zero(self):
        """d_G(U, U) = 0 for properly orthonormalized basis."""
        # Use QR to get truly orthonormal columns
        U_raw = torch.randn(64, 16)
        U, _ = torch.linalg.qr(U_raw)
        d = grassmann_geodesic_distance(U, U)
        assert d.item() < 0.1  # Nearly zero (numerical precision from arccos)

    def test_geodesic_distance_nonnegative(self):
        """d_G >= 0."""
        U = torch.randn(64, 16)
        U = U / U.norm(dim=0, keepdim=True)
        V = torch.randn(64, 16)
        V = V / V.norm(dim=0, keepdim=True)
        d = grassmann_geodesic_distance(U, V)
        assert d.item() >= 0

    def test_basis_shape(self):
        """Grassmann basis returns (D, k) — principal directions in feature space."""
        X = torch.randn(200, 128)
        U = grassmann_basis(X, k=32)
        assert U.shape == (128, 32), f"Expected (128, 32), got {U.shape}"

    def test_basis_orthonormal(self):
        """Grassmann basis columns are orthonormal."""
        X = torch.randn(200, 64)
        U = grassmann_basis(X, k=16)
        # U^T U ≈ I_k
        gram = U.T @ U
        eye = torch.eye(16)
        assert (gram - eye).abs().max() < 0.1


class TestGeometricAnchorDistance:
    def test_self_distance_zero(self):
        """Geometric anchor self-distance is near-zero on diagonal."""
        geo = torch.randn(50, 5)
        d = geometric_anchor_distance(geo, geo)
        diag = d.diag()
        assert diag.max() < 0.01  # numerical tolerance for FP32

    def test_shape(self):
        """Output shape is (N, M)."""
        a = torch.randn(30, 5)
        b = torch.randn(20, 5)
        d = geometric_anchor_distance(a, b)
        assert d.shape == (30, 20)


class TestJointOTGrassmannAligner:
    @pytest.fixture
    def aligner(self):
        return JointOTGrassmannAligner(
            subspace_dim=16, n_projections=8, alpha=0.1
        )

    def test_forward_shapes(self, aligner):
        """Aligner preserves input shapes."""
        opt = torch.randn(2, 64, 16, 16)
        sar = torch.randn(2, 64, 16, 16)
        opt_a, sar_a = aligner(opt, sar)
        assert opt_a.shape == opt.shape
        assert sar_a.shape == sar.shape

    def test_different_spatial_sizes(self, aligner):
        """Handles SAR at different resolution."""
        opt = torch.randn(1, 64, 16, 16)
        sar = torch.randn(1, 64, 8, 8)
        opt_a, sar_a = aligner(opt, sar)
        assert opt_a.shape == opt.shape
        assert sar_a.shape == (1, 64, 16, 16)

    def test_alignment_quality_diagnostic(self, aligner):
        """Quality diagnostic returns valid metrics."""
        opt = torch.randn(1, 64, 32, 32)
        sar = torch.randn(1, 64, 32, 32)
        quality = aligner.estimate_alignment_quality(opt, sar)
        assert 'd_G' in quality
        assert 'sgw' in quality
        assert quality['d_G'] >= 0
