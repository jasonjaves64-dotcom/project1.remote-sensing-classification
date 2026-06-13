"""Tests for topological_evidence module (Module 3: Algebraic Topology)."""
import torch
import pytest
from models.topological_evidence import (
    dirichlet_to_chain,
    cup_product,
    cohomology_conflict_detector,
    persistent_correction,
    TopologicalConflictClassifier,
)


class TestDirichletToChain:
    def test_chain_output(self):
        """Dirichlet → chain: (*, K) input, class is last dim."""
        # Input: (*, K) format — class dimension last
        alpha = torch.randn(4, 16, 16, 7)  # (B, H, W, K)
        chain_0, boundary_norm = dirichlet_to_chain(alpha)
        # chain_0 preserves input shape
        assert chain_0.shape == (4, 16, 16, 7)
        # boundary_norm has spatial dims (B, H, W)
        assert boundary_norm.shape == (4, 16, 16)

    def test_boundary_norm_nonnegative(self):
        """Boundary norm is non-negative."""
        alpha = torch.ones(2, 5, 8, 8) * 10.0
        _, boundary_norm = dirichlet_to_chain(alpha)
        assert (boundary_norm >= 0).all()

    def test_uniform_evidence(self):
        """Uniform evidence produces finite boundary norm."""
        # Uniform Dirichlet → equal probabilities → non-zero boundary
        # because DS conflict measures evidential spread, not agreement
        alpha = torch.ones(1, 7, 4, 4) * 2.0
        _, boundary = dirichlet_to_chain(alpha)
        assert not torch.isnan(boundary).any()
        assert not torch.isinf(boundary).any()


class TestCupProduct:
    def test_elementwise_product(self):
        """Cup product of singleton co-chains = element-wise product."""
        omega1 = torch.tensor([[0.6, 0.3, 0.1]])
        omega2 = torch.tensor([[0.5, 0.4, 0.1]])
        combined, kappa = cup_product(omega1, omega2)
        assert combined.shape == (1, 3)
        assert kappa.shape == (1,)
        assert combined[0, 0].item() == pytest.approx(0.3, rel=0.01)

    def test_kappa_for_identical_distributions(self):
        """Kappa measures inherent conflict in uncertain evidence.

        Even identical probability distributions have non-zero DS kappa
        because kappa = sum_{i≠j} p_i * p_j measures how evidence is
        spread across classes (not disagreement between modalities).
        Zero kappa only occurs for one-hot (fully certain) distributions.
        """
        omega = torch.tensor([[1.0, 0.0, 0.0]])  # one-hot = zero conflict
        _, kappa = cup_product(omega, omega)
        assert kappa.item() == pytest.approx(0.0, abs=0.01)

    def test_kappa_positive_for_different(self):
        """Conflict coefficient > 0 when distributions disagree."""
        omega1 = torch.tensor([[0.9, 0.1]])
        omega2 = torch.tensor([[0.1, 0.9]])
        _, kappa = cup_product(omega1, omega2)
        assert kappa.item() > 0.5

    def test_batch_input(self):
        """Works with batch dimensions."""
        omega1 = torch.randn(4, 7)
        omega1 = omega1.softmax(dim=-1)
        omega2 = torch.randn(4, 7)
        omega2 = omega2.softmax(dim=-1)
        combined, kappa = cup_product(omega1, omega2)
        assert combined.shape == (4, 7)
        assert kappa.shape == (4,)


class TestCohomologyConflict:
    def test_noise_detection(self):
        """One-hot identical evidence → Noise type (zero conflict)."""
        omega1 = torch.tensor([[1.0, 0.0, 0.0]])
        omega2 = torch.tensor([[1.0, 0.0, 0.0]])
        result = cohomology_conflict_detector(omega1, omega2)
        assert result['conflict_type'] == 'Noise'
        assert result['is_exact']

    def test_structural_detection(self):
        """Strongly conflicting → Structural or HighOrder."""
        omega1 = torch.tensor([[0.95, 0.05]])
        omega2 = torch.tensor([[0.05, 0.95]])
        result = cohomology_conflict_detector(omega1, omega2)
        assert result['conflict_type'] in ('Structural', 'HighOrder')
        assert not result['is_exact']

    def test_returns_kappa(self):
        """Always returns kappa coefficient."""
        omega1 = torch.randn(2, 5).softmax(dim=-1)
        omega2 = torch.randn(2, 5).softmax(dim=-1)
        result = cohomology_conflict_detector(omega1, omega2)
        assert 'kappa' in result
        assert 'h1_norm' in result


class TestPersistentCorrection:
    def test_shape_preserved(self):
        """Correction preserves input shape."""
        omega = torch.randn(4, 7)
        omega = omega.softmax(dim=-1)
        corrected = persistent_correction(omega)
        assert corrected.shape == omega.shape

    def test_no_nan(self):
        """Correction does not introduce NaN."""
        omega = torch.randn(3, 7).softmax(dim=-1)
        corrected = persistent_correction(omega)
        assert not torch.isnan(corrected).any()

    def test_reduces_noise(self):
        """Correction reduces low-confidence components."""
        omega = torch.tensor([[0.5, 0.02, 0.02, 0.46]])
        corrected = persistent_correction(omega, persistence_threshold=0.3)
        # Small components should be suppressed
        assert corrected[0, 1].item() <= omega[0, 1].item() + 1e-6
        assert corrected[0, 2].item() <= omega[0, 2].item() + 1e-6


class TestTopologicalConflictClassifier:
    @pytest.fixture
    def classifier(self):
        return TopologicalConflictClassifier(num_classes=7)

    @pytest.fixture
    def low_conflict_input(self):
        # Concentrated evidence on same class = genuinely low conflict
        B, K, H, W = 2, 7, 8, 8
        # All experts strongly agree on class 0
        alpha_opt = torch.ones(B, K, H, W)
        alpha_opt[:, 0] = 100.0  # optical: strongly class 0
        alpha_sar = torch.ones(B, K, H, W)
        alpha_sar[:, 0] = 100.0  # SAR: strongly class 0
        alpha_fused = torch.ones(B, K, H, W)
        alpha_fused[:, 0] = 200.0  # fused: even stronger class 0
        return alpha_opt, alpha_sar, alpha_fused

    @pytest.fixture
    def high_conflict_input(self):
        # Conflicting evidence
        B, K, H, W = 2, 7, 8, 8
        alpha_opt = torch.zeros(B, K, H, W)
        alpha_opt[:, 0] = 100.0  # optical strongly favors class 0
        alpha_sar = torch.zeros(B, K, H, W)
        alpha_sar[:, 1] = 100.0  # SAR strongly favors class 1
        alpha_fused = torch.zeros(B, K, H, W)
        alpha_fused[:, 0] = 50.0  # fused leans toward class 0
        return alpha_opt, alpha_sar, alpha_fused

    def test_output_keys(self, classifier, low_conflict_input):
        """Forward returns expected keys."""
        result = classifier(*low_conflict_input)
        for key in ['conflict_type', 'conflict_mask', 'safe_for_pseudo', 'kappa_map']:
            assert key in result, f"Missing key: {key}"

    def test_low_conflict_safe(self, classifier, low_conflict_input):
        """Low conflict input → safe for pseudo-labeling."""
        result = classifier(*low_conflict_input)
        # Most pixels should be safe
        assert result['safe_for_pseudo'].float().mean() > 0.5

    def test_high_conflict_unsafe(self, classifier, high_conflict_input):
        """High conflict input → fewer safe pixels."""
        result = classifier(*high_conflict_input)
        # Many pixels should be flagged as unsafe
        unsafe_ratio = 1.0 - result['safe_for_pseudo'].float().mean()
        assert unsafe_ratio > 0.3  # At least some pixels flagged

    def test_per_pixel_mode(self, classifier):
        """Per-pixel mode works with flattened inputs."""
        N, K = 64, 7
        alpha_opt = torch.ones(N, K) * 10.0
        alpha_sar = torch.ones(N, K) * 10.0
        safe = classifier.forward_per_pixel(alpha_opt, alpha_sar)
        assert safe.shape == (N,)
        assert safe.dtype == torch.bool

    def test_conflict_type_list(self, classifier, low_conflict_input):
        """Conflict types are returned per sample."""
        result = classifier(*low_conflict_input)
        assert len(result['conflict_type']) == 2
        for t in result['conflict_type']:
            assert t in ('Noise', 'Structural', 'HighOrder')

    def test_kappa_map_shape(self, classifier, low_conflict_input):
        """Kappa map matches spatial dimensions."""
        result = classifier(*low_conflict_input)
        assert result['kappa_map'].shape == (2, 8, 8)
