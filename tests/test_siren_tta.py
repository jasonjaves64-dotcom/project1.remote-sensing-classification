"""Smoke tests for siren_tta module (Module 1: TTA Engine).

Tests SIRENTTALoss, NTKStabilityMonitor, LRSAAdapter, HMAAdapter, and TTAEngine
instantiation, forward passes, and key method behaviors. All tests on CPU.
"""
import torch
import pytest
from models.siren_tta import (
    SIRENTTALoss,
    NTKStabilityMonitor,
    LRSAAdapter,
    HMAAdapter,
    TTAEngine,
)


# ── Helper fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def normals():
    """Unit normal vectors: (B, 3, H, W)."""
    # Generate random vectors and normalize to unit length
    raw = torch.randn(2, 3, 32, 32)
    norm = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return raw / norm


@pytest.fixture
def curvature_maps():
    """Mean and Gaussian curvature maps: (B, 1, H, W) each."""
    mean_curv = torch.randn(2, 1, 32, 32) * 0.01
    gauss_curv = torch.randn(2, 1, 32, 32) * 0.001
    return mean_curv, gauss_curv


@pytest.fixture
def dem_surface():
    """SIREN-evaluated DEM surface: (B, 1, H, W)."""
    return torch.randn(2, 1, 32, 32) * 100.0 + 500.0


# ── SIRENTTALoss ─────────────────────────────────────────────────────────────

class TestSIRENTTALoss:
    """Smoke tests for TTA geometric loss function."""

    def test_instantiation(self):
        """SIRENTTALoss creates learnable log-variance parameters."""
        loss_fn = SIRENTTALoss(omega_0=30.0)
        assert isinstance(loss_fn, SIRENTTALoss)
        assert loss_fn.omega_0 == 30.0
        assert hasattr(loss_fn, 'log_w_n')
        assert hasattr(loss_fn, 'log_w_c')
        assert hasattr(loss_fn, 'log_w_p')

    def test_normal_consistency_loss(self, normals):
        """Normal consistency loss returns positive scalar."""
        loss_fn = SIRENTTALoss()
        loss = loss_fn.normal_consistency_loss(normals, k=3)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0.0

    def test_curvature_smoothness_loss(self, curvature_maps):
        """Curvature smoothness loss returns positive scalar."""
        mean_curv, gauss_curv = curvature_maps
        loss_fn = SIRENTTALoss()
        loss = loss_fn.curvature_smoothness_loss(mean_curv, gauss_curv)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_photometric_without_optical(self, dem_surface):
        """Photometric loss returns zero when optical image is None."""
        loss_fn = SIRENTTALoss()
        loss = loss_fn.photometric_alignment_loss(dem_surface, optical_image=None)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_photometric_with_optical(self, dem_surface):
        """Photometric loss with synthetic optical image returns a value."""
        optical = torch.randn(2, 3, 32, 32)
        loss_fn = SIRENTTALoss()
        loss = loss_fn.photometric_alignment_loss(dem_surface, optical_image=optical)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_forward(self, normals, curvature_maps, dem_surface):
        """Forward pass returns (total_loss, dict) tuple."""
        mean_curv, gauss_curv = curvature_maps
        loss_fn = SIRENTTALoss()
        total, losses = loss_fn(normals, mean_curv, gauss_curv, dem_surface)
        assert isinstance(total, torch.Tensor)
        assert total.ndim == 0
        assert isinstance(losses, dict)
        for key in ['l_norm', 'l_curv', 'l_photo']:
            assert key in losses


# ── NTKStabilityMonitor ──────────────────────────────────────────────────────

class TestNTKStabilityMonitor:
    """Smoke tests for NTK stability monitoring."""

    @pytest.fixture
    def dummy_model(self):
        """A minimal model with a single trainable Linear layer."""
        return torch.nn.Linear(256, 128)

    def test_instantiation(self):
        """NTKStabilityMonitor instantiates with default params."""
        monitor = NTKStabilityMonitor(max_ntk_drift=0.05, window_size=10)
        assert monitor.max_drift == 0.05
        assert monitor.window_size == 10
        assert monitor.initial_params is None

    def test_initialize_snapshots_params(self, dummy_model):
        """initialize() snapshots model parameters."""
        monitor = NTKStabilityMonitor()
        monitor.initialize(dummy_model)
        assert monitor.initial_params is not None
        # At least weight and bias should be captured
        assert 'weight' in monitor.initial_params

    def test_check_after_initialize(self, dummy_model):
        """check() returns safe=True immediately after initialize (no drift yet)."""
        monitor = NTKStabilityMonitor()
        monitor.initialize(dummy_model)
        result = monitor.check(dummy_model)
        assert 'drift' in result
        assert 'safe' in result
        assert 'warning' in result
        assert result['safe'] is True
        assert result['drift'] == pytest.approx(0.0, abs=1e-6)

    def test_check_without_initialize_auto_init(self, dummy_model):
        """check() auto-initializes if not yet initialized."""
        monitor = NTKStabilityMonitor()
        result = monitor.check(dummy_model)
        assert result['safe'] is True
        assert monitor.initial_params is not None

    def test_check_detects_drift(self, dummy_model):
        """check() detects parameter drift after weight perturbation."""
        monitor = NTKStabilityMonitor(max_ntk_drift=0.01)
        monitor.initialize(dummy_model)
        # Perturb weights significantly
        with torch.no_grad():
            dummy_model.weight.add_(torch.randn_like(dummy_model.weight) * 0.1)
        result = monitor.check(dummy_model)
        # Drift should be non-zero and likely exceeds the threshold
        assert result['drift'] > 0.0
        # Whether it's safe depends on perturbation magnitude; just check return shape
        assert isinstance(result['safe'], bool)


# ── LRSAAdapter ───────────────────────────────────────────────────────────────

class TestLRSAAdapter:
    """Smoke tests for Low-Rank Sine Adapter."""

    def test_instantiation_and_forward(self):
        """LRSAAdapter forward computes low-rank modulated linear transform."""
        adapter = LRSAAdapter(in_features=256, out_features=128, rank=4)
        x = torch.randn(8, 256)
        base_weight = torch.randn(128, 256)
        out = adapter(x, base_weight)
        assert out.shape == (8, 128)

    def test_trainable_ratio(self):
        """trainable_ratio reflects fraction of trainable params."""
        adapter = LRSAAdapter(in_features=256, out_features=128, rank=4)
        ratio = adapter.trainable_ratio
        assert 0.0 < ratio < 1.0
        # rank=4: A=(128,4), B=(4,256) = 512+1024=1536 trainable
        # total = 1536(trainable) + 128*256(base) = 1536 + 32768 = 34304
        # ratio ≈ 1536 / 34304 ≈ 0.0448
        assert 0.01 < ratio < 0.1


# ── HMAAdapter ────────────────────────────────────────────────────────────────

class TestHMAAdapter:
    """Smoke tests for HyperNetwork Modulation Adapter."""

    def test_instantiation_and_forward(self):
        """HMAAdapter forward applies predicted scale+shift modulation."""
        adapter = HMAAdapter(d_model=128, hidden=16)
        x = torch.randn(4, 32, 128)  # (batch, seq, d_model)
        out = adapter(x)
        assert out.shape == (4, 32, 128)

    def test_with_explicit_drift(self):
        """HMAAdapter accepts optional drift_indicator."""
        adapter = HMAAdapter(d_model=64, hidden=16)
        x = torch.randn(4, 64)
        drift = torch.randn(4, 1) * 0.1
        out = adapter(x, drift_indicator=drift)
        assert out.shape == (4, 64)


# ── TTAEngine ─────────────────────────────────────────────────────────────────

class TestTTAEngine:
    """Smoke tests for TTA orchestration engine."""

    @pytest.fixture
    def engine(self):
        """Pre-built TTAEngine with loss and monitor."""
        tta_loss = SIRENTTALoss()
        ntk_monitor = NTKStabilityMonitor()
        return TTAEngine(tta_loss, ntk_monitor, lr=1e-4, max_steps=8)

    @pytest.fixture
    def tiny_model(self):
        """A tiny model with a single trainable parameter."""
        return torch.nn.Linear(1, 1)

    @pytest.fixture
    def adapt_inputs(self):
        """Synthetic inputs for adapt_step."""
        return {
            'dem_batch': torch.randn(2, 1, 16, 16) * 100.0 + 500.0,
            'normals': torch.randn(2, 3, 16, 16),
            'mean_curv': torch.randn(2, 1, 16, 16) * 0.01,
            'gauss_curv': torch.randn(2, 1, 16, 16) * 0.001,
        }

    def test_engine_instantiation(self, engine):
        """TTAEngine instantiates with expected attributes."""
        assert engine.lr == 1e-4
        assert engine.max_steps == 8
        assert engine.step_count == 0

    def test_initialize(self, engine, tiny_model):
        """initialize() snapshots model state."""
        engine.initialize(tiny_model)
        assert engine.ntk_monitor.initial_params is not None

    def test_adapt_step_safe(self, engine, tiny_model, adapt_inputs):
        """adapt_step runs one iteration and returns status dict."""
        engine.initialize(tiny_model)
        # Mark single weight as requiring grad
        tiny_model.weight.requires_grad = True
        result = engine.adapt_step(
            tiny_model,
            dem_batch=adapt_inputs['dem_batch'],
            normals=adapt_inputs['normals'],
            mean_curv=adapt_inputs['mean_curv'],
            gauss_curv=adapt_inputs['gauss_curv'],
        )
        assert 'loss' in result
        assert 'ntk_drift' in result
        assert 'safe' in result
        assert 'lr_used' in result
        assert 'status' in result
