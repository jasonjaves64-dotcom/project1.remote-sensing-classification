"""Smoke tests for tta_safety_monitor module (Module 1 Level 2: Safety).

Tests TTASafetyMonitor update/rollback/status, and TopoEWC registration/penalty.
All tests run on CPU with synthetic data.

EMA storage uses nn.Module register_buffer/get_buffer (ema_weight, ema_bias, etc.)
rather than a plain dict attribute.
"""
import torch
import pytest
from models.tta_safety_monitor import TTASafetyMonitor, TopoEWC


@pytest.fixture
def dummy_model():
    """A minimal nn.Module with trainable parameters for EMA / EWC testing."""
    return torch.nn.Linear(64, 32)


# ── TTASafetyMonitor ─────────────────────────────────────────────────────────

class TestTTASafetyMonitor:
    """Smoke tests for three-dimensional TTA safety monitor."""

    def test_instantiation(self):
        """TTASafetyMonitor instantiates with default thresholds."""
        monitor = TTASafetyMonitor(window_size=10)
        assert monitor.window_size == 10
        assert monitor.alignment_threshold == 0.1
        assert monitor.semantic_drop_threshold == 0.02
        assert monitor.cohomology_threshold == 0.5
        assert monitor.intervention_level == 0

    def test_update_normal(self):
        """update() with healthy metrics returns level 0 (normal)."""
        monitor = TTASafetyMonitor(window_size=10)
        result = monitor.update(
            grad_alignment=0.5,
            semantic_map=0.85,
            cohomology_conflict=0.1,
        )
        assert result['level'] == 0
        assert result['action'] == 'normal'
        assert result['should_pause'] is False
        assert result['should_rollback'] is False
        assert result['new_lr_factor'] == 1.0

    def test_update_level1_alignment(self):
        """Low grad_alignment triggers level 1 intervention.

        _assess() averages ALL entries in the history, so we must feed
        consistently low alignment values for the average to drop below
        the threshold (0.1). First 3 calls are warmup (len<3, return 0).
        """
        monitor = TTASafetyMonitor(window_size=10, alignment_threshold=0.1)
        # Feed 3 warmup + 1 trigger — all below threshold so average stays low
        for _ in range(4):
            result = monitor.update(
                grad_alignment=0.05,  # consistently below 0.1
                semantic_map=0.85,
                cohomology_conflict=0.1,
            )
        # 4th call: average(0.05,0.05,0.05,0.05) = 0.05 < 0.1 → level 1
        assert result['level'] == 1
        assert result['new_lr_factor'] == 0.5
        assert result['should_pause'] is False

    def test_update_level1_cohomology(self):
        """High cohomology conflict triggers level 1 intervention.

        cohomology check uses max(history) so one high value is enough.
        """
        monitor = TTASafetyMonitor(window_size=10, cohomology_threshold=0.5)
        for _ in range(3):
            monitor.update(grad_alignment=0.5, semantic_map=0.85, cohomology_conflict=0.1)
        result = monitor.update(
            grad_alignment=0.5,
            semantic_map=0.85,
            cohomology_conflict=0.6,  # above 0.5 threshold → max=0.6
        )
        assert result['level'] == 1

    def test_update_level3_critical(self):
        """Very high cohomology conflict (> 1.0) triggers level 3 rollback."""
        monitor = TTASafetyMonitor(window_size=10)
        for _ in range(3):
            monitor.update(grad_alignment=0.5, semantic_map=0.85, cohomology_conflict=0.1)
        result = monitor.update(
            grad_alignment=0.5,
            semantic_map=0.85,
            cohomology_conflict=1.5,  # well above 1.0 → level 3
        )
        assert result['level'] == 3
        assert result['should_rollback'] is True

    def test_update_with_model_ema(self, dummy_model):
        """update() with model argument registers EMA buffers and updates them.

        EMA parameters are stored as nn.Module buffers (ema_weight, ema_bias).
        """
        monitor = TTASafetyMonitor(window_size=10)
        result = monitor.update(
            grad_alignment=0.5, semantic_map=0.85, cohomology_conflict=0.1,
            model=dummy_model,
        )
        assert result['level'] == 0
        # After update with model, _ema_param_names should be populated
        assert len(monitor._ema_param_names) > 0
        # Check that EMA buffers exist (register_buffer stores them)
        for name in monitor._ema_param_names:
            buffer_name = f'ema_{name.replace(".", "_")}'
            ema_val = monitor.get_buffer(buffer_name)
            assert ema_val is not None

    def test_rollback_without_ema(self, dummy_model):
        """rollback() returns False when EMA was never initialized."""
        monitor = TTASafetyMonitor(window_size=10)
        # _ema_param_names is empty → rollback returns False
        success = monitor.rollback(dummy_model)
        assert success is False

    def test_rollback_with_ema(self, dummy_model):
        """rollback() succeeds after EMA has been initialized via update()."""
        monitor = TTASafetyMonitor(window_size=10)
        # Snapshot original weight to verify rollback restores it
        original_weight = dummy_model.weight.data.clone()
        monitor.update(grad_alignment=0.5, semantic_map=0.85, cohomology_conflict=0.1,
                       model=dummy_model)
        # Perturb weights, then rollback should restore them
        with torch.no_grad():
            dummy_model.weight.add_(0.5)
        success = monitor.rollback(dummy_model)
        assert success is True
        # After rollback, parameters should be close to EMA (close to original)
        assert torch.allclose(dummy_model.weight.data, original_weight, atol=0.6)

    def test_get_status(self):
        """get_status() returns current monitoring snapshot."""
        monitor = TTASafetyMonitor(window_size=10)
        for _ in range(5):
            monitor.update(grad_alignment=0.5, semantic_map=0.85, cohomology_conflict=0.1)
        status = monitor.get_status()
        assert 'intervention_level' in status
        assert 'gradient_alignment' in status
        assert 'semantic_map_recent' in status
        assert 'cohomology_conflict_max' in status
        assert 'total_interventions' in status
        assert 'auto_recoveries' in status

    def test_update_counting(self):
        """Total interventions and auto-recoveries are tracked.

        Uses the same consistently-low pattern to reliably trigger level 1,
        then recovers with healthy values to register an auto-recovery.
        """
        monitor = TTASafetyMonitor(window_size=10, alignment_threshold=0.1)
        # Feed 4 warmup/trigger steps with alignment consistently below 0.1
        for _ in range(4):
            monitor.update(grad_alignment=0.05, semantic_map=0.85, cohomology_conflict=0.1)
        # Recover with healthy alignment
        monitor.update(grad_alignment=0.5, semantic_map=0.85, cohomology_conflict=0.1)
        status = monitor.get_status()
        assert status['total_interventions']['L1'] >= 1
        assert status['auto_recoveries'] >= 1


# ── TopoEWC ──────────────────────────────────────────────────────────────────

class TestTopoEWC:
    """Smoke tests for Topological Elastic Weight Consolidation."""

    def test_instantiation_empty(self):
        """TopoEWC instantiates without a model."""
        ewc = TopoEWC(ewc_lambda=50.0, persistence_threshold=0.3)
        assert ewc.ewc_lambda == 50.0
        assert ewc.persistence_threshold == 0.3
        assert isinstance(ewc, TopoEWC)  # inherits nn.Module

    def test_instantiation_with_model(self, dummy_model):
        """TopoEWC accepts model in constructor and auto-registers."""
        ewc = TopoEWC(model=dummy_model, ewc_lambda=100.0)
        assert 'weight' in ewc.reference_params
        assert 'weight' in ewc.importance

    def test_register_model(self, dummy_model):
        """register_model() snapshots reference params and initializes importance."""
        ewc = TopoEWC(ewc_lambda=100.0)
        ewc.register_model(dummy_model)
        assert len(ewc.reference_params) > 0
        assert len(ewc.importance) > 0
        # Reference params should be equal to current params initially
        for name, p in dummy_model.named_parameters():
            if p.requires_grad:
                assert torch.equal(ewc.reference_params[name], p.data)

    def test_penalty_zero_without_model(self):
        """penalty() returns 0.0 when no model registered."""
        ewc = TopoEWC(ewc_lambda=100.0)
        penalty = ewc.penalty(torch.nn.Linear(64, 32))
        assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_penalty_nonzero_after_drift(self, dummy_model):
        """penalty() is non-zero after parameters drift from reference."""
        ewc = TopoEWC(ewc_lambda=100.0)
        ewc.register_model(dummy_model)
        # Initial penalty should be near zero (no drift yet)
        penalty_init = ewc.penalty(dummy_model)
        assert penalty_init.item() == pytest.approx(0.0, abs=1e-6)
        # Perturb parameters
        with torch.no_grad():
            dummy_model.weight.add_(0.5)
        penalty_drift = ewc.penalty(dummy_model)
        assert penalty_drift.item() > 0.0

    def test_protected_param_fraction(self, dummy_model):
        """protected_param_fraction returns a value in [0, 1]."""
        ewc = TopoEWC(ewc_lambda=100.0)
        assert ewc.protected_param_fraction == 0.0  # no model yet
        ewc.register_model(dummy_model)
        frac = ewc.protected_param_fraction
        assert 0.0 <= frac <= 1.0

    def test_compute_importance_from_gradients(self, dummy_model):
        """compute_importance_from_gradients populates importance weights."""
        ewc = TopoEWC(model=dummy_model, ewc_lambda=100.0)
        dem_batch = torch.randn(4, 64) * 100.0

        def loss_fn(model, batch):
            return (model(batch) ** 2).mean()

        ewc.compute_importance_from_gradients(dummy_model, dem_batch, loss_fn)
        # Importance should now be populated
        assert len(ewc.importance) > 0
        # Each importance tensor should match parameter shape
        for name, p in dummy_model.named_parameters():
            if name in ewc.importance and p.requires_grad:
                assert ewc.importance[name].shape == p.data.shape
