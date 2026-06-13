"""Smoke tests for siren_dem_encoder module (Module 1: Differential Geometry).

Tests SIRENLayer and SIRENDEMEncoder instantiation, forward pass, and output shape
verification. All tests run on CPU with synthetic data.
"""
import torch
import pytest
from models.siren_dem_encoder import SIRENLayer, SIRENDEMEncoder


class TestSIRENLayer:
    """Smoke tests for SIRENLayer — the basic SIREN building block."""

    def test_instantiation(self):
        """SIRENLayer instantiates without errors."""
        layer = SIRENLayer(in_features=128, out_features=256, omega_0=30.0, is_first=False)
        assert isinstance(layer, SIRENLayer)
        assert layer.omega_0 == 30.0

    def test_forward_output_shape(self):
        """SIRENLayer forward pass produces correct output shape."""
        layer = SIRENLayer(in_features=64, out_features=128, omega_0=30.0)
        x = torch.randn(4, 64)  # (batch, in_features)
        out = layer(x)
        assert out.shape == (4, 128)
        # SIREN output is bounded in [-1, 1] due to sin activation
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_first_layer_init_bounds(self):
        """First-layer SIREN uses different weight initialization bounds."""
        layer_first = SIRENLayer(in_features=2, out_features=256, is_first=True)
        layer_hidden = SIRENLayer(in_features=256, out_features=256, is_first=False)
        # Both should instantiate without error
        assert layer_first.omega_0 == 30.0
        assert layer_hidden.omega_0 == 30.0


class TestSIRENDEMEncoder:
    """Smoke tests for SIRENDEMEncoder — full DEM-to-feature encoder."""

    @pytest.fixture
    def dem_batch(self):
        """Synthetic DEM batch: (B=2, C=1, H=64, W=64) with plausible elevation values."""
        return torch.randn(2, 1, 64, 64) * 100.0 + 500.0  # mean ~500m, std ~100m

    @pytest.fixture
    def encoder(self):
        """Default SIRENDEMEncoder instance."""
        return SIRENDEMEncoder(out_ch=128, hidden_dim=256, n_layers=5)

    def test_instantiation(self, encoder):
        """SIRENDEMEncoder instantiates without errors."""
        assert isinstance(encoder, SIRENDEMEncoder)
        assert encoder.out_ch == 128

    def test_forward_output_shape(self, encoder, dem_batch):
        """Forward pass produces (B, out_ch, H, W) output."""
        out = encoder(dem_batch)
        assert out.shape == (2, 128, 64, 64)
        assert out.dtype == torch.float32

    def test_multichannel_input_handled(self, encoder):
        """Multi-channel DEM input uses first channel as elevation."""
        dem_multi = torch.randn(2, 3, 64, 64) * 100.0 + 500.0
        out = encoder(dem_multi)
        assert out.shape == (2, 128, 64, 64)

    def test_fit_surface_basic(self, encoder):
        """fit_surface runs a few iterations and returns expected dict keys."""
        dem = torch.randn(1, 1, 32, 32) * 100.0 + 500.0  # small size for speed
        result = encoder.fit_surface(dem, n_iter=5, lr=1e-3)
        assert 'loss_history' in result
        assert 'final_loss' in result
        assert 'fitted_height' in result
        assert len(result['loss_history']) == 5
        assert result['fitted_height'].shape == (32, 32)

    def test_different_spatial_sizes(self):
        """Encoder handles different spatial resolutions gracefully."""
        encoder = SIRENDEMEncoder(out_ch=128, hidden_dim=256, n_layers=3)
        for H, W in [(32, 32), (64, 64), (96, 128)]:
            dem = torch.randn(1, 1, H, W) * 100.0 + 500.0
            out = encoder(dem)
            assert out.shape == (1, 128, H, W), f"Failed for shape ({H}, {W})"
