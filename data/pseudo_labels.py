"""Pseudo-label generation from existing data — no human annotation needed.

References:
  V6-剩余瓶颈-方案评审.md Section 4.2
  V6-多尺度瓶颈-方案评审.md
"""
import torch
import torch.nn.functional as F

# Module-level Sobel kernels — created once, reused across calls
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)


def _get_sobel_kernels(device, dtype=torch.float32):
    """Return Sobel kernels moved to the correct device."""
    return (_SOBEL_X.to(device=device, dtype=dtype),
            _SOBEL_Y.to(device=device, dtype=dtype))


def generate_lai_pseudo(ndvi: torch.Tensor) -> torch.Tensor:
    """Estimate LAI from NDVI using empirical formula.

    LAI ≈ -ln((NDVI_max - NDVI) / (NDVI_max - NDVI_min)) / k

    Args:
        ndvi: (B,) or (B, N) — NDVI values in [0, 1]
    Returns:
        lai: same shape, clipped to [0, 7]
    """
    ndvi_max, ndvi_min, k = 0.9, 0.1, 0.5
    ndvi_clipped = torch.clamp(ndvi, ndvi_min + 0.01, ndvi_max - 0.01)
    lai = -torch.log((ndvi_max - ndvi_clipped) / (ndvi_max - ndvi_min)) / k
    return torch.clamp(lai, 0.0, 7.0)


def generate_growth_stage_pseudo(doy: torch.Tensor, dem: torch.Tensor = None,
                                  num_stages: int = 5) -> torch.Tensor:
    """Estimate growth stage from DOY and elevation.

    Uses growing degree day (GDD) approximation:
    - Base temperature Tbase = 10 C
    - GDD approx mean_temp * days_since_planting
    - Elevation correction: -0.6 C per 100m

    Simplified: discretize DOY into num_stages bins, adjusted by elevation.

    Args:
        doy: (B, T) — day of year [0, 1] normalized
        dem: (B, 1, H, W) or None — elevation in meters
        num_stages: number of growth stages (default 5)
    Returns:
        stage: (B*T,) long tensor — stage index [0, num_stages-1]
    """
    # Simplified: DOY range [0,1] -> 5 stages roughly spanning growing season
    # Stage 0: emergence (0-0.15), 1: vegetative (0.15-0.4),
    # 2: reproductive (0.4-0.6), 3: grain fill (0.6-0.8), 4: maturity (0.8-1.0)
    boundaries = torch.tensor([0.0, 0.15, 0.40, 0.60, 0.80, 1.0])
    doy_flat = doy.flatten()
    stage = torch.bucketize(doy_flat, boundaries[1:-1])  # returns [0, num_stages-1]
    return stage.clamp(0, num_stages - 1)


def generate_boundary_pseudo(dem: torch.Tensor, sar: torch.Tensor = None,
                              threshold: float = 0.3) -> torch.Tensor:
    """Generate field boundary pseudo-labels from DEM slope.

    Uses Sobel gradient magnitude on DEM -> threshold -> binary boundary mask.

    Args:
        dem: (B, 5, H, W) or (B, 1, H, W) — DEM features (first channel = elevation)
        sar: optional SAR VV for edge complement
        threshold: gradient magnitude threshold
    Returns:
        boundary: (B, 1, H, W) — binary boundary mask
    """
    # Use first DEM channel (elevation) or mean across channels
    if dem.shape[1] >= 1:
        elev = dem[:, 0:1]  # (B, 1, H, W)
    else:
        elev = dem.mean(dim=1, keepdim=True)

    # Sobel gradient magnitude (kernels cached at module level)
    sobel_x, sobel_y = _get_sobel_kernels(dem.device)
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    grad_x = F.conv2d(elev, sobel_x, padding=1)
    grad_y = F.conv2d(elev, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize and threshold
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    boundary = (grad_mag > threshold).float()

    return boundary
