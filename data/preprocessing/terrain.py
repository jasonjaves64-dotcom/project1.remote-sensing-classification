"""
地形辐射校正模块
"""
import numpy as np
import rasterio


class TerrainCorrector:
    """地形辐射校正（Minnaert / C-correction / Cosine）"""

    def __init__(self, dem_path: str, solar_zenith_deg: float = 30.0,
                 solar_azimuth_deg: float = 150.0):
        self.sza_rad = np.radians(solar_zenith_deg)
        self.saa_rad = np.radians(solar_azimuth_deg)
        self._load_terrain(dem_path)

    def _load_terrain(self, dem_path: str):
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            res = src.res[0]
            dy, dx = np.gradient(dem, res)
            self.slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
            self.aspect_rad = np.arctan2(-dx, dy) % (2 * np.pi)
            self.cos_i = (
                np.cos(self.sza_rad) * np.cos(self.slope_rad) +
                np.sin(self.sza_rad) * np.sin(self.slope_rad) *
                np.cos(self.saa_rad - self.aspect_rad)
            )
            self.cos_i = np.clip(self.cos_i, 0.01, 1.0)
        self.cos_theta = np.cos(self.sza_rad)

    def correct(self, bands: np.ndarray, method: str = "minnaert") -> np.ndarray:
        corrected = np.zeros_like(bands)
        for c in range(bands.shape[0]):
            band = bands[c]
            valid = ~np.isnan(band)
            if method == "cos":
                corrected[c] = band * (self.cos_theta / self.cos_i)
            elif method == "c":
                if valid.sum() < 100:
                    corrected[c] = band; continue
                x, y = self.cos_i[valid].flatten(), band[valid].flatten()
                m, b = np.polyfit(x, y, 1)
                c_coef = b / (m + 1e-6)
                corrected[c] = band * (self.cos_theta + c_coef) / (self.cos_i + c_coef)
            elif method == "minnaert":
                if valid.sum() < 100:
                    corrected[c] = band; continue
                log_r = np.log(band[valid].flatten() + 1e-6)
                log_ci = np.log(self.cos_i[valid].flatten())
                k, _ = np.polyfit(log_ci, log_r - np.log(self.cos_theta), 1)
                k = np.clip(k, 0, 1)
                corrected[c] = band * (self.cos_theta / self.cos_i) ** k
            corrected[c][~valid] = np.nan
            corrected[c] = np.clip(corrected[c], 0, 1)
        return corrected
