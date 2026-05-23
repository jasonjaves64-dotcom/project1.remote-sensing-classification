"""
多模态融合模块
"""
import numpy as np


class MultiModalFusion:
    """多源数据融合（光学 + SAR + DEM）"""

    @staticmethod
    def fuse_features(opt_seq: np.ndarray, sar_seq: np.ndarray,
                      dem_features: np.ndarray,
                      fusion_mode: str = "concat") -> np.ndarray:
        T_opt, C_opt, H, W = opt_seq.shape
        C_sar = sar_seq.shape[1]
        C_dem = dem_features.shape[0]

        if T_opt != sar_seq.shape[0]:
            raise ValueError("光学和SAR时序长度不一致")

        if fusion_mode == "concat":
            dem_repeated = np.repeat(dem_features[np.newaxis, ...], T_opt, axis=0)
            return np.concatenate([opt_seq, sar_seq, dem_repeated], axis=1)
        elif fusion_mode == "additive":
            dem_expanded = np.repeat(dem_features[np.newaxis, ...], T_opt, axis=0)
            dem_scaled = dem_expanded * 0.1
            return opt_seq + sar_seq[:, :C_opt] + dem_scaled[:, :C_opt]
        elif fusion_mode == "attention":
            dem_repeated = np.repeat(dem_features[np.newaxis, ...], T_opt, axis=0)
            attention_weights = np.mean(dem_repeated, axis=1, keepdims=True)
            return opt_seq * (1 + attention_weights) + sar_seq * (1 - attention_weights)
        else:
            return np.concatenate([opt_seq, sar_seq], axis=1)
