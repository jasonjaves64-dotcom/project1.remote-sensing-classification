"""
作物分类数据集
"""
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Callable

class CropDataset(Dataset):
    """
    作物分类数据集类
    
    支持加载光学影像、SAR影像和标签数据
    """
    
    def __init__(
        self,
        opt_paths: List[str],
        sar_paths: List[str],
        label_paths: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Args:
            opt_paths: 光学影像文件路径列表
            sar_paths: SAR影像文件路径列表
            label_paths: 标签文件路径列表
            transform: 数据变换函数
        """
        self.opt_paths = opt_paths
        self.sar_paths = sar_paths
        self.label_paths = label_paths
        self.transform = transform
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """
        验证输入路径的一致性
        """
        if len(self.opt_paths) != len(self.sar_paths):
            raise ValueError("光学影像和SAR影像数量不一致")
        
        if len(self.opt_paths) != len(self.label_paths):
            raise ValueError("影像和标签数量不一致")
    
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            数据样本数量
        """
        return len(self.opt_paths)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
        
        Returns:
            数据样本字典
        """
        opt_data = self._load_optical(idx)
        sar_data = self._load_sar(idx)
        label = self._load_label(idx)
        
        sample = {
            "opt": opt_data,
            "sar": sar_data,
            "label": label
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _load_optical(self, idx: int) -> np.ndarray:
        """
        加载光学影像数据
        
        Args:
            idx: 样本索引
        
        Returns:
            光学影像数据
        """
        path = self.opt_paths[idx]
        return np.load(path) if path.endswith('.npy') else self._load_geotiff(path)
    
    def _load_sar(self, idx: int) -> np.ndarray:
        """
        加载SAR影像数据
        
        Args:
            idx: 样本索引
        
        Returns:
            SAR影像数据
        """
        path = self.sar_paths[idx]
        return np.load(path) if path.endswith('.npy') else self._load_geotiff(path)
    
    def _load_label(self, idx: int) -> np.ndarray:
        """
        加载标签数据
        
        Args:
            idx: 样本索引
        
        Returns:
            标签数据
        """
        path = self.label_paths[idx]
        return np.load(path) if path.endswith('.npy') else self._load_geotiff(path)
    
    def _load_geotiff(self, path: str) -> np.ndarray:
        """
        加载GeoTIFF文件（占位实现）
        
        Args:
            path: 文件路径
        
        Returns:
            影像数据
        """
        import rasterio
        with rasterio.open(path) as src:
            return src.read()
    
    def get_statistics(self) -> dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "num_samples": len(self),
            "num_opt_paths": len(self.opt_paths),
            "num_sar_paths": len(self.sar_paths),
            "num_label_paths": len(self.label_paths)
        }