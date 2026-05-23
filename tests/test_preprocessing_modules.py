"""
模块化重构测试
"""
import pytest
import numpy as np
import torch
from data.preprocessing.base import BasePreprocessor
from data.preprocessing.optical import OpticalProcessor
from data.preprocessing.sar import SARProcessor
from data.preprocessing.quality import QualityControl
from data.datasets.crop_dataset import CropDataset

class TestPreprocessor:
    """测试预处理模块"""
    
    def test_optical_processor_creation(self):
        """测试光学处理器创建"""
        processor = OpticalProcessor()
        
        assert processor is not None
    
    def test_ndvi_calculation(self):
        """测试NDVI计算"""
        processor = OpticalProcessor()
        
        nir = np.random.rand(100, 100) * 0.5 + 0.2
        red = np.random.rand(100, 100) * 0.3 + 0.1
        
        ndvi = processor.calculate_ndvi(nir, red)
        
        assert ndvi.shape == (100, 100)
        assert np.all(ndvi >= -1) and np.all(ndvi <= 1)
    
    def test_evi_calculation(self):
        """测试EVI计算"""
        processor = OpticalProcessor()
        
        nir = np.random.rand(100, 100) * 0.5 + 0.2
        red = np.random.rand(100, 100) * 0.3 + 0.1
        blue = np.random.rand(100, 100) * 0.4 + 0.1
        
        evi = processor.calculate_evi(nir, red, blue)
        
        assert evi.shape == (100, 100)
    
    def test_cloud_masking(self):
        """测试云掩膜"""
        processor = OpticalProcessor()
        
        qa_pixel = np.zeros((100, 100), dtype=np.uint16)
        qa_pixel[20:30, 20:30] = 1 << 10  # 设置云标志
        
        mask = processor.apply_cloud_mask(qa_pixel)
        
        assert mask.shape == (100, 100)
        assert np.sum(mask[20:30, 20:30]) == 0  # 云区域被标记为False
    
    def test_sar_processor_creation(self):
        """测试SAR处理器创建"""
        processor = SARProcessor()
        
        assert processor is not None
    
    def test_speckle_filtering(self):
        """测试斑点噪声滤波"""
        processor = SARProcessor()
        
        sar_data = np.random.rand(100, 100) * 2 - 1  # 模拟SAR数据
        
        filtered = processor.apply_speckle_filter(sar_data, window_size=5)
        
        assert filtered.shape == (100, 100)
    
    def test_sar_orthorectification(self):
        """测试SAR正射校正"""
        processor = SARProcessor()
        
        sar_data = np.random.rand(100, 100) * 2 - 1
        
        result = processor.orthorectify(sar_data)
        
        assert result.shape == (100, 100)

class TestQualityControl:
    """测试质量控制模块"""
    
    def test_qc_creation(self):
        """测试质量控制创建"""
        qc = QualityControl()
        
        assert qc is not None
    
    def test_validity_check(self):
        """测试数据有效性检查"""
        qc = QualityControl()
        
        data = np.random.rand(10, 100, 100)
        
        report = qc.check_validity(data)
        
        assert isinstance(report, dict)
        assert "valid_pixels" in report
        assert "invalid_pixels" in report
    
    def test_statistics_report(self):
        """测试统计报告生成"""
        qc = QualityControl()
        
        data = np.random.rand(10, 100, 100)
        
        stats = qc.generate_statistics(data)
        
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats

class TestCropDataset:
    """测试作物数据集"""
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        dataset = CropDataset(
            opt_paths=[],
            sar_paths=[],
            label_paths=[],
            transform=None
        )
        
        assert dataset is not None
    
    def test_dataset_properties(self):
        """测试数据集属性"""
        dataset = CropDataset(
            opt_paths=[],
            sar_paths=[],
            label_paths=[],
            transform=None
        )
        
        assert hasattr(dataset, 'opt_paths')
        assert hasattr(dataset, 'sar_paths')
        assert hasattr(dataset, 'label_paths')