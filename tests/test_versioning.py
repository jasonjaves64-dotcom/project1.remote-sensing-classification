"""
数据版本管理模块测试
"""
import pytest
import os
import tempfile
import numpy as np
from utils.versioning import DataVersionManager, ExperimentTracker

class TestDataVersionManager:
    """测试数据版本管理器"""
    
    def test_version_manager_creation(self):
        """测试版本管理器创建"""
        manager = DataVersionManager()
        
        assert manager is not None
    
    def test_generate_data_hash(self):
        """测试生成数据哈希值"""
        manager = DataVersionManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            file1 = os.path.join(tmpdir, "test1.txt")
            file2 = os.path.join(tmpdir, "test2.txt")
            
            with open(file1, "w") as f:
                f.write("test content")
            with open(file2, "w") as f:
                f.write("another content")
            
            hash_val = manager.generate_data_hash([file1, file2])
            
            assert isinstance(hash_val, str)
            assert len(hash_val) == 8  # 取前8位
    
    def test_data_hash_consistency(self):
        """测试数据哈希一致性"""
        manager = DataVersionManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test.txt")
            with open(file1, "w") as f:
                f.write("same content")
            
            hash1 = manager.generate_data_hash([file1])
            hash2 = manager.generate_data_hash([file1])
            
            assert hash1 == hash2
    
    def test_data_hash_different(self):
        """测试不同数据产生不同哈希"""
        manager = DataVersionManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "test1.txt")
            file2 = os.path.join(tmpdir, "test2.txt")
            
            with open(file1, "w") as f:
                f.write("content1")
            with open(file2, "w") as f:
                f.write("content2")
            
            hash1 = manager.generate_data_hash([file1])
            hash2 = manager.generate_data_hash([file2])
            
            assert hash1 != hash2
    
    def test_get_version_info(self):
        """测试获取版本信息"""
        manager = DataVersionManager()
        
        version_info = manager.get_version_info()
        
        assert "python_version" in version_info
        assert "torch_version" in version_info
        assert isinstance(version_info, dict)

class TestExperimentTracker:
    """测试实验追踪器"""
    
    def test_tracker_creation(self):
        """测试追踪器创建"""
        tracker = ExperimentTracker("test_exp")
        
        assert tracker is not None
        assert tracker.exp_name == "test_exp"
    
    def test_log_experiment(self):
        """测试记录实验信息"""
        tracker = ExperimentTracker("test_exp")
        
        config = {
            "batch_size": 8,
            "epochs": 20,
            "lr": 0.001
        }
        
        metrics = {
            "mIoU": 0.85,
            "OA": 0.90,
            "loss": 0.25
        }
        
        try:
            tracker.log_experiment(config, metrics)
        except Exception as e:
            pytest.fail(f"记录实验失败: {e}")
    
    def test_generate_exp_id(self):
        """测试生成实验ID"""
        tracker = ExperimentTracker("test_exp")
        
        exp_id = tracker.generate_exp_id()
        
        assert isinstance(exp_id, str)
        assert len(exp_id) > 0
    
    def test_save_results(self):
        """测试保存实验结果"""
        tracker = ExperimentTracker("test_save")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {
                "train_loss": [0.5, 0.4, 0.3],
                "val_miou": [0.7, 0.75, 0.8]
            }
            
            try:
                tracker.save_results(results, tmpdir)
                assert os.path.exists(os.path.join(tmpdir, "results.json"))
            except Exception as e:
                pytest.fail(f"保存结果失败: {e}")