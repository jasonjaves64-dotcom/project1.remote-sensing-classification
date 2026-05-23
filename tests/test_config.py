"""
配置模块测试
"""
import pytest
import tempfile
import os
from utils.config import Config, DataConfig, TrainingConfig, PreprocessingConfig

class TestDataConfig:
    """测试数据配置类"""
    
    def test_default_values(self):
        """测试默认配置值"""
        config = DataConfig()
        
        assert config.raw_dir == "./data/raw"
        assert config.processed_dir == "./data/processed"
        assert config.output_dir == "./output"
        assert config.checkpoint_dir == "./checkpoints"
    
    def test_custom_values(self):
        """测试自定义配置值"""
        config = DataConfig(
            raw_dir="/custom/raw",
            processed_dir="/custom/processed",
            output_dir="/custom/output"
        )
        
        assert config.raw_dir == "/custom/raw"
        assert config.processed_dir == "/custom/processed"
        assert config.output_dir == "/custom/output"

class TestTrainingConfig:
    """测试训练配置类"""
    
    def test_default_values(self):
        """测试默认配置值"""
        config = TrainingConfig()
        
        assert config.batch_size == 8
        assert config.patch_size == 32
        assert config.epochs_p1 == 20
        assert config.epochs_p2 == 60
        assert config.lr_p1 == 0.001
        assert config.lr_p2 == 0.0003
        assert config.use_spatial_split == False
    
    def test_custom_values(self):
        """测试自定义配置值"""
        config = TrainingConfig(
            batch_size=16,
            epochs_p1=30,
            lr_p1=0.0005,
            use_spatial_split=True
        )
        
        assert config.batch_size == 16
        assert config.epochs_p1 == 30
        assert config.lr_p1 == 0.0005
        assert config.use_spatial_split == True

class TestPreprocessingConfig:
    """测试预处理配置类"""
    
    def test_default_values(self):
        """测试默认配置值"""
        config = PreprocessingConfig()
        
        assert config.apply_terrain_correction == False
        assert config.terrain_correction_method == "minnaert"
        assert config.purity_threshold == 0.8
        assert config.split_block_size == 64
        assert config.spatial_split_kfold == 5

class TestConfig:
    """测试主配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        
        assert config.training.batch_size == 8
        assert config.preprocessing.purity_threshold == 0.8
    
    def test_from_yaml_default(self):
        """测试从不存在的YAML文件加载（应返回默认配置）"""
        config = Config.from_yaml("non_existent.yaml")
        
        assert config.training.batch_size == 8
        assert config.preprocessing.purity_threshold == 0.8
    
    def test_to_and_from_yaml(self):
        """测试配置保存和加载"""
        config = Config()
        config.training.batch_size = 16
        config.preprocessing.purity_threshold = 0.7
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_yaml(temp_path)
            
            loaded_config = Config.from_yaml(temp_path)
            
            assert loaded_config.training.batch_size == 16
            assert loaded_config.preprocessing.purity_threshold == 0.7
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)