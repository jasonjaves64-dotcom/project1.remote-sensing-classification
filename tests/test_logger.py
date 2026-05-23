"""
日志模块测试
"""
import pytest
import os
import tempfile
from utils.logger import get_logger, TrainingLogger, InferenceLogger

class TestGetLogger:
    """测试基础日志函数"""
    
    def test_logger_creation(self):
        """测试日志记录器创建"""
        logger = get_logger("test_module")
        
        assert logger is not None
        assert logger.name == "test_module"
    
    def test_logger_level(self):
        """测试日志级别设置"""
        logger = get_logger("test_debug", log_level="DEBUG")
        
        assert logger.level == 10  # DEBUG level
    
    def test_logger_singleton(self):
        """测试日志记录器单例特性"""
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        
        assert logger1 is logger2
    
    def test_log_output(self):
        """测试日志输出"""
        logger = get_logger("output_test")
        
        try:
            logger.info("Test log message")
            logger.warning("Test warning message")
            logger.error("Test error message")
        except Exception as e:
            pytest.fail(f"日志输出失败: {e}")

class TestTrainingLogger:
    """测试训练日志记录器"""
    
    def test_logger_creation(self):
        """测试训练日志记录器创建"""
        logger = TrainingLogger()
        
        assert logger is not None
        assert logger.log_file == "logs/training.log"
    
    def test_log_epoch(self):
        """测试记录训练轮次"""
        logger = TrainingLogger()
        
        try:
            logger.log_epoch(epoch=1, phase=1, loss=0.5, miou=0.75, oa=0.85)
            logger.log_epoch(epoch=2, phase=1, loss=0.4, miou=0.78, oa=0.87,
                           iou_per_class=[0.7, 0.8, 0.6, 0.9, 0.75, 0.85])
        except Exception as e:
            pytest.fail(f"记录训练轮次失败: {e}")
    
    def test_log_best_model(self):
        """测试记录最优模型"""
        logger = TrainingLogger()
        
        try:
            logger.log_best_model(epoch=10, miou=0.85, oa=0.90)
        except Exception as e:
            pytest.fail(f"记录最优模型失败: {e}")
    
    def test_log_experiment(self):
        """测试记录实验开始和结束"""
        logger = TrainingLogger()
        
        try:
            logger.log_experiment_start("test_exp", {"batch_size": 8, "epochs": 20})
            logger.log_experiment_end(best_miou=0.85, best_oa=0.90)
        except Exception as e:
            pytest.fail(f"记录实验失败: {e}")

class TestInferenceLogger:
    """测试推理日志记录器"""
    
    def test_logger_creation(self):
        """测试推理日志记录器创建"""
        logger = InferenceLogger()
        
        assert logger is not None
    
    def test_log_inference(self):
        """测试记录推理过程"""
        logger = InferenceLogger()
        
        try:
            logger.log_inference_start((12, 10, 256, 256))
            logger.log_inference_end((256, 256), duration=15.5)
        except Exception as e:
            pytest.fail(f"记录推理过程失败: {e}")
    
    def test_log_metrics(self):
        """测试记录推理指标"""
        logger = InferenceLogger()
        
        try:
            metrics = {"mIoU": 0.85, "OA": 0.90, "Kappa": 0.88}
            logger.log_metrics(metrics)
        except Exception as e:
            pytest.fail(f"记录推理指标失败: {e}")