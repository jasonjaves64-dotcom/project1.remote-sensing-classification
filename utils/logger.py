import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

def get_logger(name: str = __name__, log_level: str = "INFO") -> logging.Logger:
    """
    创建并返回一个配置好的日志记录器
    
    Args:
        name: 日志记录器名称，默认为当前模块名
        log_level: 日志级别，可选值: DEBUG, INFO, WARNING, ERROR
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    logger.setLevel(log_levels.get(log_level.upper(), logging.INFO))
    logger.propagate = False
    
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = RotatingFileHandler(
        "logs/project.log",
        maxBytes=1024 * 1024 * 5,  
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_levels.get(log_level.upper(), logging.INFO))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TrainingLogger:
    """
    训练过程专用日志记录器
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = get_logger("training")
        self.log_file = log_file or "logs/training.log"
        
        self.training_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=1024 * 1024 * 10,
            backupCount=10,
            encoding="utf-8"
        )
        self.training_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.training_handler.setLevel(logging.INFO)
        
        self.logger.addHandler(self.training_handler)
    
    def log_epoch(self, epoch: int, phase: int, loss: float, miou: float, oa: float,
                  iou_per_class: Optional[list] = None):
        """
        记录训练轮次信息
        
        Args:
            epoch: 当前轮次
            phase: 训练阶段 (1或2)
            loss: 损失值
            miou: 平均交并比
            oa: 总体精度
            iou_per_class: 每个类别的IoU
        """
        msg = f"[P{phase}] Epoch {epoch:3d} | Loss: {loss:.4f} | mIoU: {miou:.4f} | OA: {oa:.4f}"
        if iou_per_class:
            iou_str = " | ".join([f"{i+1}:{v:.3f}" for i, v in enumerate(iou_per_class)])
            msg += f" | IoU: {iou_str}"
        self.logger.info(msg)
    
    def log_best_model(self, epoch: int, miou: float, oa: float):
        """记录最优模型信息"""
        self.logger.info(f"✓ 发现最优模型: Epoch {epoch} | mIoU: {miou:.4f} | OA: {oa:.4f}")
    
    def log_experiment_start(self, exp_name: str, config: dict):
        """记录实验开始"""
        self.logger.info(f"="*60)
        self.logger.info(f"开始实验: {exp_name}")
        self.logger.info(f"配置参数: {config}")
        self.logger.info("="*60)
    
    def log_experiment_end(self, best_miou: float, best_oa: float):
        """记录实验结束"""
        self.logger.info(f"="*60)
        self.logger.info(f"实验完成 | 最佳mIoU: {best_miou:.4f} | 最佳OA: {best_oa:.4f}")
        self.logger.info("="*60)
    
    def log_evaluation(self, metrics: dict):
        """记录评估结果"""
        self.logger.info(f"评估结果: {metrics}")

class InferenceLogger:
    """
    推理过程专用日志记录器
    """
    
    def __init__(self):
        self.logger = get_logger("inference")
    
    def log_inference_start(self, input_shape: tuple):
        """记录推理开始"""
        self.logger.info(f"开始推理 | 输入尺寸: {input_shape}")
    
    def log_inference_end(self, output_shape: tuple, duration: float):
        """记录推理结束"""
        self.logger.info(f"推理完成 | 输出尺寸: {output_shape} | 耗时: {duration:.2f}s")
    
    def log_metrics(self, metrics: dict):
        """记录推理指标"""
        self.logger.info(f"推理指标: {metrics}")