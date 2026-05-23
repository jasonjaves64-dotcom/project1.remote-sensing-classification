"""
数据版本管理模块
"""
import hashlib
import os
import json
import time
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional

class DataVersionManager:
    """
    数据版本管理器
    用于追踪数据版本和实验复现
    """
    
    def __init__(self):
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """获取日志记录器"""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def generate_data_hash(self, file_paths: List[str]) -> str:
        """
        生成数据哈希值用于版本追踪
        
        Args:
            file_paths: 文件路径列表
        
        Returns:
            8位哈希值字符串
        """
        hash_obj = hashlib.md5()
        for path in sorted(file_paths):
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    hash_obj.update(f.read())
            else:
                self.logger.warning(f"文件不存在: {path}")
        
        return hash_obj.hexdigest()[:8]
    
    def get_version_info(self) -> Dict[str, Any]:
        """
        获取当前环境版本信息
        
        Returns:
            版本信息字典
        """
        import torch
        import numpy as np
        
        return {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "os": platform.system(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_version_info(self, output_path: str):
        """
        保存版本信息到文件
        
        Args:
            output_path: 输出文件路径
        """
        version_info = self.get_version_info()
        
        with open(output_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        self.logger.info(f"版本信息已保存到: {output_path}")

class ExperimentTracker:
    """
    实验追踪器
    用于记录实验配置和结果
    """
    
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.exp_id = self.generate_exp_id()
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """获取日志记录器"""
        import logging
        logger = logging.getLogger(f"experiment.{self.exp_name}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def generate_exp_id(self) -> str:
        """
        生成唯一的实验ID
        
        Returns:
            实验ID字符串
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"{self.exp_name}_{timestamp}_{random_suffix}"
    
    def log_experiment(self, config: Dict[str, Any], metrics: Dict[str, Any]):
        """
        记录实验信息
        
        Args:
            config: 实验配置
            metrics: 实验指标
        """
        self.logger.info(f"实验ID: {self.exp_id}")
        self.logger.info(f"配置: {json.dumps(config, indent=2)}")
        self.logger.info(f"指标: {json.dumps(metrics, indent=2)}")
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """
        保存实验结果
        
        Args:
            results: 实验结果字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"实验结果已保存到: {results_path}")
    
    def create_experiment_dir(self, base_dir: str = "experiments") -> str:
        """
        创建实验目录
        
        Args:
            base_dir: 基础目录
        
        Returns:
            实验目录路径
        """
        exp_dir = os.path.join(base_dir, self.exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        self.logger.info(f"实验目录已创建: {exp_dir}")
        return exp_dir