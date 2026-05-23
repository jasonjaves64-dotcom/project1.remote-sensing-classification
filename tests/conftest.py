"""
测试配置文件
"""
import pytest
import numpy as np
import torch

@pytest.fixture
def sample_opt_data():
    """生成示例光学时序数据"""
    return np.random.rand(12, 10, 256, 256).astype(np.float32)

@pytest.fixture
def sample_sar_data():
    """生成示例SAR时序数据"""
    return np.random.rand(12, 3, 256, 256).astype(np.float32)

@pytest.fixture
def sample_doy():
    """生成示例DOY数据"""
    return np.random.rand(12).astype(np.float32)

@pytest.fixture
def sample_label():
    """生成示例标签数据"""
    return np.random.randint(0, 7, (256, 256)).astype(np.uint8)

@pytest.fixture
def device():
    """获取测试设备"""
    return torch.device("cpu")