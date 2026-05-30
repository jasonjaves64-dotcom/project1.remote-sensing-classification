"""
安装脚本 - 支持EDL模式依赖
"""

import subprocess
import sys

def install_dependencies():
    """安装所有依赖"""
    print("正在安装依赖...")
    
    # 核心依赖
    core_deps = [
        "numpy>=1.24.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.3.0",
        "timm>=0.9.0",
        "scikit-image>=0.21.0",
        "rasterio>=1.3.9",
        "earthengine-api>=0.1.355",
        "geemap>=0.22.0",
        "pillow>=9.0.0",
        "mysql-connector-python>=8.0.30",
        "fastapi>=0.109.0",
        "uvicorn>=0.23.0",
        "streamlit>=1.30.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "ray>=2.6.0",
        "PyQt5>=5.15.0",
        "gdal>=3.6.0",
        "pandas>=2.1.0",
        "xarray>=2.0.0",
        "seaborn>=0.12.0",
        "mypy>=1.5.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        # rate limiting moved to built-in middleware
        "email-validator>=2.0.0",
        "requests>=2.31.0"
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *core_deps])
        print("依赖安装成功！")
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")
        sys.exit(1)

def install_optional_deps():
    """安装可选依赖"""
    optional_deps = [
        "mlflow>=2.8.0",
        "optuna>=3.4.0",
        "tensorboard>=2.15.0"
    ]
    
    print("正在安装可选依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *optional_deps])
        print("可选依赖安装成功！")
    except subprocess.CalledProcessError as e:
        print(f"可选依赖安装失败: {e}")

def setup_gdal():
    """设置GDAL环境"""
    print("配置GDAL环境...")
    try:
        import rasterio
        print(f"GDAL版本: {rasterio.__version__}")
        print("GDAL配置成功！")
    except ImportError:
        print("警告: GDAL未安装，部分功能可能受限")

if __name__ == "__main__":
    print("=" * 50)
    print("遥感影像作物分类系统 - 依赖安装")
    print("=" * 50)
    
    install_dependencies()
    install_optional_deps()
    setup_gdal()
    
    print("\n" + "=" * 50)
    print("安装完成！")
    print("=" * 50)
