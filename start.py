"""
启动脚本 - 支持启动Web应用、API服务等
"""

import subprocess
import sys
import os
import argparse
import threading
import time

def start_web_app():
    """启动Streamlit Web应用"""
    print("启动Web应用...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("Web应用已停止")

def start_api_service():
    """启动FastAPI服务"""
    print("启动API服务...")
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    except KeyboardInterrupt:
        print("API服务已停止")

def start_jupyter():
    """启动Jupyter Notebook"""
    print("启动Jupyter Notebook...")
    try:
        subprocess.run([sys.executable, "-m", "jupyter", "notebook", "--port=8888", "--no-browser"])
    except KeyboardInterrupt:
        print("Jupyter Notebook已停止")

def start_tensorboard():
    """启动TensorBoard"""
    print("启动TensorBoard...")
    try:
        subprocess.run([sys.executable, "-m", "tensorboard", "--logdir=logs", "--port=6006"])
    except KeyboardInterrupt:
        print("TensorBoard已停止")

def start_all():
    """启动所有服务（并行）"""
    print("启动所有服务...")
    
    # 启动Web应用
    web_thread = threading.Thread(target=start_web_app)
    web_thread.daemon = True
    web_thread.start()
    
    # 启动API服务
    api_thread = threading.Thread(target=start_api_service)
    api_thread.daemon = True
    api_thread.start()
    
    print("所有服务已启动！")
    print("Web应用: http://localhost:8501")
    print("API服务: http://localhost:8000")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("所有服务已停止")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动遥感影像作物分类系统服务")
    parser.add_argument("--web", action="store_true", help="启动Web应用")
    parser.add_argument("--api", action="store_true", help="启动API服务")
    parser.add_argument("--jupyter", action="store_true", help="启动Jupyter Notebook")
    parser.add_argument("--tensorboard", action="store_true", help="启动TensorBoard")
    parser.add_argument("--all", action="store_true", help="启动所有服务")
    
    args = parser.parse_args()
    
    if args.all:
        start_all()
    elif args.web:
        start_web_app()
    elif args.api:
        start_api_service()
    elif args.jupyter:
        start_jupyter()
    elif args.tensorboard:
        start_tensorboard()
    else:
        print("请指定要启动的服务:")
        print("  --web       启动Web应用")
        print("  --api       启动API服务")
        print("  --jupyter   启动Jupyter Notebook")
        print("  --tensorboard  启动TensorBoard")
        print("  --all       启动所有服务")
