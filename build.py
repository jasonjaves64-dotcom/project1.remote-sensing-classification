"""
构建脚本 - 支持模型导出和打包
"""

import subprocess
import sys
import os
import argparse

def export_model(model_path: str, output_dir: str = "exports"):
    """导出模型为多种格式"""
    print(f"导出模型: {model_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出为ONNX格式
    print("导出为ONNX格式...")
    try:
        subprocess.run([
            sys.executable, "-m", "scripts.export_edl",
            "--model_path", model_path,
            "--output_dir", output_dir,
            "--format", "onnx"
        ], check=True)
        print("ONNX导出成功！")
    except subprocess.CalledProcessError as e:
        print(f"ONNX导出失败: {e}")
    
    # 导出为EMD格式
    print("导出为EMD格式...")
    try:
        subprocess.run([
            sys.executable, "-m", "scripts.export_edl",
            "--model_path", model_path,
            "--output_dir", output_dir,
            "--format", "emd"
        ], check=True)
        print("EMD导出成功！")
    except subprocess.CalledProcessError as e:
        print(f"EMD导出失败: {e}")
    
    # 导出为DLPK格式
    print("导出为DLPK格式...")
    try:
        subprocess.run([
            sys.executable, "-m", "scripts.export_to_arcgis",
            "--model_path", model_path,
            "--output_dir", output_dir
        ], check=True)
        print("DLPK导出成功！")
    except subprocess.CalledProcessError as e:
        print(f"DLPK导出失败: {e}")
    
    print(f"模型已导出到: {output_dir}")

def build_docker():
    """构建Docker镜像"""
    print("构建Docker镜像...")
    try:
        subprocess.run(["docker", "build", "-t", "crop-classification:latest", "."], check=True)
        print("Docker镜像构建成功！")
    except subprocess.CalledProcessError as e:
        print(f"Docker镜像构建失败: {e}")

def build_desktop_app():
    """构建桌面应用"""
    print("构建桌面应用...")
    spec_file = "作物分类系统_完整版.spec" if os.path.exists("作物分类系统_完整版.spec") else "作物分类系统.spec"
    try:
        subprocess.run([sys.executable, "-m", "PyInstaller", spec_file, "--noconfirm"], check=True)
        print("桌面应用构建成功！")
    except subprocess.CalledProcessError as e:
        print(f"桌面应用构建失败: {e}")
        print("提示: 使用 python build_exe.py 进行完整打包")

def run_tests():
    """运行测试套件"""
    print("运行测试套件...")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
        print("测试完成！")
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")

def lint_code():
    """代码检查"""
    print("代码检查...")
    try:
        subprocess.run([sys.executable, "-m", "flake8", "models/", "scripts/", "api/", "utils/", "--max-line-length=120"], check=True)
        print("代码检查完成！")
    except subprocess.CalledProcessError as e:
        print(f"代码检查失败: {e}")

def type_check():
    """类型检查"""
    print("类型检查...")
    try:
        subprocess.run([sys.executable, "-m", "mypy", "models/", "scripts/", "api/", "utils/"], check=True)
        print("类型检查完成！")
    except subprocess.CalledProcessError as e:
        print(f"类型检查失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建脚本")
    parser.add_argument("--export", type=str, help="导出模型（指定模型路径）")
    parser.add_argument("--docker", action="store_true", help="构建Docker镜像")
    parser.add_argument("--desktop", action="store_true", help="构建桌面应用")
    parser.add_argument("--tests", action="store_true", help="运行测试套件")
    parser.add_argument("--lint", action="store_true", help="代码检查")
    parser.add_argument("--typecheck", action="store_true", help="类型检查")
    parser.add_argument("--all", action="store_true", help="执行所有构建步骤")
    
    args = parser.parse_args()
    
    if args.all:
        lint_code()
        type_check()
        run_tests()
        build_docker()
        build_desktop_app()
    elif args.export:
        export_model(args.export)
    elif args.docker:
        build_docker()
    elif args.desktop:
        build_desktop_app()
    elif args.tests:
        run_tests()
    elif args.lint:
        lint_code()
    elif args.typecheck:
        type_check()
    else:
        print("请指定要执行的构建操作:")
        print("  --export <model_path>  导出模型")
        print("  --docker              构建Docker镜像")
        print("  --desktop             构建桌面应用")
        print("  --tests               运行测试套件")
        print("  --lint                代码检查")
        print("  --typecheck           类型检查")
        print("  --all                 执行所有构建步骤")
