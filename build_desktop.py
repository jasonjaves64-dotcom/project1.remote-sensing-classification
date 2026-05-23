"""
build_desktop.py — 旧版打包脚本 (仅打包桌面GUI)
请使用 build_exe.py 进行完整打包。
"""
import subprocess
import sys

if __name__ == "__main__":
    print("提示: 此脚本仅打包桌面GUI。")
    print("建议使用完整版打包: python build_exe.py")
    print()
    choice = input("继续旧版打包? (y/n, 默认n): ").strip().lower()
    if choice != "y":
        print("请运行: python build_exe.py")
        sys.exit(0)

    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "作物分类系统_完整版.spec",
        "--noconfirm", "--clean",
    ])