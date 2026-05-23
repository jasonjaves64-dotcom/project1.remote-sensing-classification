import os
import shutil
from pathlib import Path

base_dir = Path("./data")

src_landsat = base_dir / "landsat_images" / "2023"
src_labels = base_dir / "labels"

dst_landsat = base_dir / "test_data" / "landsat_images" / "2023"
dst_labels = base_dir / "test_data" / "labels"

dst_landsat.mkdir(parents=True, exist_ok=True)
dst_labels.mkdir(parents=True, exist_ok=True)

print("Moving Landsat images...")
for npy_file in src_landsat.glob("*.npy"):
    dst_file = dst_landsat / npy_file.name
    shutil.move(str(npy_file), str(dst_file))
    print(f"  {npy_file.name} -> test_data/landsat_images/2023/")

print("\nMoving labels...")
for npy_file in src_labels.glob("*.npy"):
    dst_file = dst_labels / npy_file.name
    shutil.move(str(npy_file), str(dst_file))
    print(f"  {npy_file.name} -> test_data/labels/")

print("\n✅ 数据整理完成！")
print("\n新的目录结构：")
print("data/")
print("  test_data/")
print("    landsat_images/")
print("      2023/")
print("        *.npy")
print("    labels/")
print("      crop_label_2023.npy")