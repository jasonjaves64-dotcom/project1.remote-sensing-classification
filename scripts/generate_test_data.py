"""
生成虚拟遥感数据用于测试
"""

import numpy as np
import os
import argparse
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds

def generate_optical_data(seq_len=12, height=256, width=256, channels=10):
    """生成光学时序数据"""
    data = np.zeros((seq_len, channels, height, width), dtype=np.float32)
    
    for t in range(seq_len):
        base_reflectance = np.random.uniform(0.05, 0.3, (channels, height, width))
        ndvi_phase = np.sin(2 * np.pi * t / seq_len) * 0.3 + 0.5
        base_reflectance[3] *= ndvi_phase
        noise = np.random.normal(0, 0.02, (channels, height, width))
        data[t] = np.clip(base_reflectance + noise, 0, 1)
    
    return data

def generate_sar_data(seq_len=12, height=256, width=256, channels=5):
    """生成SAR时序数据"""
    data = np.zeros((seq_len, channels, height, width), dtype=np.float32)
    
    for t in range(seq_len):
        sigma0 = np.random.uniform(-25, -5, (channels, height, width))
        linear_value = 10 ** (sigma0 / 10)
        data[t] = linear_value
    
    return data

def generate_dem_data(height=256, width=256, channels=5):
    """生成DEM数据"""
    data = np.zeros((channels, height, width), dtype=np.float32)
    
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    terrain = np.sin(xx * 4 * np.pi) * np.cos(yy * 4 * np.pi) * 50 + 100
    data[0] = terrain
    
    dx, dy = np.gradient(terrain)
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
    data[1] = slope
    
    aspect = np.arctan2(dy, dx) * 180 / np.pi
    data[2] = aspect
    
    data[3] = np.random.uniform(-0.1, 0.1, (height, width))
    data[4] = np.random.uniform(0, 0.5, (height, width))
    
    return data

def generate_label_data(height=256, width=256, num_classes=7):
    """生成标签数据"""
    labels = np.zeros((height, width), dtype=np.int32)
    
    regions = [
        (0, 0, height//2, width//2, 1),
        (0, width//2, height//2, width, 2),
        (height//2, 0, height, width//2, 3),
        (height//2, width//2, height, width, 4),
    ]
    
    for y1, x1, y2, x2, label in regions:
        labels[y1:y2, x1:x2] = label
    
    noise_mask = np.random.random((height, width)) < 0.05
    labels[noise_mask] = np.random.randint(0, num_classes, np.sum(noise_mask))
    
    return labels

def save_as_geotiff(data, filepath, is_sequence=False):
    """保存为GeoTIFF文件"""
    if is_sequence:
        seq_len = data.shape[0]
        for t in range(seq_len):
            filename = f"{filepath}_{t:03d}.tif"
            single_data = data[t]
            save_single_geotiff(single_data, filename)
    else:
        save_single_geotiff(data, filepath)

def save_single_geotiff(data, filepath):
    """保存单张GeoTIFF"""
    if len(data.shape) == 2:
        data = data[np.newaxis, ...]
    
    channels, height, width = data.shape
    transform = from_bounds(0, 0, width, height, width, height)
    
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=channels,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

def main():
    parser = argparse.ArgumentParser(description="生成虚拟遥感数据")
    parser.add_argument("--output_dir", type=str, default="test_data", help="输出目录")
    parser.add_argument("--seq_len", type=int, default=12, help="时序长度")
    parser.add_argument("--height", type=int, default=256, help="高度")
    parser.add_argument("--width", type=int, default=256, help="宽度")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("生成光学数据...")
    opt_data = generate_optical_data(args.seq_len, args.height, args.width)
    np.save(output_dir / "optical_sequence.npy", opt_data)
    opt_dir = output_dir / "optical"
    opt_dir.mkdir(exist_ok=True)
    save_as_geotiff(opt_data, str(opt_dir / "optical"), is_sequence=True)
    
    print("生成SAR数据...")
    sar_data = generate_sar_data(args.seq_len, args.height, args.width)
    np.save(output_dir / "sar_sequence.npy", sar_data)
    sar_dir = output_dir / "sar"
    sar_dir.mkdir(exist_ok=True)
    save_as_geotiff(sar_data, str(sar_dir / "sar"), is_sequence=True)
    
    print("生成DEM数据...")
    dem_data = generate_dem_data(args.height, args.width)
    np.save(output_dir / "dem.npy", dem_data)
    save_single_geotiff(dem_data, str(output_dir / "dem.tif"))
    
    print("生成标签数据...")
    label_data = generate_label_data(args.height, args.width)
    np.save(output_dir / "labels.npy", label_data)
    save_single_geotiff(label_data, str(output_dir / "labels.tif"))
    
    doy_data = np.linspace(0, 1, args.seq_len)
    np.save(output_dir / "doy.npy", doy_data)
    
    print(f"\n测试数据已生成到: {output_dir}")
    print(f"光学数据: {opt_data.shape}")
    print(f"SAR数据: {sar_data.shape}")
    print(f"DEM数据: {dem_data.shape}")
    print(f"标签数据: {label_data.shape}")

if __name__ == "__main__":
    main()
