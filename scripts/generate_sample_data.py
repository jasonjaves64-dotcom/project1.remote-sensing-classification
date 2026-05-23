import numpy as np
import os

def generate_sample_data():
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    T = 12
    H, W = 256, 256
    opt_channels = 10
    sar_channels = 3
    
    np.random.seed(42)
    
    opt_seq = np.random.uniform(0, 1, (T, opt_channels, H, W)).astype(np.float32)
    sar_seq = np.random.uniform(-15, 5, (T, sar_channels, H, W)).astype(np.float32)
    doy_norm = np.linspace(0, 1, T).astype(np.float32)
    
    label = np.zeros((H, W), dtype=np.uint8)
    
    for i in range(1, 7):
        mask = np.random.rand(H, W) < 0.15
        label[mask] = i
    
    np.save(os.path.join(output_dir, "opt_sequence_2023.npy"), opt_seq)
    np.save(os.path.join(output_dir, "sar_sequence_2023.npy"), sar_seq)
    np.save(os.path.join(output_dir, "doy_norm_2023.npy"), doy_norm)
    np.save(os.path.join(output_dir, "label_2023.npy"), label)
    
    print(f"✓ 生成示例数据完成:")
    print(f"  - opt_sequence_2023.npy: {opt_seq.shape}")
    print(f"  - sar_sequence_2023.npy: {sar_seq.shape}")
    print(f"  - doy_norm_2023.npy: {doy_norm.shape}")
    print(f"  - label_2023.npy: {label.shape}")
    print(f"  - 类别分布: {np.bincount(label.flatten())}")

if __name__ == "__main__":
    generate_sample_data()
