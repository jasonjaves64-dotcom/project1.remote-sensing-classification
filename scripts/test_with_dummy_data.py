"""
使用虚拟数据测试模型，测试完成后自动清理数据
"""

import torch
import numpy as np
import os
import sys
import shutil
from datetime import datetime

sys.path.insert(0, '.')

def generate_dummy_data(output_dir="dummy_data"):
    """生成虚拟测试数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    B, T, H, W = 2, 12, 64, 64
    
    opt_seq = np.random.randn(B, T, 10, H, W).astype(np.float32) * 0.1
    sar_seq = np.random.randn(B, T, 5, H, W).astype(np.float32) * 0.1
    dem = np.random.randn(B, 5, H, W).astype(np.float32) * 0.1
    doy = np.random.rand(B, T).astype(np.float32)
    cloud_mask = np.random.randint(0, 2, (B, T, H, W)).astype(np.float32)
    valid_count = np.random.randint(1, T+1, (B, H, W)).astype(np.int32)
    labels = np.random.randint(0, 7, (B, H, W)).astype(np.int64)
    
    np.save(os.path.join(output_dir, "opt_seq.npy"), opt_seq)
    np.save(os.path.join(output_dir, "sar_seq.npy"), sar_seq)
    np.save(os.path.join(output_dir, "dem.npy"), dem)
    np.save(os.path.join(output_dir, "doy.npy"), doy)
    np.save(os.path.join(output_dir, "cloud_mask.npy"), cloud_mask)
    np.save(os.path.join(output_dir, "valid_count.npy"), valid_count)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    
    print("Dummy data generated to", output_dir+"/")
    print("  - opt_seq:", opt_seq.shape)
    print("  - sar_seq:", sar_seq.shape)
    print("  - dem:", dem.shape)
    print("  - doy:", doy.shape)
    print("  - cloud_mask:", cloud_mask.shape)
    print("  - valid_count:", valid_count.shape)
    print("  - labels:", labels.shape)
    
    return output_dir

def load_dummy_data(data_dir="dummy_data"):
    """加载虚拟数据"""
    opt_seq = torch.from_numpy(np.load(os.path.join(data_dir, "opt_seq.npy")))
    sar_seq = torch.from_numpy(np.load(os.path.join(data_dir, "sar_seq.npy")))
    dem = torch.from_numpy(np.load(os.path.join(data_dir, "dem.npy")))
    doy = torch.from_numpy(np.load(os.path.join(data_dir, "doy.npy")))
    cloud_mask = torch.from_numpy(np.load(os.path.join(data_dir, "cloud_mask.npy")))
    valid_count = torch.from_numpy(np.load(os.path.join(data_dir, "valid_count.npy")))
    labels = torch.from_numpy(np.load(os.path.join(data_dir, "labels.npy")))
    
    return opt_seq, sar_seq, dem, doy, cloud_mask, valid_count, labels

def test_model_with_dummy_data():
    """使用虚拟数据测试模型"""
    from models.fusion_net_v5 import FusionCropNetV5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)
    
    model = FusionCropNetV5(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
        feat_dim=512, backbone="resnet50", pretrained=False,
        n_heads=16, win_size=4, n_layers=4,
        drop_timestep_p=0.1
    ).to(device)
    model.train()
    
    opt_seq, sar_seq, dem, doy, cloud_mask, valid_count, labels = load_dummy_data()
    opt_seq = opt_seq.to(device)
    sar_seq = sar_seq.to(device)
    dem = dem.to(device)
    doy = doy.to(device)
    cloud_mask = cloud_mask.to(device)
    valid_count = valid_count.to(device)
    labels = labels.to(device)
    
    print("\n=== Starting Test ===")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(3):
        optimizer.zero_grad()
        
        logits, ndvi_pred, consistency_loss = model(
            opt_seq, sar_seq, dem, doy, cloud_mask, valid_count
        )
        
        ce_loss = torch.nn.CrossEntropyLoss()(logits, labels)
        huber_loss = torch.nn.HuberLoss(delta=0.1)(ndvi_pred, torch.zeros_like(ndvi_pred))
        
        total_loss = ce_loss + 0.1 * huber_loss
        if consistency_loss is not None:
            total_loss += 0.01 * consistency_loss
        
        total_loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        print("Epoch", epoch+1, "/3:")
        print("  CE Loss:", ce_loss.item())
        print("  Huber Loss:", huber_loss.item())
        if consistency_loss is not None:
            print("  Consistency Loss:", consistency_loss.item())
        else:
            print("  Consistency Loss: None")
        print("  Total Loss:", total_loss.item())
        print("  Accuracy:", accuracy*100, "%")
    
    print("\n=== Inference Mode Test ===")
    model.eval()
    with torch.no_grad():
        logits = model(opt_seq, sar_seq, dem, doy)
    
    preds = logits.argmax(dim=1)
    unique_classes = torch.unique(preds)
    print("Predicted classes:", sorted(unique_classes.tolist()))
    print("Output shape:", logits.shape)
    
    print("\nModel test completed!")

def cleanup_dummy_data(data_dir="dummy_data"):
    """清理虚拟数据"""
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("\nDummy data cleaned:", data_dir+"/")

def main():
    print("=" * 60)
    print("Test FusionCropNetV5 with Dummy Data")
    print("=" * 60)
    print("Test time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        generate_dummy_data()
        test_model_with_dummy_data()
    finally:
        cleanup_dummy_data()
    
    print("\n" + "=" * 60)
    print("Test completed, data cleaned")
    print("=" * 60)

if __name__ == "__main__":
    main()
