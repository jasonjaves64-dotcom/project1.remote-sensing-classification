import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from models.unet_transformer import UNetTransformer, UNetTransformerWithSAR
from utils.losses import WeightedDiceFocalLoss

class CropDataset(Dataset):
    def __init__(self, opt_seq, sar_seq, label, patch_size=32, augment=True):
        self.opt_seq = opt_seq
        self.sar_seq = sar_seq
        self.label = label
        self.patch_size = patch_size
        self.augment = augment
        
        H, W = label.shape
        p = patch_size
        self.coords = [
            (r, c)
            for r in range(0, H - p + 1, p // 2)
            for c in range(0, W - p + 1, p // 2)
            if label[r:r+p, c:c+p].max() > 0
        ]
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        r, c = self.coords[idx]
        p = self.patch_size
        opt = self.opt_seq[:, :, r:r+p, c:c+p].copy()
        sar = self.sar_seq[:, :, r:r+p, c:c+p].copy()
        y = self.label[r:r+p, c:c+p].copy()
        
        y[y == 255] = 0
        
        if self.augment:
            opt, sar, y = self._augment(opt, sar, y)
        
        return {
            "opt": torch.from_numpy(opt).float(),
            "sar": torch.from_numpy(sar).float(),
            "y": torch.from_numpy(y).long()
        }
    
    def _augment(self, opt, sar, y):
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-1).copy()
            sar = np.flip(sar, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()
        
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-2).copy()
            sar = np.flip(sar, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()
        
        return opt, sar, y

def compute_metrics(preds, labels, num_classes=7):
    valid = labels != 0
    p = preds[valid]
    l = labels[valid]
    oa = (p == l).float().mean().item()
    
    iou_list = []
    for cls in range(1, num_classes):
        tp = ((p == cls) & (l == cls)).sum().float()
        fp = ((p == cls) & (l != cls)).sum().float()
        fn = ((p != cls) & (l == cls)).sum().float()
        iou_list.append((tp / (tp + fp + fn + 1e-6)).item())
    
    return {
        "OA": oa,
        "mIoU": sum(iou_list) / len(iou_list) if iou_list else 0,
        "IoU_per_class": iou_list
    }

def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4):
    criterion = WeightedDiceFocalLoss(num_classes=7)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=len(train_loader) * epochs,
        pct_start=0.1
    )
    
    best_miou = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            opt = batch["opt"].to(device)
            sar = batch["sar"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(opt, sar)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                opt = batch["opt"].to(device)
                sar = batch["sar"].to(device)
                y = batch["y"]
                
                with autocast():
                    preds = model(opt, sar).argmax(dim=1).cpu()
                
                all_preds.append(preds)
                all_labels.append(y)
        
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss:{avg_loss:.4f} | mIoU:{metrics['mIoU']:.4f} | "
              f"OA:{metrics['OA']:.4f}")
        
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics
            }, "checkpoints/unet_transformer_best.pth")
            print(f"  ✓ 保存最优模型 (mIoU: {best_miou:.4f})")
    
    return best_miou

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    try:
        opt_seq = np.load("data/processed/opt_sequence.npy")
        sar_seq = np.load("data/processed/sar_sequence.npy")
        
        if os.path.exists("data/processed/final_label.npy"):
            label = np.load("data/processed/final_label.npy")
        else:
            label = np.load("data/processed/label.npy")
        
        print(f"数据加载完成:")
        print(f"  光学时序: {opt_seq.shape}")
        print(f"  SAR时序: {sar_seq.shape}")
        print(f"  标签: {label.shape}")
        
        dataset = CropDataset(opt_seq, sar_seq, label, patch_size=32, augment=True)
        
        n_val = int(len(dataset) * 0.15)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        print(f"\n训练样本数: {len(train_ds)}")
        print(f"验证样本数: {len(val_ds)}")
        
        model = UNetTransformerWithSAR(
            opt_channels=opt_seq.shape[1],
            sar_channels=sar_seq.shape[1],
            n_classes=7,
            use_transformer=True,
            num_heads=8,
            depth=4
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型总参数量: {total_params/1e6:.2f}M")
        
        print("\n开始训练 UNet-Transformer 模型...")
        best_miou = train_model(model, train_loader, val_loader, device, epochs=50)
        
        print(f"\n训练完成！最佳mIoU: {best_miou:.4f}")
        
    except FileNotFoundError as e:
        print(f"错误: 未找到数据文件 - {e}")
        print("请先运行数据预处理脚本生成训练数据")