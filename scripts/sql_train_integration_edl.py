import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
from models.fusion_net_v5pro import FusionCropNetV5Pro
from sql.db_utils_edl import CropClassificationDBEDL
import datetime

class FusionCropDatasetEDL(Dataset):
    CROP_CLASSES = {
        0: "背景", 1: "冬小麦", 2: "夏玉米",
        3: "水稻", 4: "大豆", 5: "棉花", 6: "其他"
    }
    
    def __init__(self, opt_seq, sar_seq, doy_norm, label,
                 patch_size=32, augment=True, dem_data=None):
        self.opt_seq = opt_seq
        self.sar_seq = sar_seq
        self.doy_norm = doy_norm
        self.label = label
        self.patch_size = patch_size
        self.augment = augment
        self.dem_data = dem_data
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
        
        if self.dem_data is not None:
            dem = self.dem_data[:, r:r+p, c:c+p].copy()
        else:
            dem = np.zeros((5, p, p), dtype=np.float32)
        
        if self.augment:
            opt, sar, y, dem = self._augment(opt, sar, y, dem)
        
        return {
            "opt": torch.from_numpy(opt).float(),
            "sar": torch.from_numpy(sar).float(),
            "dem": torch.from_numpy(dem).float(),
            "doy": torch.from_numpy(self.doy_norm).float(),
            "y": torch.from_numpy(y).long()
        }
    
    def _augment(self, opt, sar, y, dem):
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-1).copy()
            sar = np.flip(sar, axis=-1).copy()
            y = np.flip(y, axis=-1).copy()
            dem = np.flip(dem, axis=-1).copy()
        if np.random.rand() > 0.5:
            opt = np.flip(opt, axis=-2).copy()
            sar = np.flip(sar, axis=-2).copy()
            y = np.flip(y, axis=-2).copy()
            dem = np.flip(dem, axis=-2).copy()
        T = opt.shape[0]
        n_drop = np.random.randint(0, 4)
        opt_drop_idx = np.random.choice(T, n_drop, replace=False)
        opt[opt_drop_idx] = 0.0
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, sar.shape).astype(np.float32)
            sar = sar + noise
        return opt, sar, y, dem

def compute_metrics(preds, labels, num_classes):
    valid = labels != 255
    p = preds[valid]
    l = labels[valid]
    oa = (p == l).float().mean().item()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, pr in zip(l.cpu().numpy(), p.cpu().numpy()):
        cm[int(t), int(pr)] += 1
    iou_list = []
    for cls in range(1, num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        iou_list.append((tp / (tp + fp + fn + 1e-6)).item())
    return {"OA": oa, "mIoU": sum(iou_list)/len(iou_list), 
            "IoU_per_class": iou_list, "confusion_matrix": cm}

def _run_training(model, train_loader, val_loader, optimizer, device, epochs, phase, exp_id=None, db=None):
    criterion = model.edl_loss_fn
    scaler = GradScaler()
    scheduler = OneCycleLR(optimizer, max_lr=[pg["lr"] for pg in optimizer.param_groups],
                           total_steps=len(train_loader) * epochs, pct_start=0.1)
    best_miou = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            opt = batch["opt"].to(device)
            sar = batch["sar"].to(device)
            dem = batch["dem"].to(device)
            doy = batch["doy"].to(device)
            y = batch["y"].to(device)
            
            optimizer.zero_grad()
            with autocast():
                alpha, ndvi_pred, consistency_loss = model(opt, sar, dem, doy, epoch=epoch)
                loss = criterion(alpha, y, epoch)
                if consistency_loss is not None:
                    loss = loss + 0.05 * consistency_loss
            
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
        total_vacuity, total_dissonance = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                opt = batch["opt"].to(device)
                sar = batch["sar"].to(device)
                dem = batch["dem"].to(device)
                doy = batch["doy"].to(device)
                y = batch["y"]
                
                with autocast():
                    alpha = model(opt, sar, dem, doy)
                    preds = dirichlet_to_predictions(alpha)
                
                all_preds.append(preds["pred_class"].cpu())
                all_labels.append(y)
                total_vacuity.append(preds["vacuity"].cpu())
                total_dissonance.append(preds["dissonance"].cpu())
        
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels), num_classes=7)
        
        vacuity_mean = torch.cat(total_vacuity).mean().item()
        vacuity_std = torch.cat(total_vacuity).std().item()
        dissonance_mean = torch.cat(total_dissonance).mean().item()
        dissonance_std = torch.cat(total_dissonance).std().item()

        # Periodic calibration metrics (every 5 epochs)
        ece_val, nll_val, brier_val = None, None, None
        if epoch % 5 == 0 or epoch == epochs:
            from utils.calibration import calibration_report
            alpha_cat = np.concatenate(
                [alpha.cpu().numpy() for _ in [
                    "placeholder"]], axis=0) if False else None
            # Collect alpha during validation
            all_alpha = []
            all_labels_cal = []
            with torch.no_grad():
                for batch in val_loader:
                    opt = batch["opt"].to(device)
                    sar = batch["sar"].to(device)
                    dem = batch["dem"].to(device)
                    doy = batch["doy"].to(device)
                    with autocast():
                        a = model(opt, sar, dem, doy)
                    all_alpha.append(a.cpu().numpy())
                    all_labels_cal.append(batch["y"])
            alpha_cat = np.concatenate(all_alpha, axis=0)
            labels_cat = torch.cat(all_labels_cal).numpy()
            cal = calibration_report(alpha_cat, labels_cat, num_classes=7, n_bins=10)
            ece_val = cal["ECE"]
            nll_val = cal["NLL"]
            brier_val = cal["Brier"]

        class_names = list(FusionCropDatasetEDL.CROP_CLASSES.values())[1:]
        print(f"[P{phase}] Epoch {epoch:3d}/{epochs} | Loss:{avg_loss:.4f} | "
              f"mIoU:{metrics['mIoU']:.4f} | OA:{metrics['OA']:.4f} | "
              f"Vacuity:{vacuity_mean:.4f} | Dissonance:{dissonance_mean:.4f}")
        if ece_val is not None:
            print(f"  校准: ECE={ece_val:.4f} NLL={nll_val:.4f} Brier={brier_val:.4f}")

        if db is not None and exp_id is not None:
            db.add_uncertainty_metrics(exp_id, epoch, phase,
                                       vacuity_mean, vacuity_std,
                                       dissonance_mean, dissonance_std,
                                       ece=ece_val, nll=nll_val, brier=brier_val)
        
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            ckpt_path = f"checkpoints/best_phase{phase}_edl.pth"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics
            }, ckpt_path)
            
            db.update_experiment_metrics(exp_id, phase, metrics["mIoU"], metrics["OA"])
            db.update_experiment_best(exp_id, metrics["mIoU"], metrics["OA"])
            db.update_experiment_uncertainty(exp_id, vacuity_mean, dissonance_mean)
            db.add_checkpoint(exp_id, epoch, ckpt_path, metrics["mIoU"], metrics["OA"])
            
            cm = metrics["confusion_matrix"]
            for cls in range(1, 7):
                db.add_confusion_matrix(exp_id, cls, class_names[cls-1],
                                        int(cm[cls, cls]),
                                        int(cm[:, cls].sum() - cm[cls, cls]),
                                        int(cm[cls, :].sum() - cm[cls, cls]))
    
    return best_miou

def train_phase1_frozen_backbone(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    print("\n" + "="*60)
    print("训练阶段1：冻结光学骨干")
    print("="*60)
    model.opt_enc.backbone.requires_grad_(False)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-4)
    return _run_training(model, train_loader, val_loader, optimizer, device, epochs, phase=1)

def train_phase2_full_finetune(model, train_loader, val_loader, device, epochs=60, lr=3e-4,
                               best_ckpt_path="best_phase1_edl.pth"):
    print("\n" + "="*60)
    print("训练阶段2：全量Fine-tune")
    print("="*60)
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✓ 加载阶段1最优权重: {best_ckpt_path}")
    
    model.opt_enc.backbone.requires_grad_(True)
    backbone_params = list(model.opt_enc.backbone.parameters())
    other_params = [p for p in model.parameters() if p not in backbone_params]
    
    param_groups = [
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": other_params, "lr": lr}
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    return _run_training(model, train_loader, val_loader, optimizer, device, epochs, phase=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5pro", action="store_true", help="Use V5Pro model instead of V5EDL")
    args = parser.parse_args()
    use_v5pro = args.v5pro

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    print(f"模型: {'V5Pro' if use_v5pro else 'V5EDL'}")
    
    db = CropClassificationDBEDL(host='localhost', database='crop_classification',
                                 user='root', password='', port=3306)
    db.connect()
    
    try:
        opt_seq = np.load("data/processed/opt_sequence_2023.npy") if os.path.exists("data/processed/opt_sequence_2023.npy") else np.random.randn(100, 10, 256, 256).astype(np.float32)
        sar_seq = np.load("data/processed/sar_sequence_2023.npy") if os.path.exists("data/processed/sar_sequence_2023.npy") else np.random.randn(100, 5, 256, 256).astype(np.float32)
        doy_norm = np.load("data/processed/doy_norm_2023.npy") if os.path.exists("data/processed/doy_norm_2023.npy") else np.linspace(0, 1, 100).astype(np.float32)
        label = np.load("data/processed/label_2023.npy") if os.path.exists("data/processed/label_2023.npy") else np.random.randint(0, 7, (256, 256)).astype(np.int64)
        
        dataset = FusionCropDatasetEDL(opt_seq, sar_seq, doy_norm, label,
                                       patch_size=32, augment=True)
        n_val = int(len(dataset) * 0.15)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        if use_v5pro:
            model = FusionCropNetV5Pro(
                opt_ch=10,
                sar_ch=5,
                dem_ch_in=5,
                num_classes=7,
                feat_dim=512,
                backbone="resnet50",
                pretrained=True,
                n_heads=16,
                win_size=4,
                n_layers=4,
                edl_dropout_p=0.3,
                edl_lambda_max=0.5,
                edl_anneal_ep=50
            ).to(device)
        else:
            model = FusionCropNetV5EDL(
                opt_ch=10,
                sar_ch=5,
                dem_ch_in=5,
                num_classes=7,
                feat_dim=512,
                backbone="resnet50",
                pretrained=True,
                n_heads=16,
                win_size=4,
                n_layers=4,
                edl_dropout_p=0.3,
                edl_lambda_max=0.5,
                edl_anneal_ep=50
            ).to(device)
        
        exp_id = db.add_experiment_edl(
            exp_name=f"FusionCropNetV5EDL_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
            model_name='FusionCropNetV5EDL',
            backbone='resnet50',
            opt_channels=10,
            sar_channels=5,
            dem_channels=5,
            num_classes=7,
            feat_dim=512,
            lr=1e-3,
            batch_size=8,
            phase1_epochs=20,
            phase2_epochs=60,
            pretrained=True,
            dataset_path='data/processed/',
            notes='使用EDL不确定性估计 + SQL数据库集成训练',
            uncertainty_enabled=True,
            edl_dropout_p=0.3,
            edl_lambda_max=0.5,
            edl_anneal_ep=50
        )
        print(f"创建EDL实验记录: exp_id={exp_id}")
        
        best_p1 = train_phase1_frozen_backbone(model, train_loader, val_loader, device,
                                                epochs=20, lr=1e-3)
        
        best_p2 = train_phase2_full_finetune(model, train_loader, val_loader, device,
                                              epochs=60, lr=3e-4,
                                              best_ckpt_path="checkpoints/best_phase1_edl.pth")
        
        print(f"\n训练完成！最佳mIoU: Phase1={best_p1:.4f}, Phase2={best_p2:.4f}")
        
    finally:
        db.disconnect()