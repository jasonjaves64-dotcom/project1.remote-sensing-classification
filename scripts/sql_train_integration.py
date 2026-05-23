import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from models.fusion_net_v5 import FusionCropNetV5
from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
from models.fusion_net_v5pro import FusionCropNetV5Pro
from models.fusion_net import PretrainedWeightManager
from sql.db_utils_edl import CropClassificationDBEDL
from utils.losses import DiceFocalLoss
import torch.nn.functional as F
import datetime

class FusionCropDataset(Dataset):
    CROP_CLASSES = {
        0: "背景", 1: "冬小麦", 2: "夏玉米",
        3: "水稻", 4: "大豆", 5: "棉花", 6: "其他"
    }
    
    def __init__(self, opt_seq, sar_seq, doy_norm, label,
                 patch_size=32, augment=True):
        self.opt_seq = opt_seq
        self.sar_seq = sar_seq
        self.doy_norm = doy_norm
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
        if self.augment:
            opt, sar, y = self._augment(opt, sar, y)
        return {
            "opt": torch.from_numpy(opt).float(),
            "sar": torch.from_numpy(sar).float(),
            "doy": torch.from_numpy(self.doy_norm).float(),
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
        T = opt.shape[0]
        n_drop = np.random.randint(0, 4)
        opt_drop_idx = np.random.choice(T, n_drop, replace=False)
        opt[opt_drop_idx] = 0.0
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, sar.shape).astype(np.float32)
            sar = sar + noise
        return opt, sar, y

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

def _run_training(model, train_loader, val_loader, optimizer, device, epochs, phase, exp_id, db, use_edl=False):
    criterion = DiceFocalLoss(num_classes=7)
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
            doy = batch["doy"].to(device)
            y = batch["y"].to(device)
            dem = torch.zeros(opt.shape[0], 5, opt.shape[-2], opt.shape[-1]).float().to(device)
            optimizer.zero_grad()
            with autocast():
                if use_edl:
                    alpha, _, _ = model(opt, sar, dem, doy, epoch=epoch)
                    preds = dirichlet_to_predictions(alpha)
                    logits = preds['probs']
                else:
                    out = model(opt, sar, dem, doy)
                    logits = out[0] if isinstance(out, tuple) else out
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
        all_vacuity, all_dissonance = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                opt = batch["opt"].to(device)
                sar = batch["sar"].to(device)
                doy = batch["doy"].to(device)
                dem = torch.zeros(opt.shape[0], 5, opt.shape[-2], opt.shape[-1]).float().to(device)
                y = batch["y"]
                with autocast():
                    if use_edl:
                        alpha = model(opt, sar, dem, doy)
                        preds = dirichlet_to_predictions(alpha)
                        pred = preds['pred_class'].cpu()
                        all_vacuity.append(preds['vacuity'].cpu())
                        all_dissonance.append(preds['dissonance'].cpu())
                    else:
                        out = model(opt, sar, dem, doy)
                        out = out[0] if isinstance(out, tuple) else out
                        pred = out.argmax(dim=1).cpu()
                all_preds.append(pred)
                all_labels.append(y)
        
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels), num_classes=7)
        
        avg_vacuity = torch.cat([v.flatten() for v in all_vacuity]).mean().item() if all_vacuity else 0
        avg_dissonance = torch.cat([d.flatten() for d in all_dissonance]).mean().item() if all_dissonance else 0
        
        class_names = list(FusionCropDataset.CROP_CLASSES.values())[1:]
        log_str = f"[P{phase}] Epoch {epoch:3d}/{epochs} | Loss:{avg_loss:.4f} | " \
                  f"mIoU:{metrics['mIoU']:.4f} | OA:{metrics['OA']:.4f}"
        if use_edl:
            log_str += f" | Vacuity:{avg_vacuity:.4f} | Dissonance:{avg_dissonance:.4f}"
        print(log_str)
        
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            ckpt_path = f"checkpoints/best_phase{phase}_edl.pth" if use_edl else f"checkpoints/best_phase{phase}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics
            }, ckpt_path)
            
            if use_edl:
                db.update_experiment_metrics(exp_id, phase, metrics["mIoU"], metrics["OA"],
                                            avg_vacuity, avg_dissonance)
                db.update_experiment_best(exp_id, metrics["mIoU"], metrics["OA"],
                                         avg_vacuity, avg_dissonance)
            else:
                db.update_experiment_metrics(exp_id, phase, metrics["mIoU"], metrics["OA"])
                db.update_experiment_best(exp_id, metrics["mIoU"], metrics["OA"])
            
            db.add_checkpoint(exp_id, epoch, ckpt_path, metrics["mIoU"], metrics["OA"])
            
            cm = metrics["confusion_matrix"]
            for cls in range(1, 7):
                db.add_confusion_matrix(exp_id, cls, class_names[cls-1],
                                        int(cm[cls, cls]),
                                        int(cm[:, cls].sum() - cm[cls, cls]),
                                        int(cm[cls, :].sum() - cm[cls, cls]))
            
            if use_edl:
                db.add_uncertainty_metrics(
                    exp_id, epoch, phase,
                    vacuity_mean=avg_vacuity, vacuity_std=0.0,
                    dissonance_mean=avg_dissonance, dissonance_std=0.0)
    
    return best_miou

def train_phase1_frozen_backbone(model, train_loader, val_loader, device, exp_id, db, 
                                  epochs=20, lr=1e-3, use_edl=False):
    print("\n" + "="*60)
    print("训练阶段1：冻结光学骨干" + (" | EDL模式" if use_edl else ""))
    print("="*60)
    manager = PretrainedWeightManager(model)
    manager.freeze_backbone(freeze_layers=6)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=1e-4)
    return _run_training(model, train_loader, val_loader, optimizer, device, epochs, 
                         phase=1, exp_id=exp_id, db=db, use_edl=use_edl)

def train_phase2_full_finetune(model, train_loader, val_loader, device, exp_id, db,
                               epochs=60, lr=3e-4, use_edl=False):
    print("\n" + "="*60)
    print("训练阶段2：全量Fine-tune" + (" | EDL模式" if use_edl else ""))
    print("="*60)
    manager = PretrainedWeightManager(model)
    best_ckpt_path = f"checkpoints/best_phase1_edl.pth" if use_edl else f"checkpoints/best_phase1.pth"
    if os.path.exists(best_ckpt_path):
        manager.load_checkpoint(best_ckpt_path, strict=True)
    manager.unfreeze_all()
    param_groups = manager.get_layerwise_lr_params(lr, backbone_lr_ratio=0.1)
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    return _run_training(model, train_loader, val_loader, optimizer, device, epochs, 
                         phase=2, exp_id=exp_id, db=db, use_edl=use_edl)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--edl", action="store_true", help="启用EDL不确定性估计")
    parser.add_argument("--v5pro", action="store_true", help="使用V5Pro模型（需要配合--edl使用）")
    args = parser.parse_args()

    use_edl = args.edl
    use_v5pro = args.v5pro
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    print(f"EDL模式: {'✓ 启用' if use_edl else '✗ 禁用'}")
    print(f"V5Pro模型: {'✓ 启用' if use_v5pro else '✗ 禁用'}")
    
    db = CropClassificationDBEDL(host='localhost', database='crop_classification',
                              user='root', password='', port=3306)
    db.connect()
    
    try:
        opt_seq = np.load("data/processed/opt_sequence_2023.npy")
        sar_seq = np.load("data/processed/sar_sequence_2023.npy")
        doy_norm = np.load("data/processed/doy_norm_2023.npy")
        label = np.load("data/processed/label_2023.npy")
        
        dataset = FusionCropDataset(opt_seq, sar_seq, doy_norm, label,
                                    patch_size=32, augment=True)
        n_val = int(len(dataset) * 0.15)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        if use_v5pro:
            model = FusionCropNetV5Pro(opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                                     feat_dim=512, backbone="resnet50", pretrained=True,
                                     n_heads=16, win_size=4, n_layers=4,
                                     edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50).to(device)
        elif use_edl:
            model = FusionCropNetV5EDL(opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                                     feat_dim=512, backbone="resnet50", pretrained=True,
                                     n_heads=16, win_size=4, n_layers=4,
                                     edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50).to(device)
        else:
            model = FusionCropNetV5(opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                                  feat_dim=512, backbone="resnet50", pretrained=True,
                                  n_heads=16, win_size=4, n_layers=4).to(device)
        
        exp_id = db.add_experiment_edl(
            exp_name=f"FusionCropNet_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}" + ("_EDL" if use_edl else ""),
            model_name='FusionCropNet' + ('EDL' if use_edl else ''),
            backbone='resnet50',
            lr=1e-3,
            batch_size=8,
            phase1_epochs=20,
            phase2_epochs=60,
            pretrained=True,
            dataset_path='data/processed/',
            notes=f'使用SQL数据库集成训练{" | EDL模式" if use_edl else ""}',
            uncertainty_enabled=use_edl,
            edl_dropout_p=0.3 if use_edl else None,
            edl_lambda_max=0.5 if use_edl else None,
            edl_anneal_ep=50 if use_edl else None
        )
        print(f"创建实验记录: exp_id={exp_id}")
        
        best_p1 = train_phase1_frozen_backbone(model, train_loader, val_loader, device, 
                                                exp_id, db, epochs=20, lr=1e-3, use_edl=use_edl)
        
        best_p2 = train_phase2_full_finetune(model, train_loader, val_loader, device,
                                              exp_id, db, epochs=60, lr=3e-4, use_edl=use_edl)
        
        print(f"\n训练完成！最佳mIoU: Phase1={best_p1:.4f}, Phase2={best_p2:.4f}")
        
    finally:
        db.disconnect()