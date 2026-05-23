"""
训练脚本 — 统一入口（标准 + EDL + 模态解耦）

Usage:
  # 标准训练
  python scripts/train_fusion.py --epochs 80

  # EDL模式
  python scripts/train_fusion.py --edl --epochs 80

  # 模态解耦训练（随机丢弃某个模态以增强鲁棒性）
  python scripts/train_fusion.py --edl --modality_dropout 0.3 --epochs 80
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fusion_net_v5_edl import (
    FusionCropNetV5EDL, EDLLoss, dirichlet_to_predictions
)
from models.fusion_net_v6 import FusionCropNetV6
from data.datasets import FusionCropDatasetEDL
from utils.trainer import FusionTrainer
from utils.metrics import compute_metrics
from utils.calibration import calibration_report, print_calibration_report


def main():
    parser = argparse.ArgumentParser(description="Train FusionCropNet model")
    parser.add_argument("--edl", action="store_true", help="Enable EDL uncertainty estimation")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--model_path", type=str, default="checkpoints/")
    parser.add_argument("--data_path", type=str, default="data/processed/")
    parser.add_argument("--modality_dropout", type=float, default=0.0,
                        help="Probability of dropping a modality during training (0-1)")
    parser.add_argument("--v5pro", action="store_true", help="Use FusionCropNetV5Pro")
    parser.add_argument("--v6", action="store_true", help="Use FusionCropNetV6")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="Backbone: resnet50, convnext_tiny, efficientnet_b0")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    os.makedirs(args.model_path, exist_ok=True)

    # Build model
    if args.v6:
        model = FusionCropNetV6(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone=args.backbone, pretrained=True,
            n_heads=16, win_size=4, n_layers=4,
            modality_dropout_p=args.modality_dropout,
            use_gradient_checkpointing=torch.cuda.is_available(),
        ).to(device)
        print(f"Using V6 with backbone={args.backbone}")
    elif args.v5pro:
        from models.fusion_net_v5pro import FusionCropNetV5Pro
        model = FusionCropNetV5Pro(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone=args.backbone, pretrained=True,
            n_heads=16, win_size=4, n_layers=4,
            modality_dropout_p=args.modality_dropout,
            use_carafe=True, dynamic_dropout=True, adaptive_kl=True,
        ).to(device)
        print(f"Using V5Pro with backbone={args.backbone}")
    else:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=True,
            n_heads=16, win_size=4, n_layers=4,
            modality_dropout_p=args.modality_dropout,
            use_v6_enhancements=False,
        ).to(device)

    # Loss
    if args.edl:
        criterion = EDLLoss(num_classes=7, lambda_max=0.5, kl_anneal_epochs=50)
    else:
        from utils.losses import WeightedDiceFocalLoss
        criterion = WeightedDiceFocalLoss(num_classes=7)

    # Load data
    data_dir = args.data_path
    opt_seq = np.load(os.path.join(data_dir, "opt_sequence.npy"))
    sar_seq = np.load(os.path.join(data_dir, "sar_sequence.npy"))
    doy_norm = np.load(os.path.join(data_dir, "doy_norm.npy"))
    label = np.load(os.path.join(data_dir, "label.npy"))

    dem_path = os.path.join(data_dir, "dem.npy")
    dem_data = np.load(dem_path) if os.path.exists(dem_path) else None

    from data.datasets import FusionCropDatasetEDL

    # Simple train/val split
    H, W = label.shape
    split_col = int(W * 0.85)
    train_ds = FusionCropDatasetEDL(
        opt_seq[:, :, :, :split_col], sar_seq[:, :, :, :split_col],
        doy_norm, label[:, :split_col],
        patch_size=args.patch_size, augment=True,
        dem_data=dem_data[:, :, :split_col] if dem_data is not None else None,
    )
    val_ds = FusionCropDatasetEDL(
        opt_seq[:, :, :, split_col:], sar_seq[:, :, :, split_col:],
        doy_norm, label[:, split_col:],
        patch_size=args.patch_size, augment=False,
        dem_data=dem_data[:, :, split_col:] if dem_data is not None else None,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train patches: {len(train_ds)}, Val patches: {len(val_ds)}")

    # Train
    from torch.optim import AdamW
    from torch.cuda.amp import GradScaler

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if device.type == "cuda" else None

    trainer = FusionTrainer(model, optimizer, criterion, device, scaler)
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    best_ckpt = os.path.join(args.model_path, "best_model.pth")
    torch.save(model.state_dict(), best_ckpt)
    print(f"Model saved to {best_ckpt}")
    print(f"Best mIoU: {history.get('best_miou', 'N/A')}")

    # EDL calibration validation after training
    if args.edl:
        print("\n" + "=" * 60)
        print("EDL校准验证 (Validation Set)")
        print("=" * 60)
        model.eval()
        all_alpha, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                opt = batch["opt"].to(device)
                sar = batch["sar"].to(device)
                dem = batch["dem"].to(device)
                doy = batch["doy"].to(device)
                alpha = model(opt, sar, dem, doy)
                all_alpha.append(alpha.cpu().numpy())
                all_labels.append(batch["y"])
        alpha_cat = np.concatenate(all_alpha, axis=0)
        labels_cat = torch.cat(all_labels).numpy()
        cal = calibration_report(alpha_cat, labels_cat, num_classes=7, n_bins=15)
        print_calibration_report(cal)

        output_dir = "calibration_output"
        os.makedirs(output_dir, exist_ok=True)
        import json
        serializable = {k: v for k, v in cal.items() if k != "_raw"}
        with open(f"{output_dir}/train_fusion_calibration.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2,
                      default=lambda x: float(x) if hasattr(x, 'item') else str(x))
        print(f"Calibration report saved to {output_dir}/train_fusion_calibration.json")


if __name__ == "__main__":
    main()
