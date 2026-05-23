#!/usr/bin/env python3
"""
MIL (Multi-Instance Learning) training script for crop classification.

Supports three training modes:
  --mode mil        : Pure MIL — image-level labels only (weakly supervised)
  --mode hybrid     : Mixed supervision — combines bag-level and pixel-level loss
  --mode pretrain   : Supervised pre-training of the base encoder (standard EDL)

Usage:
  # Phase 1: Supervised pre-training (standard pixel-level EDL)
  python scripts/train_mil.py --mode pretrain --epochs 80

  # Phase 2: MIL fine-tuning with frozen encoder
  python scripts/train_mil.py --mode mil --pretrained best_phase1_edl.pth --epochs 50

  # Phase 3: Hybrid training (bag + pixel supervision)
  python scripts/train_mil.py --mode hybrid --pretrained best_mil.pth --epochs 30
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.fusion_net_v5_edl import (
    FusionCropNetV5EDL, EDLLoss, dirichlet_to_predictions
)
from models.mil_module import (
    FusionCropNetV5MIL, create_mil_model,
    bag_dirichlet_to_predictions
)
from data.mil_dataset import MILCropDataset, mil_collate_fn

CROP_CLASSES = {
    0: "背景", 1: "冬小麦", 2: "夏玉米",
    3: "水稻", 4: "大豆", 5: "棉花", 6: "其他"
}


def build_model(args, device) -> nn.Module:
    """Build model based on training mode."""
    config = {
        "opt_channels": 10, "sar_channels": 5, "dem_channels": 5,
        "num_classes": 7, "feat_dim": 512, "backbone": "resnet50",
        "pretrained": True, "n_heads": 16, "win_size": 4,
        "n_layers": 4, "drop_timestep_p": 0.1, "edl_dropout_p": 0.3,
    }

    base_model = FusionCropNetV5EDL(**{
        k: v for k, v in config.items()
        if k in ["opt_channels", "sar_channels", "dem_channels",
                  "num_classes", "feat_dim", "backbone", "pretrained",
                  "n_heads", "win_size", "n_layers", "drop_timestep_p",
                  "edl_dropout_p"]
    })
    base_model = base_model.to(device)

    if args.pretrained and os.path.exists(args.pretrained):
        state = torch.load(args.pretrained, map_location=device)
        base_model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")

    if args.mode == "pretrain":
        return base_model

    mil_model = FusionCropNetV5MIL(
        base_model=base_model,
        num_classes=7,
        pool_method=args.pool_method,
        freeze_encoder=(args.mode == "mil"),
    )
    return mil_model.to(device)


def load_data(args) -> tuple:
    """Load data and create dataloaders."""
    print("Loading data...")
    data_root = args.data_dir or os.path.join(PROJECT_ROOT, "data")
    processed = os.path.join(data_root, "processed")

    opt_seq = np.load(os.path.join(processed, "opt_sequence.npy"))
    sar_seq = np.load(os.path.join(processed, "sar_sequence.npy"))
    doy_norm = np.load(os.path.join(processed, "doy_norm.npy"))
    label = np.load(os.path.join(processed, "label.npy"))

    dem_path = os.path.join(processed, "dem.npy")
    dem_data = np.load(dem_path) if os.path.exists(dem_path) else None

    print(f"  Optical: {opt_seq.shape}, SAR: {sar_seq.shape}, Label: {label.shape}")
    if dem_data is not None:
        print(f"  DEM: {dem_data.shape}")

    H, W = label.shape

    # Train/val split
    if args.spatial_split:
        split_row = int(H * (1 - args.val_split))
        train_slice = (slice(0, split_row), slice(0, W))
        val_slice = (slice(split_row, H), slice(0, W))
    else:
        split_col = int(W * (1 - args.val_split))
        train_slice = (slice(0, H), slice(0, split_col))
        val_slice = (slice(0, H), slice(split_col, W))

    if args.mode in ("mil", "hybrid"):
        train_ds = MILCropDataset(
            opt_seq=opt_seq[:, :, train_slice[0], train_slice[1]],
            sar_seq=sar_seq[:, :, train_slice[0], train_slice[1]],
            doy_norm=doy_norm,
            label=label[train_slice[0], train_slice[1]],
            dem_data=dem_data[:, train_slice[0], train_slice[1]]
            if dem_data is not None else None,
            bag_size=args.bag_size,
            patch_size=args.patch_size,
            stride=args.stride,
            augment=True,
            return_pixel_labels=(args.mode == "hybrid"),
        )
        val_ds = MILCropDataset(
            opt_seq=opt_seq[:, :, val_slice[0], val_slice[1]],
            sar_seq=sar_seq[:, :, val_slice[0], val_slice[1]],
            doy_norm=doy_norm,
            label=label[val_slice[0], val_slice[1]],
            dem_data=dem_data[:, val_slice[0], val_slice[1]]
            if dem_data is not None else None,
            bag_size=args.bag_size,
            patch_size=args.patch_size,
            stride=args.stride,
            augment=False,
            return_pixel_labels=False,
        )

        collate = mil_collate_fn
    else:
        from data.datasets import FusionCropDatasetEDL
        train_ds = FusionCropDatasetEDL(
            opt_seq=opt_seq[:, :, train_slice[0], train_slice[1]],
            sar_seq=sar_seq[:, :, train_slice[0], train_slice[1]],
            doy_norm=doy_norm,
            label=label[train_slice[0], train_slice[1]],
            patch_size=args.patch_size,
            augment=True,
            dem_data=dem_data[:, train_slice[0], train_slice[1]]
            if dem_data is not None else None,
        )
        val_ds = FusionCropDatasetEDL(
            opt_seq=opt_seq[:, :, val_slice[0], val_slice[1]],
            sar_seq=sar_seq[:, :, val_slice[0], val_slice[1]],
            doy_norm=doy_norm,
            label=label[val_slice[0], val_slice[1]],
            patch_size=args.patch_size,
            augment=False,
            dem_data=dem_data[:, val_slice[0], val_slice[1]]
            if dem_data is not None else None,
        )
        collate = None

    print(f"  Train bags: {len(train_ds)}, Val bags: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, val_loader


# =============================================================================
# Training loops
# =============================================================================

def train_epoch_mil(model, train_loader, optimizer, criterion,
                    scaler, scheduler, device, epoch, args):
    """MIL training loop: bag-level supervision."""
    model.train()
    total_loss = 0.0
    total_oa = 0.0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        opt = batch["opt"].to(device)       # (B, N, T, 10, P, P)
        sar = batch["sar"].to(device)       # (B, N, T, 5,  P, P)
        dem = batch["dem"].to(device)       # (B, N, 5, P, P)
        doy = batch["doy"].to(device)       # (B, N, T)
        mask = batch["instance_mask"].to(device)  # (B, N)
        bag_label = batch["bag_label"].to(device)  # (B,)

        with autocast():
            alpha, attn_weights = model(
                opt, sar, dem, doy, instance_mask=mask, epoch=epoch)
            loss = criterion(alpha, bag_label, epoch)

            loss = loss / args.grad_accum

        scaler.scale(loss).backward()

        with torch.no_grad():
            preds = bag_dirichlet_to_predictions(alpha)
            oa = (preds["pred_class"] == bag_label).float().mean()

        total_loss += loss.item() * args.grad_accum
        total_oa += oa.item()

        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_oa = total_oa / len(train_loader)
    return avg_loss, avg_oa


def train_epoch_hybrid(model, train_loader, optimizer, criterion,
                       scaler, scheduler, device, epoch, args):
    """Hybrid training: bag-level + pixel-level supervision."""
    model.train()
    total_loss = 0.0
    total_oa = 0.0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        opt = batch["opt"].to(device)
        sar = batch["sar"].to(device)
        dem = batch["dem"].to(device)
        doy = batch["doy"].to(device)
        mask = batch["instance_mask"].to(device)
        bag_label = batch["bag_label"].to(device)
        pixel_labels = batch.get("pixel_labels")

        with autocast():
            alpha, attn_weights = model(
                opt, sar, dem, doy, instance_mask=mask, epoch=epoch)

            # Bag-level EDL loss
            bag_loss = criterion(alpha, bag_label, epoch)

            # Pixel-level loss via attention-weighted instance predictions
            if pixel_labels is not None:
                pixel_labels = pixel_labels.to(device)
                B, N = opt.shape[0], opt.shape[1]
                opt_flat = opt.view(B * N, *opt.shape[2:])
                sar_flat = sar.view(B * N, *sar.shape[2:])
                dem_flat = dem.view(B * N, *dem.shape[2:])
                doy_flat = doy.view(B * N, *doy.shape[2:])

                with torch.no_grad():
                    px_alpha = model.extractor.encoder(
                        opt_flat, sar_flat, dem_flat, doy_flat)
                    if isinstance(px_alpha, tuple):
                        px_alpha = px_alpha[0]
                px_alpha = px_alpha.view(B, N, *px_alpha.shape[1:])

                px_loss = 0.0
                valid_count = 0
                for b in range(B):
                    for n in range(N):
                        if mask[b, n] and (pixel_labels[b, n] != 255).any():
                            inst_alpha = px_alpha[b, n]
                            inst_label = pixel_labels[b, n]
                            # 255 is EDLLoss.ignore_index, so no clamping needed
                            px_loss += criterion(inst_alpha.unsqueeze(0),
                                                 inst_label.unsqueeze(0), epoch)
                            valid_count += 1
                if valid_count > 0:
                    px_loss = px_loss / valid_count
                else:
                    px_loss = torch.tensor(0.0, device=device)

                loss = bag_loss + args.pixel_loss_weight * px_loss
            else:
                loss = bag_loss

            loss = loss / args.grad_accum

        scaler.scale(loss).backward()

        with torch.no_grad():
            preds = bag_dirichlet_to_predictions(alpha)
            oa = (preds["pred_class"] == bag_label).float().mean()

        total_loss += loss.item() * args.grad_accum
        total_oa += oa.item()

        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_oa = total_oa / len(train_loader)
    return avg_loss, avg_oa


@torch.no_grad()
def validate_mil(model, val_loader, criterion, device, epoch):
    """Validation for MIL model."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_vacuity, all_dissonance = [], []
    all_attn = []

    for batch in val_loader:
        opt = batch["opt"].to(device)
        sar = batch["sar"].to(device)
        dem = batch["dem"].to(device)
        doy = batch["doy"].to(device)
        mask = batch["instance_mask"].to(device)
        bag_label = batch["bag_label"].to(device)

        if hasattr(model, 'predict_bag'):
            preds = model.predict_bag(opt, sar, dem, doy, mask, n_passes=3)
            alpha = preds["alpha"]
        else:
            alpha, attn = model(opt, sar, dem, doy, instance_mask=mask)
            preds = bag_dirichlet_to_predictions(alpha)
            preds["attn_weights"] = attn

        loss = criterion(alpha, bag_label, epoch) if criterion else torch.tensor(0.0)
        total_loss += loss.item()

        all_preds.append(preds["pred_class"].cpu())
        all_labels.append(bag_label.cpu())
        all_vacuity.append(preds["vacuity"].cpu())
        all_dissonance.append(preds["dissonance"].cpu())
        if "attn_weights" in preds:
            all_attn.append(preds["attn_weights"].cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_vacuity = torch.cat(all_vacuity)
    all_dissonance = torch.cat(all_dissonance)

    # Bag-level accuracy + per-class F1
    oa = (all_preds == all_labels).float().mean().item()

    f1_per_class = {}
    for cls in range(1, 7):
        tp = ((all_preds == cls) & (all_labels == cls)).sum().float().item()
        fp = ((all_preds == cls) & (all_labels != cls)).sum().float().item()
        fn = ((all_preds != cls) & (all_labels == cls)).sum().float().item()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
        f1_per_class[CROP_CLASSES[cls]] = f1

    avg_f1 = sum(f1_per_class.values()) / len(f1_per_class)

    attn_entropy = None
    if all_attn:
        all_attn = torch.cat(all_attn, dim=0)
        eps = 1e-8
        attn_entropy = -(all_attn * (all_attn + eps).log()).sum(dim=-1).mean().item()

    return {
        "loss": total_loss / len(val_loader),
        "OA": oa,
        "macro_F1": avg_f1,
        "F1_per_class": f1_per_class,
        "vacuity": all_vacuity.mean().item(),
        "dissonance": all_dissonance.mean().item(),
        "attn_entropy": attn_entropy,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MIL training for crop classification")
    parser.add_argument("--mode", type=str, default="mil",
                        choices=["pretrain", "mil", "hybrid"],
                        help="Training mode")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to pretrained weights")
    parser.add_argument("--pool_method", type=str, default="attention",
                        choices=["attention", "gated"])
    parser.add_argument("--bag_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--spatial_split", action="store_true")
    parser.add_argument("--pixel_loss_weight", type=float, default=0.3,
                        help="Weight for pixel-level loss in hybrid mode")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = load_data(args)
    model = build_model(args, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
          f"({100 * trainable_params / total_params:.1f}%)")

    # Setup training
    if args.mode == "pretrain":
        criterion = EDLLoss(num_classes=7, lambda_max=0.5, kl_anneal_epochs=50)
    else:
        criterion = EDLLoss(num_classes=7, lambda_max=0.3, kl_anneal_epochs=50)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=total_steps, pct_start=0.1)
    scaler = GradScaler()

    best_f1 = 0.0
    best_ckpt = os.path.join(args.output_dir, f"best_{args.mode}.pth")
    history = {"train_loss": [], "val_oa": [], "val_f1": [],
               "vacuity": [], "dissonance": []}

    train_fn = train_epoch_mil if args.mode == "mil" else train_epoch_hybrid
    if args.mode == "pretrain":
        train_fn = None  # Use _run_training from train_fusion_edl

    for epoch in range(1, args.epochs + 1):
        if args.mode == "pretrain":
            # For pretrain mode, delegate to the existing EDL training loop
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                opt = batch["opt"].to(device)
                sar = batch["sar"].to(device)
                dem = batch.get("dem", torch.zeros(
                    opt.shape[0], 5, opt.shape[-2], opt.shape[-1])).to(device)
                doy = batch["doy"].to(device)
                y = batch["y"].to(device)

                with autocast():
                    alpha, ndvi_pred, consistency_loss = model(
                        opt, sar, dem, doy, epoch=epoch)
                    edl_loss = criterion(alpha, y, epoch)
                    loss = edl_loss
                    if consistency_loss is not None:
                        loss = loss + 0.05 * consistency_loss
                    loss = loss / args.grad_accum

                scaler.scale(loss).backward()

                if (batch_idx + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                total_loss += loss.item() * args.grad_accum

            avg_loss = total_loss / len(train_loader)
            train_oa = 0.0
        else:
            avg_loss, train_oa = train_fn(
                model, train_loader, optimizer, criterion,
                scaler, scheduler, device, epoch, args)

        val_metrics = validate_mil(model, val_loader, criterion, device, epoch)

        history["train_loss"].append(avg_loss)
        history["val_oa"].append(val_metrics["OA"])
        history["val_f1"].append(val_metrics["macro_F1"])
        history["vacuity"].append(val_metrics["vacuity"])
        history["dissonance"].append(val_metrics["dissonance"])

        f1_str = " | ".join(
            [f"{n}:{v:.3f}" for n, v in val_metrics["F1_per_class"].items()])

        print(f"[{args.mode.upper():>7s}] Epoch {epoch:3d}/{args.epochs} | "
              f"Loss:{avg_loss:.4f} | OA:{val_metrics['OA']:.4f} | "
              f"F1:{val_metrics['macro_F1']:.4f}")
        print(f"  Vac:{val_metrics['vacuity']:.4f} | "
              f"Dis:{val_metrics['dissonance']:.4f} | "
              f"AttnE:{val_metrics['attn_entropy'] or 0:.4f}")
        print(f"  {f1_str}")

        if val_metrics["macro_F1"] > best_f1:
            best_f1 = val_metrics["macro_F1"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [save] {best_ckpt} (F1={best_f1:.4f})")

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Model saved to: {best_ckpt}")


if __name__ == "__main__":
    main()
