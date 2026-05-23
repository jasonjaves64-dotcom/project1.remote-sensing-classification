import os
import sys
import json
import yaml
import random
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from models.fusion_net_v5_edl import (
    FusionCropNetV5EDL, EDLLoss, dirichlet_to_predictions, training_step
)
from models.fusion_net_v5pro import FusionCropNetV5Pro
from models.fusion_net_v6 import FusionCropNetV6, v6_training_step
from data.datasets import FusionCropDatasetEDL, compute_metrics
from utils.calibration import calibration_report, print_calibration_report
from utils.interpretability import pixel_explanation_report, confusion_region_analysis

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _load_config():
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_checkpoint(save_dict, path):
    torch.save(save_dict, path)


def _load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def _build_checkpoint(model, optimizer, scheduler, scaler, epoch, phase,
                      best_miou, history, rng_states, extra=None):
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "phase": phase,
        "best_miou": best_miou,
        "history": history,
        "rng_states": rng_states,
        "extra": extra or {},
        "timestamp": datetime.datetime.now().isoformat(),
    }


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    if state["python"]:
        random.setstate(state["python"])
    if state["numpy"]:
        np.random.set_state(state["numpy"])
    if state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _log_gpu_memory(tag="", writer=None, step=0):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  GPU内存 [{tag}] - 已分配:{allocated:.2f}GB | 保留:{reserved:.2f}GB | 峰值:{peak:.2f}GB")
    if writer:
        writer.add_scalar("GPU/allocated_GB", allocated, step)
        writer.add_scalar("GPU/reserved_GB", reserved, step)
        writer.add_scalar("GPU/peak_GB", peak, step)


def train_phase1_frozen_backbone(model, train_loader, val_loader,
                                  device, epochs=20, lr=1e-3,
                                  resume_path=None, writer=None,
                                  grad_clip_value=5.0, grad_accum_steps=4):
    print("\n" + "="*60)
    print("训练阶段1：冻结光学骨干（20 epochs）")
    print("="*60)

    model.opt_enc.backbone.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )

    return _run_training(
        model, train_loader, val_loader, optimizer, device,
        epochs=epochs, phase=1, resume_path=resume_path, writer=writer,
        grad_clip_value=grad_clip_value, grad_accum_steps=grad_accum_steps,
        ckpt_prefix="best_phase1_edl"
    )


def train_phase2_full_finetune(model, train_loader, val_loader,
                               device, epochs=60, lr=3e-4,
                               best_ckpt_path="best_phase1_edl.pth",
                               resume_path=None, writer=None,
                               grad_clip_value=5.0, grad_accum_steps=4):
    print("\n" + "="*60)
    print("训练阶段2：全量Fine-tune（分层学习率）")
    print("="*60)

    if not resume_path and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        print(f" ✓ 加载阶段1最优权重: {best_ckpt_path}")

    model.opt_enc.backbone.requires_grad_(True)

    backbone_params = list(model.opt_enc.backbone.parameters())
    other_params = [p for p in model.parameters() if p not in backbone_params]

    param_groups = [
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": other_params, "lr": lr}
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-4)

    return _run_training(
        model, train_loader, val_loader, optimizer, device,
        epochs=epochs, phase=2, resume_path=resume_path, writer=writer,
        grad_clip_value=grad_clip_value, grad_accum_steps=grad_accum_steps,
        ckpt_prefix="best_phase2_edl"
    )


def _run_training(model, train_loader, val_loader,
                  optimizer, device, epochs, phase,
                  resume_path=None, writer=None,
                  grad_clip_value=5.0, grad_accum_steps=4,
                  ckpt_prefix="checkpoint"):
    criterion = model.edl_loss_fn
    use_amp = (device.type == "cuda")
    scaler = GradScaler() if use_amp else None

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[pg["lr"] for pg in optimizer.param_groups],
        total_steps=len(train_loader) * epochs,
        pct_start=0.1
    )

    start_epoch = 1
    best_miou = 0.0
    history = {"train_loss": [], "val_miou": [], "val_oa": [],
               "vacuity": [], "dissonance": [],
               "ece": [], "nll": [], "brier": [], "auroc": []}
    cal_interval = max(1, epochs // 10)

    # Resume logic
    if resume_path and os.path.exists(resume_path):
        print(f"\n{'='*60}")
        print(f"从检查点恢复训练: {resume_path}")
        print(f"{'='*60}")
        ckpt = _load_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") and scaler:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        if ckpt.get("history"):
            history = ckpt["history"]
        if ckpt.get("rng_states"):
            _restore_rng_state(ckpt["rng_states"])
        print(f" ✓ 恢复到 Epoch {start_epoch} (best mIoU={best_miou:.4f})")

    # TensorBoard
    tb_log_dir = "logs/tensorboard"
    if writer is None:
        writer = SummaryWriter(log_dir=os.path.join(
            tb_log_dir, f"phase{phase}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ))
    own_writer = (writer is not None)

    global_step_base = (start_epoch - 1) * len(train_loader)

    _log_gpu_memory(f"Phase{phase} start", writer, 0)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_edl_loss = 0.0
        total_ndvi_loss = 0.0
        total_lai_loss = 0.0
        total_growth_loss = 0.0
        total_boundary_loss = 0.0
        total_consist_loss = 0.0
        step_count = 0
        batch_count = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            global_step = global_step_base + batch_idx + 1

            if use_amp:
                with autocast():
                    loss, loss_dict = train_fn(
                        model, {k: v.to(device) if torch.is_tensor(v) else v
                                for k, v in batch.items()},
                        edl_loss_fn=criterion, epoch=epoch
                    )
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                loss, loss_dict = train_fn(
                    model, {k: v.to(device) if torch.is_tensor(v) else v
                            for k, v in batch.items()},
                    edl_loss_fn=criterion, epoch=epoch
                )
                loss = loss / grad_accum_steps
                loss.backward()

            total_loss += loss.item() * grad_accum_steps
            total_edl_loss += loss_dict.get('edl_loss', 0)
            total_ndvi_loss += loss_dict.get('ndvi_loss', 0)
            total_lai_loss += loss_dict.get('lai_loss', 0)
            total_growth_loss += loss_dict.get('growth_loss', 0)
            total_boundary_loss += loss_dict.get('boundary_loss', 0)
            total_consist_loss += loss_dict.get('consist', 0)
            batch_count += 1

            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_value
                )
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

                # Log gradient norm to TensorBoard
                if writer:
                    writer.add_scalar("Train/grad_norm", grad_norm.item(), global_step)
                    current_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("Train/lr", current_lr, global_step)

        avg_loss = total_loss / len(train_loader)
        avg_edl_loss = total_edl_loss / len(train_loader)
        avg_ndvi_loss = total_ndvi_loss / len(train_loader)
        avg_lai_loss = total_lai_loss / len(train_loader)
        avg_growth_loss = total_growth_loss / len(train_loader)
        avg_boundary_loss = total_boundary_loss / len(train_loader)
        avg_consist_loss = total_consist_loss / len(train_loader)

        # TensorBoard: training scalars
        if writer:
            epoch_step = epoch
            writer.add_scalar("Train/total_loss", avg_loss, epoch_step)
            writer.add_scalar("Train/edl_loss", avg_edl_loss, epoch_step)
            writer.add_scalar("Train/ndvi_loss", avg_ndvi_loss, epoch_step)
            writer.add_scalar("Train/lai_loss", avg_lai_loss, epoch_step)
            writer.add_scalar("Train/growth_loss", avg_growth_loss, epoch_step)
            writer.add_scalar("Train/boundary_loss", avg_boundary_loss, epoch_step)
            writer.add_scalar("Train/consist_loss", avg_consist_loss, epoch_step)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        all_alpha = []
        total_vacuity = []
        total_dissonance = []

        with torch.no_grad():
            for batch in val_loader:
                opt = batch["opt"].to(device)
                sar = batch["sar"].to(device)
                dem = batch["dem"].to(device)
                doy = batch["doy"].to(device)
                y = batch["y"]

                if use_amp:
                    with autocast():
                        alpha = model(opt, sar, dem, doy)
                        preds = dirichlet_to_predictions(alpha)
                else:
                    alpha = model(opt, sar, dem, doy)
                    preds = dirichlet_to_predictions(alpha)

                all_preds.append(preds["pred_class"].cpu())
                all_labels.append(y)
                all_alpha.append(alpha.cpu().numpy())
                total_vacuity.append(preds["vacuity"].cpu())
                total_dissonance.append(preds["dissonance"].cpu())

        metrics = compute_metrics(
            torch.cat(all_preds), torch.cat(all_labels), num_classes=7
        )

        avg_vacuity = torch.cat(total_vacuity).mean().item()
        avg_dissonance = torch.cat(total_dissonance).mean().item()

        history["train_loss"].append(avg_loss)
        history["val_miou"].append(metrics["mIoU"])
        history["val_oa"].append(metrics["OA"])
        history["vacuity"].append(avg_vacuity)
        history["dissonance"].append(avg_dissonance)

        # TensorBoard: validation scalars
        if writer:
            writer.add_scalar("Val/mIoU", metrics["mIoU"], epoch)
            writer.add_scalar("Val/OA", metrics["OA"], epoch)
            writer.add_scalar("Val/vacuity", avg_vacuity, epoch)
            writer.add_scalar("Val/dissonance", avg_dissonance, epoch)

        # Log model weight histograms periodically
        if writer and epoch % cal_interval == 0:
            for name, param in model.named_parameters():
                if param.requires_grad and param.numel() > 0:
                    writer.add_histogram(f"Weights/{name}", param.data, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        # Periodic calibration validation
        if epoch % cal_interval == 0 or epoch == epochs:
            alpha_cat = np.concatenate(all_alpha, axis=0)
            labels_cat = torch.cat(all_labels).numpy()
            cal = calibration_report(alpha_cat, labels_cat, num_classes=7, n_bins=10)
            history["ece"].append(cal["ECE"])
            history["nll"].append(cal["NLL"])
            history["brier"].append(cal["Brier"])
            history["auroc"].append(cal["AUROC_error_detection"])
            print(f"  校准指标 - ECE:{cal['ECE']:.4f} NLL:{cal['NLL']:.4f} "
                  f"Brier:{cal['Brier']:.4f} AUROC(err):{cal['AUROC_error_detection']:.4f}")
            if writer:
                writer.add_scalar("Cal/ECE", cal["ECE"], epoch)
                writer.add_scalar("Cal/NLL", cal["NLL"], epoch)
                writer.add_scalar("Cal/Brier", cal["Brier"], epoch)
                writer.add_scalar("Cal/AUROC_error", cal["AUROC_error_detection"], epoch)

        class_names = list(FusionCropDatasetEDL.CROP_CLASSES.values())[1:]
        iou_str = " | ".join([f"{n}:{v:.3f}"
                              for n, v in zip(class_names,
                                              metrics["IoU_per_class"])])

        print(f"[P{phase}] Epoch {epoch:3d}/{epochs} | "
              f"Loss:{avg_loss:.4f} | EDL:{avg_edl_loss:.4f} | "
              f"NDVI:{avg_ndvi_loss:.4f} | LAI:{avg_lai_loss:.4f} | "
              f"Growth:{avg_growth_loss:.4f} | Boundary:{avg_boundary_loss:.4f} | "
              f"Consist:{avg_consist_loss:.4f} | "
              f"mIoU:{metrics['mIoU']:.4f} | OA:{metrics['OA']:.4f}")
        print(f"  不确定性 - Vacuity:{avg_vacuity:.4f} | Dissonance:{avg_dissonance:.4f}")
        print(f"  {iou_str}")

        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_prefix}.pth")
            rng_states = _capture_rng_state()
            save_dict = _build_checkpoint(
                model, optimizer, scheduler, scaler, epoch, phase,
                best_miou, history, rng_states,
                extra={"mIoU": metrics["mIoU"], "OA": metrics["OA"]}
            )
            _save_checkpoint(save_dict, ckpt_path)
            print(f"  ✓ 保存最优模型: {ckpt_path}")

        # Save periodic checkpoint every 10 epochs for safety
        if epoch % 10 == 0:
            periodic_path = os.path.join(CHECKPOINT_DIR,
                                         f"{ckpt_prefix}_epoch{epoch}.pth")
            rng_states = _capture_rng_state()
            save_dict = _build_checkpoint(
                model, optimizer, scheduler, scaler, epoch, phase,
                best_miou, history, rng_states,
                extra={"mIoU": metrics["mIoU"], "OA": metrics["OA"]}
            )
            _save_checkpoint(save_dict, periodic_path)
            print(f"  ✓ 周期性检查点: {periodic_path}")

        # Memory optimization: clear cache after each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if epoch % cal_interval == 0:
            _log_gpu_memory(f"Epoch {epoch}", writer, epoch)

    # Save training history as JSON
    history_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_prefix}_history.json")
    serializable_history = {}
    for k, v in history.items():
        if isinstance(v, list) and len(v) > 0:
            if hasattr(v[0], 'item'):
                serializable_history[k] = [float(x) for x in v]
            else:
                serializable_history[k] = v
        else:
            serializable_history[k] = v
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 训练历史已保存: {history_path}")

    # Close writer if we own it
    if own_writer:
        writer.close()

    return history


def test_uncertainty_inference(model, test_loader, device, n_passes=5):
    print("\n" + "="*60)
    print("不确定性推理测试")
    print("="*60)

    model.eval()
    total_alpha = []
    total_probs = []
    total_vacuity = []
    total_dissonance = []
    total_class_var = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            opt = batch["opt"].to(device)
            sar = batch["sar"].to(device)
            dem = batch["dem"].to(device)
            doy = batch["doy"].to(device)

            result = model.predict_uncertainty(
                opt, sar, dem, doy,
                n_passes=n_passes,
                use_tta=True
            )

            total_alpha.append(result["alpha_fused"].cpu().numpy())
            total_probs.append(result["probs"].cpu())
            total_vacuity.append(result["vacuity"].cpu())
            total_dissonance.append(result["dissonance"].cpu())
            total_class_var.append(result["class_var"].cpu())
            all_labels.append(batch["y"])

    avg_vacuity = torch.cat(total_vacuity).mean().item()
    avg_dissonance = torch.cat(total_dissonance).mean().item()
    avg_class_var = torch.cat(total_class_var).mean().item()

    print(f"测试集不确定性统计:")
    print(f"  Vacuity (aleatoric): {avg_vacuity:.4f}")
    print(f"  Dissonance (epistemic): {avg_dissonance:.4f}")
    print(f"  Per-class variance: {avg_class_var:.6f}")

    print("\n" + "="*60)
    print("EDL校准验证报告")
    print("="*60)
    alpha_cat = np.concatenate(total_alpha, axis=0)
    labels_cat = torch.cat(all_labels).numpy()
    cal = calibration_report(alpha_cat, labels_cat, num_classes=7, n_bins=15)
    print_calibration_report(cal)

    px_report = pixel_explanation_report(alpha_cat, labels_cat, num_classes=7)
    print("\n像素级解释 (正确 vs 错误):")
    for kind in ["correct", "incorrect"]:
        r = px_report[kind]
        print(f"  {kind} (n={r['n']}): vacuity={r['vacuity']['mean']:.4f}±{r['vacuity']['std']:.4f}, "
              f"margin={r['margin']['mean']:.4f}, entropy={r['entropy']['mean']:.4f}")

    confusion = confusion_region_analysis(alpha_cat, labels_cat, num_classes=7)
    top_confused = sorted(confusion.items(), key=lambda x: x[1]["n"], reverse=True)[:5]
    print("\nTop-5 混淆类别对:")
    for pair_name, info in top_confused:
        print(f"  {pair_name}: n={info['n']}, vac={info.get('mean_vacuity', 'N/A')}")

    output_dir = "calibration_output"
    os.makedirs(output_dir, exist_ok=True)
    serializable = {k: v for k, v in cal.items() if k != "_raw"}
    with open(f"{output_dir}/final_calibration.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\n校准报告已保存至: {output_dir}/final_calibration.json")

    return {
        "vacuity": avg_vacuity,
        "dissonance": avg_dissonance,
        "class_var": avg_class_var,
        "calibration": cal
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    # CUDA optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM总量: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB")
        _log_gpu_memory("启动时")

    cfg = _load_config()
    training_cfg = cfg.get("training", {})

    # Config-driven hyperparams
    grad_clip_value = float(training_cfg.get("grad_clip_value", 5.0))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 4))
    use_amp = training_cfg.get("use_amp", True)

    # Resume flag
    resume_path = None
    resume_phase = 1
    if "--resume" in sys.argv:
        try:
            idx = sys.argv.index("--resume")
            resume_path = sys.argv[idx + 1]
        except IndexError:
            pass
    if not resume_path:
        resume_path = training_cfg.get("resume_from", "") or None
    if resume_path:
        resume_phase = int(training_cfg.get("resume_phase", 1))
        print(f"恢复模式: phase={resume_phase}, path={resume_path}")

    opt_seq = np.load("data/processed/opt_sequence.npy") if os.path.exists("data/processed/opt_sequence.npy") else np.random.randn(100, 10, 256, 256).astype(np.float32)
    sar_seq = np.load("data/processed/sar_sequence.npy") if os.path.exists("data/processed/sar_sequence.npy") else np.random.randn(100, 5, 256, 256).astype(np.float32)
    doy_norm = np.load("data/processed/doy_norm.npy") if os.path.exists("data/processed/doy_norm.npy") else np.linspace(0, 1, 100).astype(np.float32)

    if os.path.exists("data/processed/final_label.npy"):
        label = np.load("data/processed/final_label.npy")
        print(" ✓ 使用融合版预处理的精细标签")
    elif os.path.exists("data/processed/label.npy"):
        label = np.load("data/processed/label.npy")
        print(" ✓ 使用标准标签")
    else:
        label = np.random.randint(0, 7, (256, 256)).astype(np.int64)
        print(" ✓ 使用随机生成的测试标签")

    dem_data = np.load("data/processed/dem.npy") if os.path.exists("data/processed/dem.npy") else None

    print(f"数据加载完成:")
    print(f"  光学时序: {opt_seq.shape}")
    print(f"  SAR时序: {sar_seq.shape}")
    print(f"  DOY: {doy_norm.shape}")
    print(f"  标签: {label.shape}")
    print(f"  DEM: {'已加载' if dem_data is not None else '未加载'}")
    print(f"  有效类别数: {len(np.unique(label)) - (1 if 255 in np.unique(label) else 0)}")

    use_spatial_split = False
    if use_spatial_split and os.path.exists("data/processed/train_mask.npy"):
        print("\n使用空间独立划分")
        train_mask = np.load("data/processed/train_mask.npy")
        val_mask = np.load("data/processed/val_mask.npy")

        train_dataset = FusionCropDatasetEDL(opt_seq, sar_seq, doy_norm, label,
                                             patch_size=32, augment=True,
                                             mask=train_mask, dem_data=dem_data)
        val_dataset = FusionCropDatasetEDL(opt_seq, sar_seq, doy_norm, label,
                                           patch_size=32, augment=False,
                                           mask=val_mask, dem_data=dem_data)
    else:
        print("\n使用随机划分")
        dataset = FusionCropDatasetEDL(opt_seq, sar_seq, doy_norm, label,
                                       patch_size=32, augment=True,
                                       dem_data=dem_data)

        n_val = int(len(dataset) * 0.15)
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    use_v5pro = "--v5pro" in sys.argv
    use_v6 = "--v6" in sys.argv
    backbone_name = "resnet50"
    if "--backbone" in sys.argv:
        try:
            backbone_name = sys.argv[sys.argv.index("--backbone") + 1]
        except IndexError:
            pass
    rs_weights_path = None
    if "--rs_weights" in sys.argv:
        try:
            rs_weights_path = sys.argv[sys.argv.index("--rs_weights") + 1]
        except IndexError:
            pass

    train_fn = training_step
    if use_v6:
        model = FusionCropNetV6(
            opt_ch=opt_seq.shape[1] if len(opt_seq.shape) > 3 else 10,
            sar_ch=sar_seq.shape[1] if len(sar_seq.shape) > 3 else 5,
            dem_ch_in=5, num_classes=7, feat_dim=512,
            backbone=backbone_name, pretrained=True,
            n_heads=16, win_size=4, n_layers=4,
            drop_timestep_p=0.1, edl_dropout_p=0.3,
            edl_lambda_max=0.5, edl_anneal_ep=50,
            modality_dropout_p=0.1, use_gradient_checkpointing=True,
            rs_weights=rs_weights_path,
        ).to(device)
        train_fn = v6_training_step
        print(f"使用 V6 (backbone={backbone_name}, use_v6_enhancements=True)")
    elif use_v5pro:
        model = FusionCropNetV5Pro(
            opt_ch=opt_seq.shape[1] if len(opt_seq.shape) > 3 else 10,
            sar_ch=sar_seq.shape[1] if len(sar_seq.shape) > 3 else 5,
            dem_ch_in=5, num_classes=7, feat_dim=512,
            backbone=backbone_name, pretrained=True,
            n_heads=16, win_size=4, n_layers=4,
            drop_timestep_p=0.1, edl_dropout_p=0.3,
            edl_lambda_max=0.5, edl_anneal_ep=50,
            use_carafe=True, dynamic_dropout=True, adaptive_kl=True,
            rs_weights_path=rs_weights_path,
        ).to(device)
        print(f"使用 V5Pro (backbone={backbone_name})")
    else:
        model = FusionCropNetV5EDL(
            opt_ch=opt_seq.shape[1] if len(opt_seq.shape) > 3 else 10,
            sar_ch=sar_seq.shape[1] if len(sar_seq.shape) > 3 else 5,
            dem_ch_in=5,
            num_classes=7, feat_dim=512,
            backbone="resnet50", pretrained=True,
            n_heads=16, win_size=4, n_layers=4,
            drop_timestep_p=0.1, edl_dropout_p=0.3,
            edl_lambda_max=0.5, edl_anneal_ep=50,
            rs_weights_path=rs_weights_path,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params/1e6:.2f}M")

    # Shared TensorBoard writer across both phases
    tb_log_dir = training_cfg.get("tb_log_dir", "logs/tensorboard")
    writer = SummaryWriter(log_dir=os.path.join(
        tb_log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    print(f"TensorBoard 日志目录: {writer.log_dir}")

    # Phase routing based on resume
    if resume_path and resume_phase == 2:
        print("从阶段2检查点恢复，跳过阶段1")
        history_p1 = {}
        history_p2 = train_phase2_full_finetune(
            model, train_loader, val_loader, device,
            epochs=60, lr=3e-4,
            resume_path=resume_path, writer=writer,
            grad_clip_value=grad_clip_value,
            grad_accum_steps=grad_accum_steps
        )
    else:
        history_p1 = train_phase1_frozen_backbone(
            model, train_loader, val_loader, device,
            epochs=20, lr=1e-3,
            resume_path=resume_path if resume_phase == 1 else None,
            writer=writer,
            grad_clip_value=grad_clip_value,
            grad_accum_steps=grad_accum_steps
        )
        history_p2 = train_phase2_full_finetune(
            model, train_loader, val_loader, device,
            epochs=60, lr=3e-4,
            resume_path=resume_path if resume_phase == 2 else None,
            writer=writer,
            grad_clip_value=grad_clip_value,
            grad_accum_steps=grad_accum_steps
        )

    writer.close()

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

    # Final memory report
    _log_gpu_memory("训练结束")

    test_uncertainty_inference(model, val_loader, device, n_passes=5)

    print("\n最优模型保存为:")
    print(f"  - {CHECKPOINT_DIR}/best_phase1_edl.pth (阶段1最优)")
    print(f"  - {CHECKPOINT_DIR}/best_phase2_edl.pth (阶段2最优)")
    print(f"  - {CHECKPOINT_DIR}/best_phase1_edl_history.json (阶段1历史)")
    print(f"  - {CHECKPOINT_DIR}/best_phase2_edl_history.json (阶段2历史)")
    print(f"\nTensorBoard: tensorboard --logdir {writer.log_dir}")
