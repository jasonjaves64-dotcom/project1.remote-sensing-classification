"""
使用虚拟数据训练模型并绘制 loss 曲线，训练完成后自动清理数据
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import shutil
import time
import argparse
from datetime import datetime

sys.path.insert(0, '.')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, will use text plot")

def generate_dummy_data(output_dir="dummy_data"):
    """生成虚拟训练数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = 100
    T, H, W = 12, 32, 32
    
    opt_seq = np.random.randn(num_samples, T, 10, H, W).astype(np.float32) * 0.1
    sar_seq = np.random.randn(num_samples, T, 5, H, W).astype(np.float32) * 0.1
    dem = np.random.randn(num_samples, 5, H, W).astype(np.float32) * 0.1
    doy = np.random.rand(num_samples, T).astype(np.float32)
    labels = np.random.randint(0, 7, (num_samples, H, W)).astype(np.int64)
    
    np.save(os.path.join(output_dir, "opt_seq.npy"), opt_seq)
    np.save(os.path.join(output_dir, "sar_seq.npy"), sar_seq)
    np.save(os.path.join(output_dir, "dem.npy"), dem)
    np.save(os.path.join(output_dir, "doy.npy"), doy)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    
    print("Dummy data generated to", output_dir+"/")
    print("  Samples:", num_samples)
    print("  Shape: T=%d, H=%d, W=%d" % (T, H, W))
    
    return output_dir

def load_dummy_data(data_dir="dummy_data"):
    """加载虚拟数据"""
    opt_seq = np.load(os.path.join(data_dir, "opt_seq.npy"))
    sar_seq = np.load(os.path.join(data_dir, "sar_seq.npy"))
    dem = np.load(os.path.join(data_dir, "dem.npy"))
    doy = np.load(os.path.join(data_dir, "doy.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    
    return opt_seq, sar_seq, dem, doy, labels

def train_with_dummy_data(use_edl=False):
    """使用虚拟数据训练模型"""
    if use_edl:
        from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
    else:
        from models.fusion_net_v5 import FusionCropNetV5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)
    print("EDL Mode:", use_edl)
    
    if use_edl:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=8, win_size=4, n_layers=2,
            edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=10
        ).to(device)
    else:
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=8, win_size=4, n_layers=2,
            drop_timestep_p=0.1
        ).to(device)
    
    opt_seq, sar_seq, dem, doy, labels = load_dummy_data()
    
    indices = np.random.permutation(len(opt_seq))
    train_idx = indices[:int(0.8*len(opt_seq))]
    val_idx = indices[int(0.8*len(opt_seq)):]
    
    train_opt = torch.from_numpy(opt_seq[train_idx])
    train_sar = torch.from_numpy(sar_seq[train_idx])
    train_dem = torch.from_numpy(dem[train_idx])
    train_doy = torch.from_numpy(doy[train_idx])
    train_labels = torch.from_numpy(labels[train_idx])
    
    val_opt = torch.from_numpy(opt_seq[val_idx])
    val_sar = torch.from_numpy(sar_seq[val_idx])
    val_dem = torch.from_numpy(dem[val_idx])
    val_doy = torch.from_numpy(doy[val_idx])
    val_labels = torch.from_numpy(labels[val_idx])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 20
    batch_size = 4
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    if use_edl:
        train_vacuity_history = []
        val_vacuity_history = []
        val_ece_history = []
        val_nll_history = []
    
    print("\n=== Starting Training ===")
    print("Epochs:", num_epochs)
    print("Batch size:", batch_size)
    print("Train samples:", len(train_idx))
    print("Val samples:", len(val_idx))
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        perm = torch.randperm(len(train_opt))
        train_opt = train_opt[perm]
        train_sar = train_sar[perm]
        train_dem = train_dem[perm]
        train_doy = train_doy[perm]
        train_labels = train_labels[perm]
        
        for i in range(0, len(train_opt), batch_size):
            opt_batch = train_opt[i:i+batch_size].to(device)
            sar_batch = train_sar[i:i+batch_size].to(device)
            dem_batch = train_dem[i:i+batch_size].to(device)
            doy_batch = train_doy[i:i+batch_size].to(device)
            label_batch = train_labels[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            if use_edl:
                alpha, _, _ = model(opt_batch, sar_batch, dem_batch, doy_batch, epoch=epoch)
                preds = dirichlet_to_predictions(alpha)
                logits = preds['probs']
            else:
                logits = model(opt_batch, sar_batch, dem_batch, doy_batch)
            
            loss = criterion(logits, label_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(opt_batch)
            preds = logits.argmax(dim=1)
            train_correct += (preds == label_batch).sum().item()
            train_total += label_batch.numel()
        
        avg_train_loss = train_loss / len(train_opt)
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_vacuity_sum = 0.0
        val_vacuity_count = 0
        
        with torch.no_grad():
            for i in range(0, len(val_opt), batch_size):
                opt_batch = val_opt[i:i+batch_size].to(device)
                sar_batch = val_sar[i:i+batch_size].to(device)
                dem_batch = val_dem[i:i+batch_size].to(device)
                doy_batch = val_doy[i:i+batch_size].to(device)
                label_batch = val_labels[i:i+batch_size].to(device)
                
                if use_edl:
                    alpha = model(opt_batch, sar_batch, dem_batch, doy_batch)
                    preds = dirichlet_to_predictions(alpha)
                    logits = preds['probs']
                    val_vacuity_sum += preds['vacuity'].sum().item()
                    val_vacuity_count += preds['vacuity'].numel()
                else:
                    logits = model(opt_batch, sar_batch, dem_batch, doy_batch)
                
                loss = criterion(logits, label_batch)
                
                val_loss += loss.item() * len(opt_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == label_batch).sum().item()
                val_total += label_batch.numel()
        
        avg_val_loss = val_loss / len(val_opt)
        val_acc = val_correct / val_total
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Periodic calibration check (every 5 epochs)
        if use_edl and (epoch % 5 == 0 or epoch == num_epochs - 1):
            from utils.calibration import calibration_report
            val_alpha_all = []
            val_labels_all = []
            with torch.no_grad():
                for i in range(0, len(val_opt), batch_size):
                    opt_b = val_opt[i:i+batch_size].to(device)
                    sar_b = val_sar[i:i+batch_size].to(device)
                    dem_b = val_dem[i:i+batch_size].to(device)
                    doy_b = val_doy[i:i+batch_size].to(device)
                    a = model(opt_b, sar_b, dem_b, doy_b)
                    val_alpha_all.append(a.cpu().numpy())
                    val_labels_all.append(val_labels[i:i+batch_size])
            alpha_cat = np.concatenate(val_alpha_all, axis=0)
            labels_cat = torch.cat(val_labels_all).numpy()
            cal = calibration_report(alpha_cat, labels_cat, num_classes=7, n_bins=10)
            val_ece_history.append(cal["ECE"])
            val_nll_history.append(cal["NLL"])

        if use_edl:
            avg_val_vacuity = val_vacuity_sum / val_vacuity_count
            val_vacuity_history.append(avg_val_vacuity)
            print(f"Epoch {epoch+1:2d}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | "
                  f"Val Acc: {val_acc*100:.1f}% | "
                  f"Val Vacuity: {avg_val_vacuity:.4f}")
        else:
            print(f"Epoch {epoch+1:2d}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | "
                  f"Val Acc: {val_acc*100:.1f}%")
    
    result = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history
    }
    
    if use_edl:
        result["val_vacuity"] = val_vacuity_history
        result["val_ece"] = val_ece_history
        result["val_nll"] = val_nll_history

    return result

def plot_loss_curves_text(history):
    """文本绘制 loss 曲线"""
    print("\n" + "="*60)
    print("Loss Curve (Text Version)")
    print("="*60)
    
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    
    max_loss = max(max(train_loss), max(val_loss))
    min_loss = min(min(train_loss), min(val_loss))
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
    
    print("\n1. Loss Plot:")
    print("   Epoch | Train Loss | Val Loss | Plot")
    print("   " + "-"*50)
    
    for i, (tl, vl) in enumerate(zip(train_loss, val_loss)):
        tl_norm = (tl - min_loss) / loss_range
        vl_norm = (vl - min_loss) / loss_range
        
        tl_bar = "#" * int(tl_norm * 20)
        vl_bar = "*" * int(vl_norm * 20)
        
        print(f"   {i+1:4d} | {tl:.4f}    | {vl:.4f}    | {tl_bar} {vl_bar}")
    
    print("\n   Legend: Train Loss=#, Val Loss=*")
    
    print("\n2. Accuracy Plot:")
    print("   Epoch | Train Acc | Val Acc | Plot")
    print("   " + "-"*50)
    
    for i, (ta, va) in enumerate(zip(train_acc, val_acc)):
        ta_bar = "#" * int(ta * 20)
        va_bar = "*" * int(va * 20)
        
        print(f"   {i+1:4d} | {ta*100:6.1f}%  | {va*100:6.1f}%  | {ta_bar} {va_bar}")
    
    print("\n   Legend: Train Acc=#, Val Acc=*")
    
    if "val_vacuity" in history:
        print("\n3. Val Vacuity:")
        print("   Epoch | Vacuity")
        print("   " + "-"*20)
        for i, v in enumerate(history["val_vacuity"]):
            print(f"   {i+1:4d} | {v:.4f}")
    
    print("\n" + "="*60)

def plot_loss_curves(history, save_path="loss_plot.png"):
    """绘制 loss 曲线（含校准指标）"""
    if HAS_MATPLOTLIB:
        has_cal = "val_ece" in history and len(history.get("val_ece", [])) > 0
        has_vac = "val_vacuity" in history

        # Determine grid layout
        if has_cal and has_vac:
            n_rows, n_cols = 2, 3
        elif has_vac:
            n_rows, n_cols = 1, 3
        else:
            n_rows, n_cols = 1, 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        axes[0].plot(history["train_loss"], label="Train Loss", color="blue")
        axes[0].plot(history["val_loss"], label="Val Loss", color="red")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["train_acc"], label="Train Acc", color="blue")
        axes[1].plot(history["val_acc"], label="Val Acc", color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        ax_idx = 2
        if has_vac:
            axes[ax_idx].plot(history["val_vacuity"], label="Val Vacuity", color="green")
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel("Vacuity")
            axes[ax_idx].set_title("Validation Vacuity (Data Uncertainty)")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

        if has_cal:
            cal_epochs = list(range(0, len(history["train_loss"]), 5))[:len(history["val_ece"])]
            axes[ax_idx].plot(cal_epochs, history["val_ece"], 'o-', color='#FF9800', label="ECE")
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel("ECE")
            axes[ax_idx].set_title("Expected Calibration Error")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1
            if "val_nll" in history:
                axes[ax_idx].plot(cal_epochs, history["val_nll"], 's-', color='#9C27B0', label="NLL")
                axes[ax_idx].set_xlabel("Epoch")
                axes[ax_idx].set_ylabel("NLL")
                axes[ax_idx].set_title("Negative Log-Likelihood (Dirichlet)")
                axes[ax_idx].legend()
                axes[ax_idx].grid(True, alpha=0.3)
                ax_idx += 1

        # Hide unused axes
        for i in range(ax_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print("\nLoss plot saved to:", save_path)

        plt.close()
    else:
        plot_loss_curves_text(history)

def cleanup_dummy_data(data_dir="dummy_data"):
    """清理虚拟数据"""
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("\nDummy data cleaned:", data_dir+"/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edl", action="store_true", help="使用 EDL 模型")
    args = parser.parse_args()
    
    use_edl = args.edl
    
    print("=" * 60)
    print("Train FusionCropNetV5 with Dummy Data" + (" (EDL Mode)" if use_edl else ""))
    print("=" * 60)
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        generate_dummy_data()
        history = train_with_dummy_data(use_edl=use_edl)
        plot_loss_curves(history, save_path="loss_plot_edl.png" if use_edl else "loss_plot.png")
    finally:
        cleanup_dummy_data()
    
    print("\n" + "=" * 60)
    print("Training completed, data cleaned")
    print("=" * 60)

if __name__ == "__main__":
    main()