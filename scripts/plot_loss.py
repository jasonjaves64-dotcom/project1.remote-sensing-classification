"""
训练损失可视化脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def generate_synthetic_loss(n_epochs=80):
    """生成模拟的训练损失曲线"""
    # 基础训练损失（模拟下降趋势）
    base_loss = 2.5 * np.exp(-0.03 * np.arange(n_epochs)) + 0.3
    
    # 添加噪声
    noise = np.random.normal(0, 0.08, n_epochs)
    train_loss = base_loss + noise
    
    # 验证损失（略高于训练损失）
    val_loss = base_loss * 1.1 + np.random.normal(0, 0.05, n_epochs)
    
    # EDL损失组件（如果启用EDL）
    kl_loss = 0.5 * np.exp(-0.05 * np.arange(n_epochs)) + 0.02
    ce_loss = 1.5 * np.exp(-0.04 * np.arange(n_epochs)) + 0.2
    
    return train_loss, val_loss, kl_loss, ce_loss

def plot_loss_curves(train_loss, val_loss, kl_loss=None, ce_loss=None, save_path=None):
    """绘制损失曲线"""
    n_epochs = len(train_loss)
    epochs = np.arange(1, n_epochs + 1)
    
    plt.figure(figsize=(12, 6))
    
    # 主损失曲线
    plt.plot(epochs, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # EDL损失组件
    if kl_loss is not None:
        plt.plot(epochs, kl_loss, label='KL Loss', color='#2ca02c', linewidth=1.5, alpha=0.7)
    if ce_loss is not None:
        plt.plot(epochs, ce_loss, label='CE Loss', color='#d62728', linewidth=1.5, alpha=0.7)
    
    # 标注最佳验证点
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = val_loss[np.argmin(val_loss)]
    plt.scatter(best_epoch, best_val_loss, color='red', s=50, zorder=5, 
                label=f'Best Val Point (epoch {best_epoch}, loss={best_val_loss:.4f})')
    
    # 设置图表属性
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(1, n_epochs)
    plt.ylim(bottom=0)
    
    # 添加注释
    plt.annotate(f'Final Train Loss: {train_loss[-1]:.4f}', 
                 xy=(n_epochs, train_loss[-1]), 
                 xytext=(n_epochs-10, train_loss[-1]+0.1),
                 arrowprops=dict(arrowstyle='->'))
    
    plt.annotate(f'Final Val Loss: {val_loss[-1]:.4f}', 
                 xy=(n_epochs, val_loss[-1]), 
                 xytext=(n_epochs-10, val_loss[-1]+0.1),
                 arrowprops=dict(arrowstyle='->'))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {save_path}")
    
    plt.show()

def plot_loss_distribution(train_loss, val_loss, save_path=None):
    """绘制损失分布直方图"""
    plt.figure(figsize=(10, 5))
    
    plt.hist(train_loss, bins=20, alpha=0.5, label='Training Loss', color='#1f77b4')
    plt.hist(val_loss, bins=20, alpha=0.5, label='Validation Loss', color='#ff7f0e')
    
    plt.title('Loss Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Loss Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_dist.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_metrics(metrics, save_path=None):
    """绘制评估指标曲线"""
    n_epochs = len(metrics['train_miou'])
    epochs = np.arange(1, n_epochs + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, metrics['train_miou'], label='Train mIoU', color='#1f77b4', linewidth=2)
    plt.plot(epochs, metrics['val_miou'], label='Val mIoU', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.plot(epochs, metrics['train_oa'], label='Train OA', color='#2ca02c', linewidth=1.5)
    plt.plot(epochs, metrics['val_oa'], label='Val OA', color='#d62728', linewidth=1.5, linestyle='--')
    
    plt.title('Model Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(1, n_epochs)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_metrics.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_metrics(n_epochs=80):
    """生成模拟的评估指标"""
    base_miou = 0.3 + 0.6 * (1 - np.exp(-0.04 * np.arange(n_epochs)))
    base_oa = 0.4 + 0.55 * (1 - np.exp(-0.03 * np.arange(n_epochs)))
    
    return {
        'train_miou': base_miou + np.random.normal(0, 0.02, n_epochs),
        'val_miou': base_miou * 0.95 + np.random.normal(0, 0.015, n_epochs),
        'train_oa': base_oa + np.random.normal(0, 0.015, n_epochs),
        'val_oa': base_oa * 0.95 + np.random.normal(0, 0.01, n_epochs)
    }

def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    parser.add_argument("--edl", action="store_true", help="Show EDL loss components")
    parser.add_argument("--metrics", action="store_true", help="Show evaluation metrics")
    args = parser.parse_args()
    
    # Generate synthetic data
    train_loss, val_loss, kl_loss, ce_loss = generate_synthetic_loss(args.epochs)
    
    # Plot loss curves
    edl_losses = (kl_loss, ce_loss) if args.edl else (None, None)
    save_path = args.save if args.save else 'loss_plot.png'
    plot_loss_curves(train_loss, val_loss, *edl_losses, save_path=save_path)
    
    # Plot loss distribution
    plot_loss_distribution(train_loss, val_loss, save_path=save_path)
    
    # Plot metrics
    if args.metrics:
        metrics = generate_metrics(args.epochs)
        plot_metrics(metrics, save_path=save_path)

if __name__ == "__main__":
    main()
