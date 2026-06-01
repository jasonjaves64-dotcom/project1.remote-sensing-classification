import torch
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from models.fusion_net import FusionCropNet, PretrainedWeightManager

CROP_PALETTE = {
    0: ("#FFFFFF", "背景"),
    1: ("#FFD700", "冬小麦"),
    2: ("#228B22", "夏玉米"),
    3: ("#4682B4", "水稻"),
    4: ("#9ACD32", "大豆"),
    5: ("#FF8C00", "棉花"),
    6: ("#A9A9A9", "其他作物"),
}

def sliding_window_predict(model, opt_seq, sar_seq, doy_norm,
                          patch_size=32, overlap=0.5, batch_size=16,
                          device="cuda", return_attn=False):
    model.eval()
    T, C_opt, H, W = opt_seq.shape
    num_classes = 7
    
    pred_accum = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    attn_accum = np.zeros((T,), dtype=np.float32)
    attn_count = 0
    
    stride = int(patch_size * (1 - overlap))
    rows = list(range(0, H - patch_size + 1, stride)) + [H - patch_size]
    cols = list(range(0, W - patch_size + 1, stride)) + [W - patch_size]
    
    patches_opt, patches_sar, positions = [], [], []
    for r in set(rows):
        for c in set(cols):
            opt_p = opt_seq[:, :, r:r+patch_size, c:c+patch_size]
            sar_p = sar_seq[:, :, r:r+patch_size, c:c+patch_size]
            patches_opt.append(opt_p)
            patches_sar.append(sar_p)
            positions.append((r, c))
    
    doy_batch = torch.from_numpy(doy_norm).float().unsqueeze(0)
    
    for i in range(0, len(patches_opt), batch_size):
        batch_opt = np.stack(patches_opt[i:i+batch_size])
        batch_sar = np.stack(patches_sar[i:i+batch_size])
        batch_doy = doy_batch.expand(batch_opt.shape[0], -1)
        
        opt_t = torch.from_numpy(batch_opt).float().to(device)
        sar_t = torch.from_numpy(batch_sar).float().to(device)
        doy_t = batch_doy.to(device)
        
        with torch.no_grad():
            logits = model(opt_t, sar_t, doy_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            if return_attn and hasattr(model, 'attn_weights'):
                attn = model.attn_weights.squeeze(1).mean(0).cpu().numpy()
                attn_accum += attn
                attn_count += 1
        
        for j, (r, c) in enumerate(positions[i:i+batch_size]):
            pred_accum[:, r:r+patch_size, c:c+patch_size] += probs[j]
            count_map[r:r+patch_size, c:c+patch_size] += 1
    
    pred_accum /= np.maximum(count_map[np.newaxis], 1)
    pred_map = pred_accum.argmax(axis=0).astype(np.uint8)
    conf_map = pred_accum.max(axis=0)
    
    if return_attn and attn_count > 0:
        avg_attn = attn_accum / attn_count
        return pred_map, conf_map, avg_attn
    
    return pred_map, conf_map

def visualize_results(pred_map, conf_map, opt_seq,
                     doy_norm, attn_weights=None,
                     save_path="result.png"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("农作物遥感分类结果", fontsize=14, fontweight="bold")
    
    ndvi_band = opt_seq[:, 6, :, :]
    peak_t = ndvi_band.mean(axis=(1,2)).argmax()
    rgb = np.stack([
        opt_seq[peak_t, 2],
        opt_seq[peak_t, 1],
        opt_seq[peak_t, 0],
    ], axis=-1)
    rgb = np.clip(rgb / 0.3, 0, 1)
    axes[0].imshow(rgb)
    axes[0].set_title(f"真彩色 (DOY {int(doy_norm[peak_t]*365)})")
    axes[0].axis("off")
    
    cmap_colors = [CROP_PALETTE[i][0] for i in range(len(CROP_PALETTE))]
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = list(range(len(CROP_PALETTE) + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = axes[1].imshow(pred_map, cmap=cmap, norm=norm)
    axes[1].set_title("作物分类结果")
    axes[1].axis("off")
    
    patches = [
        plt.Rectangle((0,0),1,1, color=CROP_PALETTE[i][0],
                      label=CROP_PALETTE[i][1])
        for i in range(1, len(CROP_PALETTE))
    ]
    axes[1].legend(handles=patches, loc="lower right",
                   fontsize=7, framealpha=0.8)
    
    im3 = axes[2].imshow(conf_map, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axes[2].set_title("预测置信度")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    if attn_weights is not None:
        months = [f"M{i+1}" for i in range(len(doy_norm))]
        bars = axes[3].bar(range(len(attn_weights)), attn_weights,
                          color="steelblue", alpha=0.8)
        axes[3].set_xticks(range(len(attn_weights)))
        axes[3].set_xticklabels(months, rotation=45, fontsize=8)
        axes[3].set_title("时序注意力权重\n（模型关注的关键物候时相）")
        axes[3].set_ylabel("注意力强度")
        
        peak_idx = attn_weights.argmax()
        axes[3].bar(peak_idx, attn_weights[peak_idx],
                   color="crimson", alpha=0.9,
                   label=f"关键时相: {months[peak_idx]}")
        axes[3].legend(fontsize=8)
    else:
        axes[3].text(0.5, 0.5, "注意力权重\n不可用",
                    ha="center", va="center", transform=axes[3].transAxes)
        axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ 可视化图已保存: {save_path}")
    plt.show()

def save_geotiff(pred_map, reference_raster_path, output_path):
    with rasterio.open(reference_raster_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=255
        )
        
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred_map.astype(np.uint8), 1)
            
            colormap = {i: tuple(int(c.lstrip('#')[j:j+2], 16)
                                for j in (0,2,4)) + (255,)
                        for i, (c, _) in CROP_PALETTE.items()}
            dst.write_colormap(1, colormap)
    
    print(f"✓ GeoTIFF已保存: {output_path}")

def save_as_geotiff(pred_map, output_path, bounds, crs="EPSG:4326"):
    H, W = pred_map.shape
    transform = from_bounds(*bounds, W, H)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=H,
        width=W,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=255
    ) as dst:
        dst.write(pred_map.astype(np.uint8), 1)
        
        colormap = {i: tuple(int(c.lstrip('#')[j:j+2], 16)
                            for j in (0,2,4)) + (255,)
                    for i, (c, _) in CROP_PALETTE.items()}
        dst.write_colormap(1, colormap)
    
    print(f"✓ GeoTIFF已保存: {output_path}")

def print_accuracy_report(pred_map, label_map, num_classes=7):
    valid = label_map != 0
    p = pred_map[valid].flatten()
    l = label_map[valid].flatten()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, pr in zip(l, p):
        cm[int(t), int(pr)] += 1
    
    print("\n" + "="*70)
    print("精度评估报告")
    print("="*70)
    print(f"{'类别':<10} {'PA(召回率)':<12} {'UA(精确率)':<12} {'F1':<10} {'IoU':<10}")
    print("-"*70)
    
    class_names = [v[1] for v in CROP_PALETTE.values()]
    ious = []
    
    for cls in range(1, num_classes):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        
        pa = tp / (tp + fn + 1e-6)
        ua = tp / (tp + fp + 1e-6)
        f1 = 2 * pa * ua / (pa + ua + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        ious.append(iou)
        
        print(f"{class_names[cls]:<10} {pa:.4f} {ua:.4f} "
              f"{f1:.4f} {iou:.4f}")
    
    oa = np.diag(cm).sum() / cm.sum()
    miou = np.mean(ious)
    
    n = cm.sum()
    po = np.diag(cm).sum() / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n**2
    kappa = (po - pe) / (1 - pe + 1e-6)
    
    print("-"*70)
    print(f"总体精度 OA : {oa:.4f}")
    print(f"平均交并比 mIoU: {miou:.4f}")
    print(f"Kappa系数 : {kappa:.4f}")
    print("="*70)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device}")
    
    model = FusionCropNet(
        opt_channels=10, sar_channels=3,
        num_classes=7, feat_dim=256,
        backbone="resnet18", pretrained=False
    ).to(device)
    
    manager = PretrainedWeightManager(model)
    manager.load_checkpoint("best_phase2.pth")
    model.eval()
    
    opt_seq = np.load("data/processed/opt_sequence_2023.npy")
    sar_seq = np.load("data/processed/sar_sequence_2023.npy")
    doy_norm = np.load("data/processed/doy_norm_2023.npy")
    label = np.load("data/processed/label_2023.npy")
    
    print("开始推理...")
    pred_map, conf_map = sliding_window_predict(
        model, opt_seq, sar_seq, doy_norm,
        patch_size=32, overlap=0.5,
        batch_size=16, device=device
    )
    print(f"✓ 推理完成，输出尺寸: {pred_map.shape}")
    
    print_accuracy_report(pred_map, label)
    
    visualize_results(pred_map, conf_map, opt_seq, doy_norm,
                      save_path="output/crop_classification_result.png")
    
    Path("output").mkdir(parents=True, exist_ok=True)
    bounds = (115.0, 36.0, 117.0, 38.0)
    save_as_geotiff(pred_map, "output/crop_map_2023.tif", bounds)
    
    print("\n推理完成！输出文件:")
    print("  - output/crop_map_2023.tif (GeoTIFF格式分类图)")
    print("  - output/crop_classification_result.png (可视化结果)")

if __name__ == "__main__":
    main()