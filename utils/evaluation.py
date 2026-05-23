# =============================================================================
# utils/evaluation.py
# 模型验证策略模块
# Module B : 空间独立验证 + 跨年度泛化验证
# =============================================================================
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class ValidationStrategy:
    CROP_CLASSES = {
        0:"背景", 1:"冬小麦", 2:"夏玉米",
        3:"水稻", 4:"大豆", 5:"棉花", 6:"其他"
    }

    def spatial_kfold_split(self, label_map: np.ndarray, k: int = 5,
                            block_size_px: int = 64, seed: int = 42
                            ) -> list[dict]:
        """生成K折空间交叉验证划分方案"""
        H, W = label_map.shape
        bs = block_size_px
        n_rows = (H + bs - 1) // bs
        n_cols = (W + bs - 1) // bs
        n_blocks = n_rows * n_cols
        
        rng = np.random.default_rng(seed)
        block_ids = rng.permutation(n_blocks)
        folds = np.array_split(block_ids, k)
        splits = []
        valid_px = (label_map > 0) & (label_map < 255)
        
        for fold_idx in range(k):
            val_block_ids = set(folds[fold_idx])
            train_block_ids = set(np.concatenate([folds[j] for j in range(k) if j != fold_idx]))
            
            val_mask = np.zeros((H, W), dtype=bool)
            train_mask = np.zeros((H, W), dtype=bool)
            
            for block_id in range(n_blocks):
                r0 = (block_id // n_cols) * bs
                c0 = (block_id % n_cols) * bs
                r1 = min(r0 + bs, H)
                c1 = min(c0 + bs, W)
                
                if block_id in val_block_ids:
                    val_mask[r0:r1, c0:c1] = True
                elif block_id in train_block_ids:
                    train_mask[r0:r1, c0:c1] = True
            
            val_mask &= valid_px
            train_mask &= valid_px
            
            splits.append({
                "fold": fold_idx + 1,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "train_px": int(train_mask.sum()),
                "val_px": int(val_mask.sum())
            })
            
            print(f" Fold {fold_idx+1}: 训练{train_mask.sum():,}px | 验证{val_mask.sum():,}px")
        
        return splits

    def compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict:
        """计算分类精度指标"""
        num_classes = len(self.CROP_CLASSES)
        oa = (preds == labels).mean()
        iou_list = []
        
        for cls in range(1, num_classes):
            tp = ((preds==cls) & (labels==cls)).sum()
            fp = ((preds==cls) & (labels!=cls)).sum()
            fn = ((preds!=cls) & (labels==cls)).sum()
            if tp + fp + fn == 0:
                continue
            iou_list.append(tp / (tp + fp + fn + 1e-6))
        
        n = len(preds)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(labels, preds):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        
        po = cm.diagonal().sum() / (n + 1e-6)
        pe = (cm.sum(0) * cm.sum(1)).sum() / (n**2 + 1e-6)
        
        return {
            "OA": float(oa),
            "mIoU": float(np.mean(iou_list)) if iou_list else 0.0,
            "Kappa": float((po - pe) / (1 - pe + 1e-6)),
            "IoU_per_class": [float(v) for v in iou_list]
        }

    def run_spatial_kfold(self, model, splits: list, opt_seq, sar_seq, doy_norm, 
                          label_map, device="cuda", output_dir="./validation") -> dict:
        """执行空间K折交叉验证"""
        import torch
        os.makedirs(output_dir, exist_ok=True)
        fold_results = []
        
        for split in splits:
            fold = split["fold"]
            vmask = split["val_mask"]
            val_r, val_c = np.where(vmask)
            all_preds, all_labels = [], label_map[vmask].tolist()
            
            model.eval()
            with torch.no_grad():
                for i in range(0, len(val_r), 512):
                    r_b, c_b = val_r[i:i+512], val_c[i:i+512]
                    opt_b = opt_seq[:, :, r_b, c_b].transpose(2,0,1)[:, :, :, np.newaxis, np.newaxis]
                    sar_b = sar_seq[:, :, r_b, c_b].transpose(2,0,1)[:, :, :, np.newaxis, np.newaxis]
                    opt_t = torch.from_numpy(opt_b).float().to(device)
                    sar_t = torch.from_numpy(sar_b).float().to(device)
                    doy_t = torch.from_numpy(doy_norm).float().unsqueeze(0).expand(len(r_b), -1).to(device)
                    preds = model(opt_t, sar_t, doy_t).squeeze(-1).squeeze(-1).argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy().tolist())
            
            metrics = self.compute_metrics(np.array(all_preds), np.array(all_labels))
            metrics["fold"] = fold
            fold_results.append(metrics)
            print(f" Fold {fold}: mIoU={metrics['mIoU']:.4f} OA={metrics['OA']:.4f}")
        
        miou_list = [r["mIoU"] for r in fold_results]
        oa_list = [r["OA"] for r in fold_results]
        
        summary = {
            "mIoU_mean": float(np.mean(miou_list)),
            "mIoU_std": float(np.std(miou_list)),
            "OA_mean": float(np.mean(oa_list)),
            "OA_std": float(np.std(oa_list)),
            "fold_details": fold_results
        }
        
        print(f"\n空间K折汇总: mIoU={summary['mIoU_mean']:.4f}±{summary['mIoU_std']:.4f}")
        with open(f"{output_dir}/spatial_kfold_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary

    def cross_year_validation(self, model, train_data: list, test_data: dict, 
                             device="cuda", output_dir="./validation") -> dict:
        """跨年度泛化验证"""
        import torch
        os.makedirs(output_dir, exist_ok=True)
        test_year = test_data["year"]
        
        opt_seq, sar_seq, doy_norm, label_map = test_data["opt_sequence"], test_data["sar_sequence"], test_data["doy_norm"], test_data["label"]
        valid_mask = (label_map > 0) & (label_map < 255)
        val_r, val_c = np.where(valid_mask)
        all_preds, all_labels = [], label_map[valid_mask].tolist()
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_r), 512):
                r_b, c_b = val_r[i:i+512], val_c[i:i+512]
                opt_b = opt_seq[:, :, r_b, c_b].transpose(2,0,1)[:, :, :, np.newaxis, np.newaxis]
                sar_b = sar_seq[:, :, r_b, c_b].transpose(2,0,1)[:, :, :, np.newaxis, np.newaxis]
                opt_t = torch.from_numpy(opt_b).float().to(device)
                sar_t = torch.from_numpy(sar_b).float().to(device)
                doy_t = torch.from_numpy(doy_norm).float().unsqueeze(0).expand(len(r_b), -1).to(device)
                preds = model(opt_t, sar_t, doy_t).squeeze(-1).squeeze(-1).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
        
        metrics = self.compute_metrics(np.array(all_preds), np.array(all_labels))
        
        result = {
            "train_years": [d["year"] for d in train_data],
            "test_year": test_year,
            "metrics": metrics
        }
        
        print(f"\n跨年度验证结果: mIoU={metrics['mIoU']:.4f} OA={metrics['OA']:.4f}")
        with open(f"{output_dir}/cross_year_result_{test_year}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return result