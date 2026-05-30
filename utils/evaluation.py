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

    def _predict_batch(self, model, opt_list, sar_list, dem_list, doy_list, device):
        """Run inference on a batch of patches. Returns list of (ps, ps) numpy preds."""
        import torch
        opts = torch.cat([o for o in opt_list], dim=0).to(device)
        sars = torch.cat([s for s in sar_list], dim=0).to(device)
        doys = torch.cat([d for d in doy_list], dim=0).to(device)

        if dem_list:
            dems = torch.cat([d for d in dem_list], dim=0).to(device)
            alpha = model(opts, sars, dems, doys)
        else:
            alpha = model(opts, sars,
                          torch.zeros(opts.shape[0], 5, opts.shape[3], opts.shape[4], device=device),
                          doys)

        if isinstance(alpha, tuple):
            alpha = alpha[0]
        # alpha: (B, K, H, W)
        preds = alpha.argmax(dim=1).cpu().numpy()  # (B, H, W)
        return [preds[i] for i in range(preds.shape[0])]

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

    def cross_year_degradation_analysis(self, model, test_years: dict,
                                         reference_year=None, device="cuda",
                                         output_dir="./validation",
                                         patch_size: int = 32,
                                         batch_size: int = 8) -> dict:
        """Per-class cross-year degradation analysis.

        Runs inference on multiple test years, computes per-class IoU for each,
        and quantifies degradation relative to a reference year (default: first).

        Uses patch-based inference (grid of patch_size × patch_size tiles) to
        handle models that require minimum spatial dimensions.

        Args:
            model:            trained FusionCropNet model
            test_years:       dict {year_label: {opt_sequence, sar_sequence, doy_norm, label, dem}}
            reference_year:   year label to use as baseline (default: first key)
            device:           torch device
            output_dir:       directory for output files
            patch_size:       spatial patch size for inference (default 32)
            batch_size:       patches per batch

        Returns:
            dict with per-year metrics, per-class degradation deltas, and
            stability ranking.
        """
        import torch
        os.makedirs(output_dir, exist_ok=True)

        if reference_year is None:
            reference_year = list(test_years.keys())[0]

        year_results = {}
        all_years = list(test_years.keys())

        print(f"\n{'='*60}")
        print(f"  Cross-Year Per-Class Degradation Analysis")
        print(f"  Reference year: {reference_year}")
        print(f"  Test years: {all_years}")
        print(f"{'='*60}")

        for year_key in all_years:
            data = test_years[year_key]
            opt_seq = data["opt_sequence"]
            sar_seq = data["sar_sequence"]
            doy_norm = data["doy_norm"]
            dem = data.get("dem", None)
            label_map = data["label"]

            H, W = label_map.shape
            ps = patch_size
            stride = ps // 2  # 50% overlap

            # Collect all valid patches
            full_preds = np.full((H, W), 255, dtype=np.int64)

            model.eval()
            with torch.no_grad():
                for y0 in range(0, H - ps + 1, stride):
                    y1 = y0 + ps
                    patches_opt, patches_sar, patches_dem, patches_doy = [], [], [], []
                    positions = []

                    for x0 in range(0, W - ps + 1, stride):
                        x1 = x0 + ps
                        # Extract patch: (T, C, ps, ps) → add batch dim → (1, T, C, ps, ps)
                        opt_p = opt_seq[np.newaxis, :, :, y0:y1, x0:x1]
                        sar_p = sar_seq[np.newaxis, :, :, y0:y1, x0:x1]

                        patches_opt.append(torch.from_numpy(opt_p.copy()).float())
                        patches_sar.append(torch.from_numpy(sar_p.copy()).float())
                        patches_doy.append(torch.from_numpy(doy_norm).float().unsqueeze(0))

                        if dem is not None:
                            dem_p = dem[np.newaxis, :, y0:y1, x0:x1]  # (1, 5, ps, ps)
                            patches_dem.append(torch.from_numpy(dem_p.copy()).float())

                        positions.append((y0, x0))

                        if len(patches_opt) >= batch_size:
                            # Run batch
                            batch_preds = self._predict_batch(
                                model, patches_opt, patches_sar,
                                patches_dem if dem is not None else None,
                                patches_doy, device)
                            for (py, px), pred_patch in zip(positions, batch_preds):
                                full_preds[py:py+ps, px:px+ps] = pred_patch
                            patches_opt, patches_sar, patches_dem, patches_doy = [], [], [], []
                            positions = []

                    # Process remaining patches in row
                    if patches_opt:
                        batch_preds = self._predict_batch(
                            model, patches_opt, patches_sar,
                            patches_dem if dem is not None else None,
                            patches_doy, device)
                        for (py, px), pred_patch in zip(positions, batch_preds):
                            full_preds[py:py+ps, px:px+ps] = pred_patch
                        patches_opt, patches_sar, patches_dem, patches_doy = [], [], [], []
                        positions = []

            # Compute metrics on valid pixels (non-overlap center regions for clean eval)
            valid_mask = (label_map > 0) & (label_map < 255)
            # Use only center region of each patch to avoid overlap double-counting bias
            center_mask = np.zeros((H, W), dtype=bool)
            margin = ps // 4
            for y0 in range(0, H - ps + 1, stride):
                for x0 in range(0, W - ps + 1, stride):
                    center_mask[y0+margin:y0+ps-margin, x0+margin:x0+ps-margin] = True
            eval_mask = valid_mask & center_mask & (full_preds != 255)

            metrics = self.compute_metrics(full_preds[eval_mask], label_map[eval_mask])
            year_results[year_key] = metrics
            print(f"  [{year_key}] OA={metrics['OA']:.4f}  mIoU={metrics['mIoU']:.4f}  "
                  f"Kappa={metrics['Kappa']:.4f}")

        # ── Per-class degradation vs reference ──
        ref_metrics = year_results[reference_year]
        ref_iou = ref_metrics.get("IoU_per_class", [])

        per_class_degradation = {}
        for year_key in all_years:
            if year_key == reference_year:
                continue
            yr_iou = year_results[year_key].get("IoU_per_class", [])
            deltas = []
            for ci in range(len(ref_iou)):
                ri = ref_iou[ci] if ci < len(ref_iou) else 0.0
                yi = yr_iou[ci] if ci < len(yr_iou) else 0.0
                deltas.append({
                    "class_id": ci + 1,
                    "class_name": self.CROP_CLASSES.get(ci + 1, f"Class_{ci+1}"),
                    "ref_IoU": float(ri),
                    "year_IoU": float(yi),
                    "delta_IoU": float(ri - yi),
                    "delta_rel": float((ri - yi) / max(ri, 1e-6)),
                })
            deltas.sort(key=lambda x: x["delta_IoU"], reverse=True)
            per_class_degradation[year_key] = deltas

        # ── Stability ranking: which classes degrade most across all years? ──
        class_stability = {}
        num_classes = len(ref_iou)
        for ci in range(num_classes):
            cname = self.CROP_CLASSES.get(ci + 1, f"Class_{ci+1}")
            degradations = []
            for year_key in all_years:
                if year_key == reference_year:
                    continue
                yr_iou = year_results[year_key].get("IoU_per_class", [])
                if ci < len(yr_iou):
                    degradations.append(ref_iou[ci] - yr_iou[ci])
            mean_degrad = float(np.mean(degradations)) if degradations else 0.0
            max_degrad = float(np.max(degradations)) if degradations else 0.0
            std_degrad = float(np.std(degradations)) if degradations else 0.0
            class_stability[ci + 1] = {
                "class_name": cname,
                "ref_IoU": float(ref_iou[ci]),
                "mean_degradation": mean_degrad,
                "max_degradation": max_degrad,
                "std_degradation": std_degrad,
                "stability_score": float(1.0 / (1.0 + abs(mean_degrad) + std_degrad)),
            }

        stability_ranking = sorted(class_stability.items(),
                                    key=lambda x: x[1]["stability_score"],
                                    reverse=True)

        # ── Summary ──
        summary = {
            "reference_year": reference_year,
            "test_years": all_years,
            "year_results": {y: {k: v for k, v in m.items()}
                             for y, m in year_results.items()},
            "per_class_degradation": per_class_degradation,
            "class_stability": {str(k): v for k, v in class_stability.items()},
            "stability_ranking": [
                {"class_id": cid, "class_name": info["class_name"],
                 "stability_score": info["stability_score"],
                 "mean_degradation": info["mean_degradation"]}
                for cid, info in stability_ranking
            ],
            "mIoU_degradation": {
                y: float(ref_metrics["mIoU"] - year_results[y]["mIoU"])
                for y in all_years if y != reference_year
            },
        }

        # Save
        out_path = f"{output_dir}/cross_year_degradation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Print per-class degradation table
        print(f"\n{'─'*60}")
        print(f"  Per-Class IoU Degradation (ref={reference_year})")
        print(f"{'─'*60}")

        for year_key in all_years:
            if year_key == reference_year:
                continue
            deltas = per_class_degradation[year_key]
            print(f"\n  [{reference_year} → {year_key}]  "
                  f"mIoU drop: {summary['mIoU_degradation'][year_key]:.4f}")
            print(f"  {'Class':<14s} {'Ref IoU':>8s} {'Year IoU':>8s} "
                  f"{'ΔIoU':>8s} {'ΔRel%':>8s} {'Trend':>6s}")
            for d in deltas:
                trend = "▼▼" if d["delta_rel"] > 0.2 else ("▼" if d["delta_IoU"] > 0.01
                        else ("─" if abs(d["delta_IoU"]) < 0.01 else "▲"))
                print(f"  {d['class_name']:<14s} {d['ref_IoU']:8.4f} {d['year_IoU']:8.4f} "
                      f"{d['delta_IoU']:8.4f} {d['delta_rel']*100:7.1f}% {trend:>6s}")

        # Stability ranking
        print(f"\n{'─'*60}")
        print(f"  Crop Class Stability Ranking (higher = more stable across years)")
        print(f"{'─'*60}")
        print(f"  {'Rank':<5s} {'Class':<14s} {'Stability':>9s} {'Mean ΔIoU':>10s}")
        for rank, (cid, info) in enumerate(stability_ranking, 1):
            print(f"  {rank:<5d} {info['class_name']:<14s} "
                  f"{info['stability_score']:9.4f} {info['mean_degradation']:+10.4f}")

        print(f"\n  Full results saved to: {out_path}")
        return summary