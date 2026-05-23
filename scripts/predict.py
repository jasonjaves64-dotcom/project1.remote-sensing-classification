"""
推理脚本 - 支持EDL不确定性估计
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path

# 导入模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fusion_net_v5_edl import FusionCropNetV5EDL
from models.fusion_net_v5pro import FusionCropNetV5Pro
from models.fusion_net_v6 import FusionCropNetV6
from data.preprocess_pipeline import PreprocessPipeline, PreprocessConfig

def main():
    parser = argparse.ArgumentParser(description="作物分类推理")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--input_path", type=str, required=True, help="输入数据路径")
    parser.add_argument("--output_path", type=str, default="output/", help="输出结果路径")
    parser.add_argument("--v5pro", action="store_true", help="使用 V5Pro 模型")
    parser.add_argument("--v6", action="store_true", help="使用 V6 模型 (自动启用 V6 enhancements)")
    parser.add_argument("--backbone", type=str, default="resnet50", help="骨干网络")
    parser.add_argument("--edl", action="store_true", help="启用EDL不确定性估计")
    parser.add_argument("--n_passes", type=int, default=5, help="不确定性推理次数")
    parser.add_argument("--use_tta", action="store_true", help="使用测试时增强")
    parser.add_argument("--calibration", action="store_true", help="输出校准验证报告")
    parser.add_argument("--interpretability", action="store_true", help="输出可解释性分析")
    parser.add_argument("--label_path", type=str, default=None, help="标签路径 (用于校准验证)")
    parser.add_argument('--rs_weights', type=str, default=None,
                        help='Path to remote sensing pre-trained weights (SeCo, GASSL)')
    args = parser.parse_args()
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.v6:
        model = FusionCropNetV6(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone=args.backbone, pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            modality_dropout_p=0.0, use_gradient_checkpointing=False,
            rs_weights=args.rs_weights,
        ).to(device)
    elif args.v5pro:
        model = FusionCropNetV5Pro(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone=args.backbone, pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            use_carafe=True, dynamic_dropout=False, adaptive_kl=False,
            rs_weights_path=args.rs_weights,
        ).to(device)
    else:
        model = FusionCropNetV5EDL(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone=args.backbone, pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            rs_weights_path=args.rs_weights,
        ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 初始化预处理管道
    config = PreprocessConfig(
        normalize=True, freeze_stats=True, sar_log_transform=True, augment=False
    )
    pipeline = PreprocessPipeline(config)
    
    # 读取数据
    print(f"正在读取数据: {args.input_path}")
    if args.input_path.endswith('.npy'):
        data = np.load(args.input_path)
    else:
        from data.preprocess_pipeline import DataReader
        data = DataReader.read_data(args.input_path, 'opt')
    
    # 预处理
    T, _, H, W = data.shape if data.ndim == 4 else (data.shape[0], 1, *data.shape[-2:])
    has_sar = os.path.exists(os.path.join(os.path.dirname(args.input_path), "sar_sequence.npy"))
    has_dem = os.path.exists(os.path.join(os.path.dirname(args.input_path), "dem.npy"))

    data_quality = {"sar_available": has_sar, "dem_available": has_dem}
    if not has_sar:
        print("⚠ Warning: SAR data not found — using zeros (SAR encoder will receive placeholder). Prediction quality may degrade.")
    if not has_dem:
        print("⚠ Warning: DEM data not found — using zeros (DEM encoder will receive placeholder). Prediction quality may degrade.")

    sample = pipeline.process(
        {
            'opt': data,
            'sar': np.load(os.path.join(os.path.dirname(args.input_path), "sar_sequence.npy"))
                   if has_sar else np.zeros((data.shape[0], 5, H, W), dtype=np.float32),
            'dem': np.load(os.path.join(os.path.dirname(args.input_path), "dem.npy"))
                   if has_dem else np.zeros((5, H, W), dtype=np.float32),
            'doy': np.linspace(0, 1, data.shape[0]),
        },
        transforms={},
        is_training=False,
    )
    
    # 推理
    with torch.no_grad():
        opt_t = torch.from_numpy(sample.opt_seq).unsqueeze(0).to(next(model.parameters()).device)
        sar_t = torch.from_numpy(sample.sar_seq).unsqueeze(0).to(next(model.parameters()).device)
        dem_t = torch.from_numpy(sample.dem).unsqueeze(0).to(next(model.parameters()).device)
        doy_t = torch.from_numpy(sample.doy).unsqueeze(0).to(next(model.parameters()).device)
        
        if args.edl:
            result = model.predict_uncertainty(
                opt_t, sar_t, dem_t, doy_t,
                n_passes=args.n_passes,
                use_tta=args.use_tta
            )
            pred = result['pred_class'].squeeze().cpu().numpy()
            probs = result['probs'].squeeze().cpu().numpy()
            vacuity = result['vacuity'].squeeze().cpu().numpy()
            dissonance = result['dissonance'].squeeze().cpu().numpy()
            alpha = result.get('alpha_fused', probs)  # for calibration report
            
            # 保存结果
            os.makedirs(args.output_path, exist_ok=True)
            np.save(os.path.join(args.output_path, 'prediction.npy'), pred)
            np.save(os.path.join(args.output_path, 'probabilities.npy'), probs)
            np.save(os.path.join(args.output_path, 'vacuity.npy'), vacuity)
            np.save(os.path.join(args.output_path, 'dissonance.npy'), dissonance)
            print(f"结果已保存到: {args.output_path}")
        else:
            pred = model(opt_t, sar_t, dem_t, doy_t)
            pred = pred.argmax(dim=1).squeeze().cpu().numpy()

            os.makedirs(args.output_path, exist_ok=True)
            np.save(os.path.join(args.output_path, 'prediction.npy'), pred)
            print(f"结果已保存到: {args.output_path}")

        # Calibration report
        if args.calibration and args.edl and args.label_path and os.path.exists(args.label_path):
            from utils.calibration import calibration_report, print_calibration_report
            label = np.load(args.label_path)
            print("\n" + "=" * 60)
            print("EDL校准验证报告")
            print("=" * 60)
            cal = calibration_report(
                alpha if args.edl else pred[np.newaxis, np.newaxis],
                label, num_classes=7, n_bins=15)
            print_calibration_report(cal)
            import json
            with open(os.path.join(args.output_path, 'calibration_report.json'), 'w', encoding='utf-8') as f:
                serializable = {k: v for k, v in cal.items() if k != "_raw"}
                json.dump(serializable, f, ensure_ascii=False, indent=2,
                         default=lambda x: float(x) if hasattr(x, 'item') else str(x))
            print(f"校准报告已保存到: {os.path.join(args.output_path, 'calibration_report.json')}")

        # Interpretability analysis
        if args.interpretability and args.edl and has_sar and has_dem:
            from utils.interpretability import modality_ablation, temporal_importance
            device_str = str(next(model.parameters()).device)
            print("\n模态消融分析...")
            abl = modality_ablation(model, opt_t, sar_t, dem_t, doy_t, device=device_str)
            print(f"  模态贡献: {abl.get('relative_importance', 'N/A')}")
            print("时序重要性分析...")
            t_imp, _ = temporal_importance(model, opt_t, sar_t, dem_t, doy_t, device=device_str)
            print(f"  前3重要时间步: {np.argsort(t_imp)[-3:][::-1]}")

if __name__ == "__main__":
    main()
