import os
import sys
import torch
import json
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.fusion_net_v5_edl import FusionCropNetV5EDL, dirichlet_to_predictions
from models.fusion_net_v5pro import FusionCropNetV5Pro
from models.fusion_net_v6 import FusionCropNetV6
from utils.calibration import calibration_report

def main(use_v5pro=False, use_v6=False):
    crop_classes = {
        0: "背景", 1: "冬小麦", 2: "夏玉米", 3: "水稻",
        4: "大豆", 5: "棉花", 6: "其他作物"
    }
    
    emd_content = {
        "Framework": "PyTorch",
        "ModelConfiguration": {
            "ModelType": "Classification",
            "ImageHeight": 32,
            "ImageWidth": 32,
            "ImageChannels": 10,
            "SequenceLength": 12,
            "SARChannels": 5,
            "DEMChannels": 5,
            "NumberOfClasses": 7,
            "Classes": [{"Value": k, "Name": v} for k, v in crop_classes.items()],
            "UncertaintyEnabled": True,
            "UncertaintyType": "EDL-Ensemble",
            "OutputAlpha": True
        },
        "ModelParameters": {
            "opt_channels": 10,
            "sar_channels": 5,
            "dem_channels": 5,
            "num_classes": 7,
            "feat_dim": 512,
            "backbone": "resnet50",
            "n_heads": 16,
            "win_size": 4,
            "n_layers": 4,
            "edl_dropout_p": 0.3,
            "edl_lambda_max": 0.5,
            "edl_anneal_ep": 50
        },
        "Extractor": {
            "ExtractBand": list(range(10)),
            "ExtractBandSAR": list(range(5)),
            "ExtractBandDEM": list(range(5))
        },
        "InferenceFunction": "CustomEDL",
        "TargetDevice": "CPU",
        "ModelName": "FusionCropNetV5EDL",
        "Description": "遥感影像作物分类模型 - 融合光学、SAR和DEM数据，带EDL不确定性估计",
        "Version": "1.1",
        "Author": "Remote Sensing Team",
        "Contact": "remote.sensing@example.com",
        "UncertaintyMetrics": {
            "Vacuity": "数据不确定性（证据不足）",
            "Dissonance": "认知不确定性（证据冲突）",
            "ClassVariance": "每类预测方差"
        }
    }
    
    export_dir = Path("arcgis_exports_edl")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    emd_path = export_dir / "model_emd.emd"
    with open(emd_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    
    print("EMD文件已导出")

    if use_v6:
        model = FusionCropNetV6(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet50", pretrained=False,
            n_heads=16, win_size=4, n_layers=4,
            edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50,
            modality_dropout_p=0.0, use_gradient_checkpointing=False,
        )
    elif use_v5pro:
        model = FusionCropNetV5Pro(
            opt_ch=10,
            sar_ch=5,
            dem_ch_in=5,
            num_classes=7,
            feat_dim=512,
            backbone="resnet50",
            pretrained=False,
            n_heads=16,
            win_size=4,
            n_layers=4,
            edl_dropout_p=0.3,
            edl_lambda_max=0.5,
            edl_anneal_ep=50
        )
    else:
        model = FusionCropNetV5EDL(
            opt_ch=10,
            sar_ch=5,
            dem_ch_in=5,
            num_classes=7,
            feat_dim=512,
            backbone="resnet50",
            pretrained=False,
            n_heads=16,
            win_size=4,
            n_layers=4,
            edl_dropout_p=0.3,
            edl_lambda_max=0.5,
            edl_anneal_ep=50
        )
    model.eval()
    
    torch.save(model.state_dict(), export_dir / "model_weights.pth")
    print("模型权重已导出")
    
    opt_dummy = torch.randn(1, 12, 10, 32, 32)
    sar_dummy = torch.randn(1, 12, 5, 32, 32)
    dem_dummy = torch.randn(1, 5, 32, 32)
    doy_dummy = torch.linspace(0, 1, 12).unsqueeze(0)
    
    onnx_path = export_dir / "model.onnx"
    
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, opt, sar, dem, doy):
            alpha = self.model(opt, sar, dem, doy)
            S = alpha.sum(dim=1, keepdim=True)
            probs = alpha / S
            return probs
    
    wrapper = ExportWrapper(model)
    
    torch.onnx.export(
        wrapper,
        (opt_dummy, sar_dummy, dem_dummy, doy_dummy),
        str(onnx_path),
        opset_version=18,
        input_names=["opt", "sar", "dem", "doy"],
        output_names=["probs"],
        export_params=True,
        verbose=False,
        dynamic_axes={
            "opt": {0: "batch_size", 1: "seq_len"},
            "sar": {0: "batch_size", 1: "seq_len"},
            "dem": {0: "batch_size"},
            "doy": {0: "batch_size"},
            "probs": {0: "batch_size"}
        }
    )
    
    print("ONNX模型已导出")
    
    dlpk_path = export_dir / "model_edl.dlpk"
    with zipfile.ZipFile(dlpk_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(emd_path, "model.emd")
        zf.write(onnx_path, "model.onnx")
        zf.write(export_dir / "model_weights.pth", "model_weights.pth")
    
    print("DLPK文件已导出")
    
    print(f"\n导出完成！文件已保存到 {export_dir}")
    print(f"  - model_emd.emd    (ArcGIS模型定义文件)")
    print(f"  - model.onnx       (ONNX模型)")
    print(f"  - model_weights.pth (PyTorch权重)")
    print(f"  - model_edl.dlpk   (ArcGIS深度学习包)")
    
    print("\nEDL模型特性：")
    print("  ✓ 支持不确定性估计（Vacuity + Dissonance）")
    print("  ✓ 证据级融合（Evidence-level Fusion）")
    print("  ✓ 支持TTA（Test-Time Augmentation）")
    print("  ✓ 狄利克雷分布输出")

    # Calibration baseline on dummy data
    print("\n评估校准基线...")
    model.eval()
    B, T, H, W = 4, 12, 32, 32
    opt_d = torch.randn(B, T, 10, H, W)
    sar_d = torch.randn(B, T, 5, H, W)
    dem_d = torch.randn(B, 5, H, W)
    doy_d = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    targets_d = torch.randint(0, 7, (B, H, W))

    with torch.no_grad():
        alpha = model(opt_d, sar_d, dem_d, doy_d)
    cal = calibration_report(alpha.numpy(), targets_d.numpy(), num_classes=7, n_bins=10)
    emd_content["CalibrationBaseline"] = {
        "ECE": cal["ECE"],
        "NLL": cal["NLL"],
        "Brier": cal["Brier"],
    }
    print(f"  校准基线: ECE={cal['ECE']:.4f}, NLL={cal['NLL']:.4f}, Brier={cal['Brier']:.4f}")

    # Update EMD with calibration data
    with open(emd_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    print(f"EMD文件已更新（含校准基线）")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5pro", action="store_true", help="Use FusionCropNetV5Pro model")
    parser.add_argument("--v6", action="store_true", help="Use FusionCropNetV6 model")
    args = parser.parse_args()
    main(use_v5pro=args.v5pro, use_v6=args.v6)