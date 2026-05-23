import os
import sys
import torch
import yaml
import json
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.fusion_net_v5 import FusionCropNetV5
from models.fusion_net_v5pro import FusionCropNetV5Pro

def load_config(path="config.yaml"):
    config_path = Path(path)
    if not config_path.exists():
        print("警告: 配置文件不存在，使用默认配置")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def export_emd_only(output_path, opt_channels=10, sar_channels=5, dem_channels=5,
                   seq_len=12, patch_size=32, num_classes=7, crop_classes=None):
    if crop_classes is None:
        crop_classes = {
            0: "背景", 1: "冬小麦", 2: "夏玉米", 3: "水稻",
            4: "大豆", 5: "棉花", 6: "其他作物"
        }
    
    emd_content = {
        "Framework": "PyTorch",
        "ModelConfiguration": {
            "ModelType": "Classification",
            "ImageHeight": patch_size,
            "ImageWidth": patch_size,
            "ImageChannels": opt_channels,
            "SequenceLength": seq_len,
            "SARChannels": sar_channels,
            "DEMChannels": dem_channels,
            "NumberOfClasses": num_classes,
            "Classes": [{"Value": k, "Name": v} for k, v in crop_classes.items()]
        },
        "ModelParameters": {
            "opt_channels": opt_channels,
            "sar_channels": sar_channels,
            "dem_channels": dem_channels,
            "num_classes": num_classes,
            "feat_dim": 512,
            "backbone": "resnet50",
            "n_heads": 16,
            "win_size": 4,
            "n_layers": 4
        },
        "Extractor": {
            "ExtractBand": list(range(opt_channels)),
            "ExtractBandSAR": list(range(sar_channels)),
            "ExtractBandDEM": list(range(dem_channels))
        },
        "InferenceFunction": "Custom",
        "TargetDevice": "CPU",
        "ModelName": "FusionCropNetV5",
        "Description": "遥感影像作物分类模型 - 融合光学、SAR和DEM数据",
        "Version": "1.0",
        "Author": "Remote Sensing Team",
        "Contact": "remote.sensing@example.com"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    
    print(f"EMD文件已导出: {output_path}")

def export_dlpk_with_onnx(model, output_path, opt_channels=10, sar_channels=5, dem_channels=5,
                          seq_len=12, patch_size=32, num_classes=7, crop_classes=None):
    export_dir = Path(output_path).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = export_dir / f"dlpk_temp_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = temp_dir / "model.onnx"
    emd_path = temp_dir / "model.emd"
    
    print("导出ONNX模型...")
    model.eval()
    opt_dummy = torch.randn(1, seq_len, opt_channels, patch_size, patch_size)
    sar_dummy = torch.randn(1, seq_len, sar_channels, patch_size, patch_size)
    dem_dummy = torch.randn(1, dem_channels, patch_size, patch_size)
    doy_dummy = torch.randn(1, seq_len)
    
    torch.onnx.export(
        model,
        (opt_dummy, sar_dummy, dem_dummy, doy_dummy),
        str(onnx_path),
        opset_version=18,
        input_names=["opt", "sar", "dem", "doy"],
        output_names=["logits"],
        export_params=True,
        verbose=False
    )
    print("ONNX模型导出完成")
    
    print("导出EMD文件...")
    export_emd_only(str(emd_path), opt_channels, sar_channels, dem_channels,
                   seq_len, patch_size, num_classes, crop_classes)
    
    print("创建DLPK包...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, "model.onnx")
        zf.write(emd_path, "model.emd")
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"DLPK文件已导出: {output_path}")

def main(use_v5pro=False):
    config = load_config()

    model_params = config.get('model', {})
    opt_channels = model_params.get('opt_channels', 10)
    sar_channels = model_params.get('sar_channels', 5)
    dem_channels = model_params.get('dem_channels', 5)
    num_classes = model_params.get('num_classes', 7)
    feat_dim = model_params.get('feat_dim', 512)
    backbone = model_params.get('backbone', 'resnet50')
    n_heads = model_params.get('n_heads', 16)
    win_size = model_params.get('win_size', 4)
    n_layers = model_params.get('n_layers', 4)

    crop_classes = config.get('crop_classes', {
        0: "背景", 1: "冬小麦", 2: "夏玉米", 3: "水稻",
        4: "大豆", 5: "棉花", 6: "其他作物"
    })

    export_dir = Path("arcgis_exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    print("导出EMD文件...")
    emd_path = export_dir / "model.emd"
    export_emd_only(str(emd_path), opt_channels, sar_channels, dem_channels,
                   12, 32, num_classes, crop_classes)

    print("\n创建模型...")
    if use_v5pro:
        model = FusionCropNetV5Pro(
            opt_ch=opt_channels,
            sar_ch=sar_channels,
            dem_ch_in=dem_channels,
            num_classes=num_classes,
            feat_dim=feat_dim,
            backbone=backbone,
            pretrained=False,
            n_heads=n_heads,
            win_size=win_size,
            n_layers=n_layers
        )
    else:
        model = FusionCropNetV5(
            opt_ch=opt_channels,
            sar_ch=sar_channels,
            dem_ch_in=dem_channels,
            num_classes=num_classes,
            feat_dim=feat_dim,
            backbone=backbone,
            pretrained=False,
            n_heads=n_heads,
            win_size=win_size,
            n_layers=n_layers
        )
    
    model.eval()
    print("模型创建完成")
    
    checkpoint_path = Path("checkpoints/best_phase2.pth")
    if checkpoint_path.exists():
        print(f"加载预训练权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
        print("预训练权重加载完成")
    else:
        print("未找到预训练权重文件，将使用随机初始化的模型")
    
    print("\n导出DLPK文件...")
    dlpk_path = export_dir / "model.dlpk"
    export_dlpk_with_onnx(model, str(dlpk_path), opt_channels, sar_channels, dem_channels,
                          12, 32, num_classes, crop_classes)
    
    print(f"\n导出完成！文件已保存到 {export_dir}")
    print(f"  - model.emd    (ArcGIS模型定义文件)")
    print(f"  - model.dlpk   (ArcGIS深度学习包)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5pro", action="store_true", help="Use V5Pro model")
    args = parser.parse_args()
    main(use_v5pro=args.v5pro)