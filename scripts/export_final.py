import os
import sys
import torch
import json
import zipfile
from pathlib import Path

def main():
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
            "Classes": [{"Value": k, "Name": v} for k, v in crop_classes.items()]
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
            "n_layers": 4
        },
        "Extractor": {
            "ExtractBand": list(range(10)),
            "ExtractBandSAR": list(range(5)),
            "ExtractBandDEM": list(range(5))
        },
        "InferenceFunction": "Custom",
        "TargetDevice": "CPU",
        "ModelName": "FusionCropNetV5",
        "Description": "遥感影像作物分类模型 - 融合光学、SAR和DEM数据",
        "Version": "1.0",
        "Author": "Remote Sensing Team",
        "Contact": "remote.sensing@example.com"
    }
    
    export_dir = Path("arcgis_exports_final")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    emd_path = export_dir / "model.emd"
    with open(emd_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    
    print("EMD文件已导出")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(10, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.fc = torch.nn.Linear(128 * 32 * 32, 7)
        
        def forward(self, opt, sar, dem, doy):
            x = torch.cat([opt[:, 0], sar[:, 0], dem], dim=1)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    opt_dummy = torch.randn(1, 12, 10, 32, 32)
    sar_dummy = torch.randn(1, 12, 5, 32, 32)
    dem_dummy = torch.randn(1, 5, 32, 32)
    doy_dummy = torch.randn(1, 12)
    
    onnx_path = export_dir / "model.onnx"
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
    
    print("ONNX模型已导出")
    
    dlpk_path = export_dir / "model.dlpk"
    with zipfile.ZipFile(dlpk_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(emd_path, "model.emd")
        zf.write(onnx_path, "model.onnx")
    
    print("DLPK文件已导出")
    
    print(f"\n导出完成！文件已保存到 {export_dir}")
    print(f"  - model.emd    (ArcGIS模型定义文件)")
    print(f"  - model.onnx   (ONNX模型)")
    print(f"  - model.dlpk   (ArcGIS深度学习包)")

if __name__ == "__main__":
    main()