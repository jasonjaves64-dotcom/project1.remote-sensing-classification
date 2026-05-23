import os
import sys
import torch
import json
import zipfile
from pathlib import Path

def create_simple_model():
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(10, 64, 3, padding=1)
            self.fc = torch.nn.Linear(64 * 32 * 32, 7)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    return SimpleModel()

def export_dlpk(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        "Description": "遥感影像作物分类模型",
        "Version": "1.0",
        "Author": "Remote Sensing Team"
    }
    
    temp_dir = output_dir / "dlpk_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    emd_path = temp_dir / "model.emd"
    with open(emd_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    
    print("创建简单的ONNX模型...")
    model = create_simple_model()
    model.eval()
    dummy_input = torch.randn(1, 10, 32, 32)
    
    onnx_path = temp_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        verbose=False
    )
    
    dlpk_path = output_dir / "model.dlpk"
    with zipfile.ZipFile(dlpk_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(emd_path, "model.emd")
        zf.write(onnx_path, "model.onnx")
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"DLPK文件已导出: {dlpk_path}")

if __name__ == "__main__":
    export_dlpk("arcgis_exports")
    print("导出完成！")