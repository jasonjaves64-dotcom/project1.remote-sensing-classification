import os
import json
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
        "Description": "遥感影像作物分类模型",
        "Version": "1.0",
        "Author": "Remote Sensing Team"
    }
    
    export_dir = Path("arcgis_exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    emd_path = export_dir / "model.emd"
    with open(emd_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    
    print(f"EMD文件已保存到: {emd_path}")

if __name__ == "__main__":
    main()