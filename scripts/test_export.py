import os
import sys
import json
import zipfile
from pathlib import Path

crop_classes = {
    0: "背景", 1: "冬小麦", 2: "夏玉米", 3: "水稻",
    4: "大豆", 5: "棉花", 6: "其他作物"
}

def test_emd_export():
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
    
    output_path = "test_output/model.emd"
    os.makedirs("test_output", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    
    print(f"EMD文件已导出: {output_path}")

def test_dlpk_export():
    temp_dir = Path("test_output/dlpk_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    emd_path = temp_dir / "model.emd"
    with open(emd_path, 'w', encoding='utf-8') as f:
        json.dump(emd_content, f, ensure_ascii=False, indent=2)
    
    dummy_onnx_path = temp_dir / "model.onnx"
    with open(dummy_onnx_path, 'wb') as f:
        f.write(b'ONNX_MODEL_DATA')
    
    output_path = "test_output/model.dlpk"
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(emd_path, "model.emd")
        zf.write(dummy_onnx_path, "model.onnx")
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"DLPK文件已导出: {output_path}")

if __name__ == "__main__":
    print("测试EMD导出...")
    test_emd_export()
    
    print("\n测试DLPK导出...")
    test_dlpk_export()
    
    print("\n测试完成！")