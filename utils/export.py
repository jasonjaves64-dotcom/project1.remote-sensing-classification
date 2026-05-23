import torch
import onnx
import os
import json
import zipfile
from pathlib import Path

def export_to_onnx(model, output_path="model.onnx", 
                   opt_channels=10, sar_channels=5, dem_channels=5,
                   seq_len=12, patch_size=32, batch_size=1):
    model.eval()
    
    opt_dummy = torch.randn(batch_size, seq_len, opt_channels, patch_size, patch_size)
    sar_dummy = torch.randn(batch_size, seq_len, sar_channels, patch_size, patch_size)
    dem_dummy = torch.randn(batch_size, dem_channels, patch_size, patch_size)
    doy_dummy = torch.randn(batch_size, seq_len)
    
    torch.onnx.export(
        model,
        (opt_dummy, sar_dummy, dem_dummy, doy_dummy),
        output_path,
        opset_version=13,
        input_names=["opt", "sar", "dem", "doy"],
        output_names=["logits"],
        dynamic_axes={
            "opt": {0: "batch_size", 1: "seq_len"},
            "sar": {0: "batch_size", 1: "seq_len"},
            "dem": {0: "batch_size"},
            "doy": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"}
        },
        verbose=False,
        export_params=True
    )
    
    print(f"ONNX模型已导出: {output_path}")

def export_to_torchscript(model, output_path="model.pt", 
                         opt_channels=10, sar_channels=3,
                         seq_len=12, patch_size=32):
    model.eval()
    
    opt_dummy = torch.randn(1, seq_len, opt_channels, patch_size, patch_size)
    sar_dummy = torch.randn(1, seq_len, sar_channels, patch_size, patch_size)
    doy_dummy = torch.randn(1, seq_len)
    
    traced_model = torch.jit.trace(model, (opt_dummy, sar_dummy, doy_dummy))
    traced_model.save(output_path)
    
    print(f"✅ TorchScript模型已导出: {output_path}")

def export_to_coreml(model, output_path="model.mlmodel",
                     opt_channels=10, sar_channels=3,
                     seq_len=12, patch_size=32):
    try:
        import coremltools as ct
        
        model.eval()
        
        opt_dummy = torch.randn(1, seq_len, opt_channels, patch_size, patch_size)
        sar_dummy = torch.randn(1, seq_len, sar_channels, patch_size, patch_size)
        doy_dummy = torch.randn(1, seq_len)
        
        traced_model = torch.jit.trace(model, (opt_dummy, sar_dummy, doy_dummy))
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="opt", shape=opt_dummy.shape),
                ct.TensorType(name="sar", shape=sar_dummy.shape),
                ct.TensorType(name="doy", shape=doy_dummy.shape)
            ]
        )
        
        mlmodel.save(output_path)
        print(f"✅ CoreML模型已导出: {output_path}")
        
    except ImportError:
        print("❌ 需要安装coremltools: pip install coremltools")
    except Exception as e:
        print(f"❌ CoreML导出失败: {e}")

def export_to_emd(model, output_path="model.emd", 
                  opt_channels=10, sar_channels=5, dem_channels=5,
                  seq_len=12, patch_size=32, num_classes=7, 
                  crop_classes=None):
    if crop_classes is None:
        crop_classes = {
            0: "背景", 1: "冬小麦", 2: "夏玉米", 3: "水稻",
            4: "大豆", 5: "棉花", 6: "其他作物"
        }
    
    emd_content = {
        "Framework": "PyTorch",
        "ModelConfiguration": {
            "ModelType": "ObjectDetection" if num_classes > 1 else "Classification",
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
    
    print(f"✅ EMD模型定义文件已导出: {output_path}")

def export_to_dlpk(model, output_path="model.dlpk", 
                   opt_channels=10, sar_channels=5, dem_channels=5,
                   seq_len=12, patch_size=32, num_classes=7,
                   crop_classes=None):
    temp_dir = Path(output_path).parent / f"dlpk_temp_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = temp_dir / "model.onnx"
    emd_path = temp_dir / "model.emd"
    
    export_to_onnx(model, str(onnx_path), opt_channels, sar_channels, dem_channels, seq_len, patch_size)
    export_to_emd(model, str(emd_path), opt_channels, sar_channels, dem_channels,
                  seq_len, patch_size, num_classes, crop_classes)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, "model.onnx")
        zf.write(emd_path, "model.emd")
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"✅ DLPK深度学习包已导出: {output_path}")

def export_model(model, export_dir="exports", formats=["onnx", "torchscript"],
                 opt_channels=10, sar_channels=5, dem_channels=5,
                 seq_len=12, patch_size=32, num_classes=7, crop_classes=None):
    os.makedirs(export_dir, exist_ok=True)
    
    if "onnx" in formats:
        export_to_onnx(model, os.path.join(export_dir, "model.onnx"),
                       opt_channels, sar_channels, dem_channels, seq_len, patch_size)
    
    if "torchscript" in formats:
        export_to_torchscript(model, os.path.join(export_dir, "model.pt"),
                             opt_channels, sar_channels, seq_len, patch_size)
    
    if "coreml" in formats:
        export_to_coreml(model, os.path.join(export_dir, "model.mlmodel"),
                        opt_channels, sar_channels, seq_len, patch_size)
    
    if "emd" in formats:
        export_to_emd(model, os.path.join(export_dir, "model.emd"),
                     opt_channels, sar_channels, dem_channels,
                     seq_len, patch_size, num_classes, crop_classes)
    
    if "dlpk" in formats:
        export_to_dlpk(model, os.path.join(export_dir, "model.dlpk"),
                      opt_channels, sar_channels, dem_channels,
                      seq_len, patch_size, num_classes, crop_classes)