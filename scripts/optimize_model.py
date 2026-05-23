"""
模型性能优化工具 - ONNX导出、量化、推理优化
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, '.')

def export_to_onnx(model, output_path="model.onnx", opset_version=16):
    """导出模型到 ONNX 格式"""
    print("Exporting model to ONNX...")
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    dummy_opt = torch.randn(1, 12, 10, 64, 64, device=device)
    dummy_sar = torch.randn(1, 12, 5, 64, 64, device=device)
    dummy_dem = torch.randn(1, 5, 64, 64, device=device)
    dummy_doy = torch.rand(1, 12, device=device)
    
    torch.onnx.export(
        model,
        (dummy_opt, dummy_sar, dummy_dem, dummy_doy),
        output_path,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["opt_seq", "sar_seq", "dem", "doy"],
        output_names=["logits"],
        dynamic_axes={
            "opt_seq": {0: "batch_size", 4: "height", 5: "width"},
            "sar_seq": {0: "batch_size", 4: "height", 5: "width"},
            "dem": {0: "batch_size", 2: "height", 3: "width"},
            "logits": {0: "batch_size", 2: "height", 3: "width"}
        },
        verbose=False
    )
    
    print(f"ONNX model exported to: {output_path}")
    return output_path

def optimize_with_tensorrt(onnx_path, engine_path="model.engine"):
    """使用 TensorRT 优化模型"""
    try:
        import tensorrt as trt
        print("Optimizing model with TensorRT...")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(1) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            builder.max_workspace_size = 1 << 30
            builder.fp16_mode = True
            
            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())
            
            engine = builder.build_cuda_engine(network)
            
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine saved to: {engine_path}")
            return engine_path
    
    except ImportError:
        print("TensorRT not installed, skipping...")
        return None

def quantize_model(model):
    """量化模型（INT8/FP16）"""
    print("Quantizing model...")
    
    model_fp16 = model.half()
    
    try:
        model_quantized = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        print("INT8 quantization completed")
        return model_quantized
    except Exception as e:
        print(f"INT8 quantization failed: {e}")
        return model_fp16

def benchmark_inference(model, device, iterations=100, batch_size=1):
    """基准测试推理性能"""
    print(f"\nBenchmarking inference on {device}...")
    
    model = model.to(device)
    model.eval()
    
    opt_seq = torch.randn(batch_size, 12, 10, 64, 64).to(device)
    sar_seq = torch.randn(batch_size, 12, 5, 64, 64).to(device)
    dem = torch.randn(batch_size, 5, 64, 64).to(device)
    doy = torch.rand(batch_size, 12).to(device)
    
    warmup_iterations = 10
    for _ in range(warmup_iterations):
        with torch.no_grad():
            model(opt_seq, sar_seq, dem, doy)
    
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            logits = model(opt_seq, sar_seq, dem, doy)
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / iterations
    fps = iterations / elapsed_time
    
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed_time:.3f}s")
    print(f"Average time per inference: {avg_time*1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
    
    return {
        "device": str(device),
        "iterations": iterations,
        "total_time": elapsed_time,
        "avg_time_ms": avg_time * 1000,
        "fps": fps
    }

def optimize_for_edge(model, output_dir="optimized_model"):
    """为边缘设备优化模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nOptimizing for edge deployment...")
    
    model_quantized = quantize_model(model)
    
    torch.save(model_quantized.state_dict(), os.path.join(output_dir, "model_quantized.pth"))
    
    onnx_path = os.path.join(output_dir, "model.onnx")
    export_to_onnx(model_quantized, onnx_path)
    
    print(f"Optimized models saved to: {output_dir}/")
    
    return output_dir

def main():
    from models.fusion_net_v5 import FusionCropNetV5
    from models.fusion_net_v5pro import FusionCropNetV5Pro

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5pro", action="store_true", help="Use V5Pro model")
    args = ap.parse_args()

    print("=" * 60)
    print("FusionCropNetV5" + ("Pro" if args.v5pro else "") + " Performance Optimization")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.v5pro:
        model = FusionCropNetV5Pro(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
    else:
        model = FusionCropNetV5(
            opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
            feat_dim=512, backbone="resnet18", pretrained=False,
            n_heads=8, win_size=4, n_layers=2
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("\n" + "=" * 40)
    print("1. Original Model Benchmark")
    print("=" * 40)
    original_stats = benchmark_inference(model, device)
    
    print("\n" + "=" * 40)
    print("2. FP16 Model Benchmark")
    print("=" * 40)
    model_fp16 = model.half()
    fp16_stats = benchmark_inference(model_fp16, device)
    
    print("\n" + "=" * 40)
    print("3. Model Export")
    print("=" * 40)
    export_to_onnx(model, "fusion_net_v5.onnx")
    
    print("\n" + "=" * 40)
    print("4. Edge Optimization")
    print("=" * 40)
    optimize_for_edge(model)
    
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    print(f"{'Metric':<20} {'Original':<15} {'FP16':<15}")
    print(f"{'Avg Time':<20} {original_stats['avg_time_ms']:.2f}ms {fp16_stats['avg_time_ms']:.2f}ms")
    print(f"{'FPS':<20} {original_stats['fps']:.2f} {fp16_stats['fps']:.2f}")
    
    print("\n" + "=" * 60)
    print("Optimization completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
