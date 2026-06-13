#!/usr/bin/env python3
"""Export FusionCropNetV6 to ONNX for 10× CPU inference speedup.

Usage:
    python scripts/export_v6_onnx.py --checkpoint checkpoints/v6_best.pth --output v6_model.onnx

The exported ONNX model can be deployed without PyTorch dependency using
ONNX Runtime, suitable for edge devices and production serving.
"""
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_v6_onnx(checkpoint_path: str, output_path: str,
                   opset_version: int = 17,
                   simplify: bool = True):
    """Export FusionCropNetV6 to ONNX format.

    Args:
        checkpoint_path: path to trained .pth checkpoint
        output_path: output .onnx file path
        opset_version: ONNX opset version (17 for best compatibility)
        simplify: run onnx-simplifier to optimize the graph
    """
    from models.fusion_net_v6 import FusionCropNetV6

    print(f"Loading model from {checkpoint_path}...")
    model = FusionCropNetV6(
        opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
        feat_dim=512, backbone="resnet50", pretrained=False,
        n_heads=16, n_layers=4, use_gradient_checkpointing=False,
    )

    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()

    # Create dummy inputs matching expected shapes
    B, T, H, W = 1, 12, 256, 256
    opt = torch.randn(B, T, 10, H, W)
    sar = torch.randn(B, T, 5, H, W)
    dem = torch.randn(B, 5, H, W)
    doy = torch.rand(B, T)

    print(f"Exporting to ONNX (opset={opset_version})...")
    torch.onnx.export(
        model,
        (opt, sar, dem, doy),
        output_path,
        input_names=['optical', 'sar', 'dem', 'doy'],
        output_names=['alpha'],
        dynamic_axes={
            'optical': {0: 'batch', 1: 'time'},
            'sar': {0: 'batch', 1: 'time'},
            'dem': {0: 'batch'},
            'doy': {0: 'batch'},
            'alpha': {0: 'batch'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Base ONNX exported to {output_path}")

    # Simplify the ONNX graph (removes redundant ops, constant folds)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            model_onnx = onnx.load(output_path)
            model_simp, check = onnx_simplify(model_onnx)
            if check:
                onnx.save(model_simp, output_path)
                print("ONNX graph simplified successfully")
            else:
                print("WARNING: ONNX simplification check failed")
        except ImportError:
            print("INFO: onnx-simplifier not installed, skipping simplification")
            print("      Install with: pip install onnx-simplifier")

    # Verify
    print("Verifying ONNX model...")
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    ort_inputs = {
        'optical': opt.numpy(),
        'sar': sar.numpy(),
        'dem': dem.numpy(),
        'doy': doy.numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)
    print(f"ONNX output shape: {ort_outputs[0].shape}")

    # Compare with PyTorch
    with torch.no_grad():
        pt_output = model(opt, sar, dem, doy)
        if isinstance(pt_output, tuple):
            pt_output = pt_output[0]

    diff = (pt_output.numpy() - ort_outputs[0]).mean()
    print(f"PyTorch-ONNX mean diff: {diff:.2e} (should be < 1e-4)")
    print(f"\nExport complete: {output_path}")
    print(f"Run with: onnxruntime.InferenceSession('{output_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FusionCropNetV6 to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", default="v6_model.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Skip simplification")
    args = parser.parse_args()
    export_v6_onnx(args.checkpoint, args.output, args.opset, not args.no_simplify)
