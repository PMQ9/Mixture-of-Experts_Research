"""
Export PyTorch models to ONNX format for use with NNV (Neural Network Verification)

This script handles exporting individual expert models (micro/tiny/small CNNs)
to ONNX format, which can then be loaded into NNV for formal verification.

Usage:
    # Export a single model
    python export_to_onnx.py --model_path artifacts/results/gtsrb_micro_cnn_best.pth --output_dir artifacts/nnv_models

    # Export all models in a directory
    python export_to_onnx.py --models_dir artifacts/results --output_dir artifacts/nnv_models
"""

import torch
import torch.nn as nn
import os
import sys
import argparse
from pathlib import Path

# Add Vision Transformer directory to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Vision_Transformer_Pytorch'))
sys.path.append(SRC_DIR)

from small_expert import MicroExpertCNN, TinyExpertCNN, SmallExpertCNN


def export_model_to_onnx(model_path, output_path, input_size=(1, 3, 32, 32), opset_version=11):
    """
    Export a PyTorch model to ONNX format

    Args:
        model_path: Path to .pth model file
        output_path: Path to save .onnx file
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version (11 is well-supported by NNV)
    """
    print(f"\n{'='*80}")
    print(f"Exporting: {model_path}")
    print(f"{'='*80}")

    # Load model
    device = torch.device('cpu')  # Always use CPU for ONNX export

    try:
        # Load the complete model object
        model = torch.load(model_path, map_location=device, weights_only=False)

        # Handle different model wrapper types
        if hasattr(model, 'model'):
            # ModelWrapper or ExpertModelWrapper
            model = model.model
        elif hasattr(model, 'experts'):
            # MetaMoE - cannot directly export, needs individual expert extraction
            print("ERROR: This is a MetaMoE model. Please export individual experts separately.")
            return False

    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

    # Set to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)

    # Get model info
    print(f"Model type: {type(model).__name__}")
    print(f"Input size: {input_size}")

    # Test forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        return False

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"\nSuccessfully exported to: {output_path}")

        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")

        return True

    except Exception as e:
        print(f"ERROR during ONNX export: {e}")
        return False


def detect_input_size(model_path):
    """
    Try to detect the expected input size from the model
    Returns (batch, channels, height, width)
    """
    # Default sizes for common datasets
    defaults = {
        'gtsrb': (1, 3, 32, 32),
        'cifar': (1, 3, 32, 32),
        'mnist': (1, 1, 28, 28),
    }

    model_name = os.path.basename(model_path).lower()

    for key, size in defaults.items():
        if key in model_name:
            return size

    # Default to CIFAR-10 size
    return (1, 3, 32, 32)


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX for NNV verification')
    parser.add_argument('--model_path', type=str, help='Path to a single .pth model file')
    parser.add_argument('--models_dir', type=str, help='Directory containing multiple .pth files')
    parser.add_argument('--output_dir', type=str, default='artifacts/nnv_models',
                       help='Output directory for ONNX models')
    parser.add_argument('--input_size', type=str, default=None,
                       help='Input size as "B,C,H,W" (e.g., "1,3,32,32")')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--filter', type=str, default='',
                       help='Only export models containing this string')

    args = parser.parse_args()

    # Parse input size if provided
    if args.input_size:
        input_size = tuple(map(int, args.input_size.split(',')))
    else:
        input_size = None

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect model paths
    model_paths = []

    if args.model_path:
        model_paths.append(args.model_path)
    elif args.models_dir:
        model_dir = Path(args.models_dir)
        model_paths = list(model_dir.glob('*.pth'))

        # Apply filter
        if args.filter:
            model_paths = [p for p in model_paths if args.filter in str(p)]
    else:
        parser.error("Must provide either --model_path or --models_dir")

    if not model_paths:
        print("No model files found!")
        return

    # Export each model
    success_count = 0
    for model_path in model_paths:
        model_path = str(model_path)

        # Determine output path
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.onnx")

        # Detect input size if not provided
        current_input_size = input_size if input_size else detect_input_size(model_path)

        # Export
        if export_model_to_onnx(model_path, output_path, current_input_size, args.opset_version):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"Export complete: {success_count}/{len(model_paths)} successful")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
