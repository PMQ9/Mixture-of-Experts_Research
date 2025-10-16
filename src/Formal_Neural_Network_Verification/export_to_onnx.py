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
import torch.nn.functional as F
import os
import sys
import argparse
from pathlib import Path

# Add Vision Transformer directory to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Vision_Transformer_Pytorch'))
sys.path.append(SRC_DIR)

from small_expert import MicroExpertCNN, TinyExpertCNN, SmallExpertCNN, NNVCompatibleCNN, UltraVerifiableCNN


def fold_batch_norm_into_conv(conv, bn):
    """
    Fold BatchNorm parameters into Conv layer weights and biases.
    This eliminates the BatchNorm layer entirely, making the model simpler for NNV.

    Formula:
        y = gamma * (conv(x) - running_mean) / sqrt(running_var + eps) + beta

    Folded conv:
        new_weight = gamma / sqrt(running_var + eps) * old_weight
        new_bias = gamma / sqrt(running_var + eps) * (old_bias - running_mean) + beta

    Args:
        conv: nn.Conv2d layer
        bn: nn.BatchNorm2d layer

    Returns:
        New Conv2d layer with folded weights
    """
    # Get BatchNorm parameters
    gamma = bn.weight.data
    beta = bn.bias.data
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    # Compute scaling factor
    scale = gamma / torch.sqrt(running_var + eps)

    # Create new conv layer with bias
    new_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=True  # Always use bias after folding
    )

    # Fold weights: w_new = scale * w_old
    new_conv.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)

    # Fold bias: b_new = scale * (b_old - mean) + beta
    if conv.bias is not None:
        new_conv.bias.data = scale * (conv.bias.data - running_mean) + beta
    else:
        new_conv.bias.data = scale * (0 - running_mean) + beta

    return new_conv


class NNVSimplifiedWrapper(nn.Module):
    """
    Ultra-simplified wrapper for NNV compatibility.

    Key improvements:
    1. Folds BatchNorm into Conv layers (eliminates BN from ONNX)
    2. Uses static Flatten operation (no dynamic Shape+Reshape)
    3. Minimal layer count for faster verification

    This addresses the "13 layers" issue by eliminating BatchNorm entirely.
    """
    def __init__(self, model, fold_bn=True):
        super().__init__()

        model_type = type(model).__name__
        self.model_type = model_type
        self.num_classes = model.num_classes

        if model_type in ['MicroExpertCNN', 'NNVCompatibleCNN']:
            # Fold BatchNorm into Conv layers
            if fold_bn:
                self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
                self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
                self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
            else:
                self.conv1 = model.conv1
                self.bn1 = model.bn1
                self.conv2 = model.conv2
                self.bn2 = model.bn2
                self.conv3 = model.conv3
                self.bn3 = model.bn3

            self.pool1 = model.pool1
            self.pool2 = model.pool2
            self.pool3 = model.pool3
            self.fc = model.fc
            self.fold_bn = fold_bn

        elif model_type == 'UltraVerifiableCNN':
            # Ultra-verifiable architecture: 3 conv + global avg pool + 1 FC
            if fold_bn:
                self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
                self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
                self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
            else:
                self.conv1 = model.conv1
                self.bn1 = model.bn1
                self.conv2 = model.conv2
                self.bn2 = model.bn2
                self.conv3 = model.conv3
                self.bn3 = model.bn3

            self.global_avg_pool = model.global_avg_pool
            self.fc = model.fc
            self.fold_bn = fold_bn

        elif model_type == 'TinyExpertCNN':
            if fold_bn:
                self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
                self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
                self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
            else:
                self.conv1 = model.conv1
                self.bn1 = model.bn1
                self.conv2 = model.conv2
                self.bn2 = model.bn2
                self.conv3 = model.conv3
                self.bn3 = model.bn3

            self.pool1 = model.pool1
            self.pool2 = model.pool2
            self.pool3 = model.pool3
            self.fc1 = model.fc1
            self.fc2 = model.fc2
            self.dropout = model.dropout
            self.fold_bn = fold_bn

        elif model_type == 'SmallExpertCNN':
            if fold_bn:
                self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
                self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
                self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
                self.conv4 = fold_batch_norm_into_conv(model.conv4, model.bn4)
            else:
                self.conv1 = model.conv1
                self.bn1 = model.bn1
                self.conv2 = model.conv2
                self.bn2 = model.bn2
                self.conv3 = model.conv3
                self.bn3 = model.bn3
                self.conv4 = model.conv4
                self.bn4 = model.bn4

            self.pool1 = model.pool1
            self.pool2 = model.pool2
            self.pool3 = model.pool3
            self.pool4 = model.pool4
            self.fc1 = model.fc1
            self.fc2 = model.fc2
            self.dropout = model.dropout
            self.fold_bn = fold_bn

        else:
            raise ValueError(f"Unsupported model type for NNV export: {model_type}")

    def forward(self, x):
        """
        Forward pass with:
        1. Optional BatchNorm (if fold_bn=False)
        2. Static flatten operation (no dynamic view)
        """
        if self.model_type in ['MicroExpertCNN', 'NNVCompatibleCNN']:
            # Block 1
            x = self.conv1(x)
            if not self.fold_bn:
                x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)

            # Block 2
            x = self.conv2(x)
            if not self.fold_bn:
                x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)

            # Block 3
            x = self.conv3(x)
            if not self.fold_bn:
                x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)

            # Static flatten
            x = torch.flatten(x, 1)

            # FC layer
            x = self.fc(x)

        elif self.model_type == 'UltraVerifiableCNN':
            # Ultra-verifiable: 3 strided/regular convs + global avg pool
            # Block 1: strided conv
            x = self.conv1(x)
            if not self.fold_bn:
                x = self.bn1(x)
            x = F.relu(x)

            # Block 2: strided conv
            x = self.conv2(x)
            if not self.fold_bn:
                x = self.bn2(x)
            x = F.relu(x)

            # Block 3: regular conv
            x = self.conv3(x)
            if not self.fold_bn:
                x = self.bn3(x)
            x = F.relu(x)

            # Global average pooling
            x = self.global_avg_pool(x)

            # Static flatten
            x = torch.flatten(x, 1)

            # FC layer
            x = self.fc(x)

        elif self.model_type == 'TinyExpertCNN':
            # Block 1
            x = self.conv1(x)
            if not self.fold_bn:
                x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)

            # Block 2
            x = self.conv2(x)
            if not self.fold_bn:
                x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)

            # Block 3
            x = self.conv3(x)
            if not self.fold_bn:
                x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)

            # Static flatten
            x = torch.flatten(x, 1)

            # FC layers
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)

        elif self.model_type == 'SmallExpertCNN':
            # Block 1
            x = self.conv1(x)
            if not self.fold_bn:
                x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)

            # Block 2
            x = self.conv2(x)
            if not self.fold_bn:
                x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)

            # Block 3
            x = self.conv3(x)
            if not self.fold_bn:
                x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)

            # Block 4
            x = self.conv4(x)
            if not self.fold_bn:
                x = self.bn4(x)
            x = F.relu(x)
            x = self.pool4(x)

            # Static flatten
            x = torch.flatten(x, 1)

            # FC layers
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)

        return x


# Legacy wrapper (keep for compatibility)
class NNVCompatibleWrapper(nn.Module):
    """
    DEPRECATED: Use NNVSimplifiedWrapper instead for better NNV compatibility.

    Wrapper to make models NNV-compatible by replacing dynamic view operations
    with static reshape operations that MATLAB's ONNX importer can handle.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Determine the model type and set flatten size
        model_type = type(model).__name__

        if model_type == 'MicroExpertCNN':
            # MicroExpertCNN: after conv3+pool3, we have [B, 64, 4, 4]
            self.flatten_size = 64 * 4 * 4  # 1024
        elif model_type == 'NNVCompatibleCNN':
            # NNVCompatibleCNN: same structure as MicroExpertCNN but with AvgPool
            self.flatten_size = 64 * 4 * 4  # 1024
        elif model_type == 'TinyExpertCNN':
            # TinyExpertCNN: after conv3+pool3, we have [B, 128, 4, 4]
            self.flatten_size = 128 * 4 * 4  # 2048
        elif model_type == 'SmallExpertCNN':
            # SmallExpertCNN: after conv4+pool4, we have [B, 256, 2, 2]
            self.flatten_size = 256 * 2 * 2  # 1024
        else:
            raise ValueError(f"Unsupported model type for NNV export: {model_type}")

        self.model_type = model_type
        self.num_classes = model.num_classes

    def forward(self, x):
        """
        Forward pass with static reshaping instead of dynamic view()
        This recreates the forward pass without using x.view()
        """
        if self.model_type == 'MicroExpertCNN' or self.model_type == 'NNVCompatibleCNN':
            # Both MicroExpertCNN and NNVCompatibleCNN have the same structure
            # Block 1: Conv + BN + ReLU + Pool
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = F.relu(x)
            x = self.model.pool1(x)

            # Block 2: Conv + BN + ReLU + Pool
            x = self.model.conv2(x)
            x = self.model.bn2(x)
            x = F.relu(x)
            x = self.model.pool2(x)

            # Block 3: Conv + BN + ReLU + Pool
            x = self.model.conv3(x)
            x = self.model.bn3(x)
            x = F.relu(x)
            x = self.model.pool3(x)

            # Static flatten (no dynamic Shape operation)
            x = torch.flatten(x, 1)  # Flatten from dim 1 onwards

            # FC layer
            x = self.model.fc(x)

        elif self.model_type == 'TinyExpertCNN':
            # Block 1
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = F.relu(x)
            x = self.model.pool1(x)

            # Block 2
            x = self.model.conv2(x)
            x = self.model.bn2(x)
            x = F.relu(x)
            x = self.model.pool2(x)

            # Block 3
            x = self.model.conv3(x)
            x = self.model.bn3(x)
            x = F.relu(x)
            x = self.model.pool3(x)

            # Static flatten
            x = torch.flatten(x, 1)

            # FC layers
            x = self.model.fc1(x)
            x = F.relu(x)
            x = self.model.dropout(x)
            x = self.model.fc2(x)

        elif self.model_type == 'SmallExpertCNN':
            # Block 1
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = F.relu(x)
            x = self.model.pool1(x)

            # Block 2
            x = self.model.conv2(x)
            x = self.model.bn2(x)
            x = F.relu(x)
            x = self.model.pool2(x)

            # Block 3
            x = self.model.conv3(x)
            x = self.model.bn3(x)
            x = F.relu(x)
            x = self.model.pool3(x)

            # Block 4
            x = self.model.conv4(x)
            x = self.model.bn4(x)
            x = F.relu(x)
            x = self.model.pool4(x)

            # Static flatten
            x = torch.flatten(x, 1)

            # FC layers
            x = self.model.fc1(x)
            x = F.relu(x)
            x = self.model.dropout(x)
            x = self.model.fc2(x)

        return x


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

    # Wrap model for NNV compatibility
    model_type = type(model).__name__
    if model_type in ['MicroExpertCNN', 'TinyExpertCNN', 'SmallExpertCNN', 'NNVCompatibleCNN', 'UltraVerifiableCNN']:
        print(f"Wrapping {model_type} for NNV compatibility...")
        print("  - Folding BatchNorm into Conv layers (reduces layer count)")
        print("  - Using static Flatten operation (avoids Shape+Reshape)")

        if model_type == 'UltraVerifiableCNN':
            print("  - UltraVerifiableCNN: Minimal architecture for fast verification")
            print("    * Only 3 conv layers (16-24-32 channels)")
            print("    * Global average pooling (reduces parameters)")
            print("    * Expected NNV layers: ~5-6 (vs 13 for other models)")

        model = NNVSimplifiedWrapper(model, fold_bn=True)
        model.eval()
    else:
        print(f"Warning: {model_type} may not be NNV-compatible. Consider adding support.")

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
