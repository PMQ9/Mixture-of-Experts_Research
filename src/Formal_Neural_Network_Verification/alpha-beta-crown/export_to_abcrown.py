"""
Export PyTorch Expert Models to ONNX for alpha-beta-CROWN Verification

This script exports trained expert models to ONNX format optimized for alpha-beta-CROWN.
Unlike NNV, alpha-beta-CROWN can handle larger models efficiently.

Key features:
- Automatic BatchNorm folding into Conv layers
- Simplified ONNX graph (no dynamic shapes)
- Support for all expert architectures
- Validation of ONNX export correctness

Usage:
    python export_to_abcrown.py --model_path artifacts/gtsrb_small_cnn_best.pth --output_dir artifacts/abcrown_models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxsim import simplify
import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Add src directory to path for imports (go up 3 levels to project root)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src' / 'Vision_Transformer_Pytorch'))  # For model_wrapper

from src.Vision_Transformer_Pytorch.small_expert import (
    SmallExpertCNN, TinyExpertCNN, MicroExpertCNN,
    NNVCompatibleCNN, UltraVerifiableCNN
)


def fold_batch_norm_into_conv(conv, bn):
    """
    Fold BatchNorm parameters into Conv layer weights and biases.

    Mathematical equivalence:
    BN(Conv(x)) = gamma * (Conv(x) - mean) / sqrt(var + eps) + beta
                = (gamma / sqrt(var + eps)) * Conv(x) + (beta - gamma * mean / sqrt(var + eps))
                = Conv'(x) + bias'

    where:
    - Conv' has weights: conv.weight * (gamma / sqrt(var + eps))
    - bias' = beta - gamma * mean / sqrt(var + eps) + conv.bias * (gamma / sqrt(var + eps))

    Args:
        conv: nn.Conv2d layer
        bn: nn.BatchNorm2d layer

    Returns:
        nn.Conv2d with folded BatchNorm parameters
    """
    # Create new Conv2d with bias
    folded_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True
    )

    # Get BatchNorm parameters
    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_gamma = bn.weight
    bn_beta = bn.bias
    bn_eps = bn.eps

    # Compute scaling factor: gamma / sqrt(var + eps)
    bn_std = torch.sqrt(bn_var + bn_eps)
    scale = bn_gamma / bn_std

    # Fold into conv weights: w_new = w_old * scale
    # Shape: [out_channels, in_channels, kernel_h, kernel_w]
    folded_conv.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)

    # Fold into conv bias: b_new = beta - gamma * mean / sqrt(var + eps) + b_old * scale
    if conv.bias is not None:
        folded_conv.bias.data = bn_beta - bn_gamma * bn_mean / bn_std + conv.bias * scale
    else:
        folded_conv.bias.data = bn_beta - bn_gamma * bn_mean / bn_std

    return folded_conv


class SimplifiedWrapper(nn.Module):
    """
    Wrapper that applies BatchNorm folding for ONNX export.
    Removes BatchNorm layers by folding them into Conv layers.
    """

    def __init__(self, model):
        super(SimplifiedWrapper, self).__init__()
        self.model_type = type(model).__name__

        if self.model_type == 'SmallExpertCNN':
            self._fold_small_expert(model)
        elif self.model_type == 'TinyExpertCNN':
            self._fold_tiny_expert(model)
        elif self.model_type == 'MicroExpertCNN':
            self._fold_micro_expert(model)
        elif self.model_type == 'NNVCompatibleCNN':
            self._fold_nnv_compatible(model)
        elif self.model_type == 'UltraVerifiableCNN':
            self._fold_ultra_verifiable(model)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _fold_small_expert(self, model):
        """Fold BatchNorm for SmallExpertCNN (4 conv blocks)"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1

        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2

        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

        self.conv4 = fold_batch_norm_into_conv(model.conv4, model.bn4)
        self.pool4 = model.pool4

        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.dropout = model.dropout
        self.num_blocks = 4

    def _fold_tiny_expert(self, model):
        """Fold BatchNorm for TinyExpertCNN (3 conv blocks)"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1

        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2

        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.dropout = model.dropout
        self.num_blocks = 3

    def _fold_micro_expert(self, model):
        """Fold BatchNorm for MicroExpertCNN (3 conv blocks)"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        if model.use_maxpool:
            self.pool1 = model.pool1
        else:
            self.pool1 = None

        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        if model.use_maxpool:
            self.pool2 = model.pool2
        else:
            self.pool2 = None

        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        if model.use_maxpool:
            self.pool3 = model.pool3
        else:
            self.pool3 = None

        self.fc = model.fc
        self.num_blocks = 3
        self.use_maxpool = model.use_maxpool

    def _fold_nnv_compatible(self, model):
        """Fold BatchNorm for NNVCompatibleCNN (3 conv blocks with AvgPool)"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1

        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2

        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

        self.fc = model.fc
        self.num_blocks = 3

    def _fold_ultra_verifiable(self, model):
        """Fold BatchNorm for UltraVerifiableCNN (4 conv blocks with AvgPool)"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1

        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2

        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

        self.conv4 = fold_batch_norm_into_conv(model.conv4, model.bn4)

        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.dropout = model.dropout
        self.num_blocks = 4

    def forward(self, x):
        """Forward pass with folded BatchNorm (no BN layers)"""
        if self.model_type in ['SmallExpertCNN', 'TinyExpertCNN']:
            # SmallExpert and TinyExpert have similar structure
            x = F.relu(self.conv1(x))
            x = self.pool1(x)

            x = F.relu(self.conv2(x))
            x = self.pool2(x)

            x = F.relu(self.conv3(x))
            x = self.pool3(x)

            if self.num_blocks == 4:
                x = F.relu(self.conv4(x))
                x = self.pool4(x)

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            # Dropout is always disabled for verification (eval mode)
            x = self.fc2(x)

        elif self.model_type == 'MicroExpertCNN':
            if self.use_maxpool:
                x = F.relu(self.conv1(x))
                x = self.pool1(x)

                x = F.relu(self.conv2(x))
                x = self.pool2(x)

                x = F.relu(self.conv3(x))
                x = self.pool3(x)
            else:
                # Strided conv instead of pooling
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))

            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.model_type == 'NNVCompatibleCNN':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)

            x = F.relu(self.conv2(x))
            x = self.pool2(x)

            x = F.relu(self.conv3(x))
            x = self.pool3(x)

            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.model_type == 'UltraVerifiableCNN':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)

            x = F.relu(self.conv2(x))
            x = self.pool2(x)

            x = F.relu(self.conv3(x))
            x = self.pool3(x)

            x = F.relu(self.conv4(x))

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            # Dropout is always disabled for verification (eval mode)
            x = self.fc2(x)

        return x


def validate_onnx_export(pytorch_model, onnx_path, input_shape=(1, 3, 32, 32), device='cpu'):
    """
    Validate that ONNX export produces same outputs as PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        input_shape: Input tensor shape
        device: Device for PyTorch model

    Returns:
        bool: True if validation passes
    """
    import onnxruntime as ort

    # Generate random input
    dummy_input = torch.randn(*input_shape).to(device)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    # ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"  Max difference between PyTorch and ONNX: {max_diff:.2e}")

    if max_diff < 1e-4:
        print(f"  [OK] Validation PASSED (max diff < 1e-4)")
        return True
    else:
        print(f"  [FAIL] Validation FAILED (max diff >= 1e-4)")
        return False


def export_model_to_onnx(model_path, output_dir, device='cuda', simplify_onnx=True, validate=True, input_shape=None):
    """
    Export a trained PyTorch model to ONNX format for alpha-beta-CROWN.

    Args:
        model_path: Path to .pth model file
        output_dir: Directory to save ONNX file
        device: Device to load model on
        simplify_onnx: Whether to simplify ONNX graph
        validate: Whether to validate ONNX export
        input_shape: Input tensor shape (B, C, H, W). If None, auto-detect from model path

    Returns:
        Path to exported ONNX file
    """
    print(f"\n{'='*80}")
    print(f"Exporting model: {model_path}")
    print(f"{'='*80}")

    # Load model
    print("Loading PyTorch model...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Unwrap ModelWrapper if needed
    if type(model).__name__ == 'ModelWrapper':
        print("Detected ModelWrapper, unwrapping...")
        model = model.model  # Get the actual model from wrapper

    # Detect model type
    model_type = type(model).__name__
    print(f"Model type: {model_type}")

    # Get number of classes
    if hasattr(model, 'num_classes'):
        num_classes = model.num_classes
    elif hasattr(model, 'fc2'):
        num_classes = model.fc2.out_features
    elif hasattr(model, 'fc'):
        num_classes = model.fc.out_features
    else:
        raise ValueError(f"Cannot determine number of classes for {model_type}")

    print(f"Number of classes: {num_classes}")

    # Create simplified wrapper with BatchNorm folding
    print("Creating simplified wrapper (folding BatchNorm)...")
    simplified_model = SimplifiedWrapper(model).to(device)
    simplified_model.eval()

    # Auto-detect input shape from model path if not provided
    if input_shape is None:
        model_name_lower = Path(model_path).name.lower()

        # Check if model was trained on MNIST (but note: expert architectures still use 3 channels)
        # MNIST images are converted from 1->3 channels during data loading for compatibility
        if 'mnist' in model_name_lower:
            # All expert architectures expect 3-channel input (RGB)
            # MNIST grayscale images are converted to 3 channels during training
            input_shape = (1, 3, 32, 32)  # 3 channels (converted from 1), 32x32 (padded from 28x28)
            print(f"Auto-detected MNIST dataset: input_shape = {input_shape}")
            print(f"  Note: MNIST models use 3-channel input (grayscale duplicated to RGB)")
        elif 'gtsrb' in model_name_lower or 'cifar' in model_name_lower:
            input_shape = (1, 3, 32, 32)  # GTSRB/CIFAR: RGB, 32x32
            print(f"Auto-detected GTSRB/CIFAR dataset: input_shape = {input_shape}")
        else:
            # Default to CIFAR/GTSRB size
            input_shape = (1, 3, 32, 32)
            print(f"Using default input_shape = {input_shape}")

    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)

    # Test forward pass
    with torch.no_grad():
        output = simplified_model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    model_name = Path(model_path).stem
    onnx_path = output_dir / f"{model_name}.onnx"

    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        simplified_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Fixed batch size for verification
    )

    # Simplify ONNX graph
    if simplify_onnx:
        print("Simplifying ONNX graph...")
        onnx_model = onnx.load(str(onnx_path))
        simplified_onnx, check = simplify(onnx_model)

        if check:
            onnx.save(simplified_onnx, str(onnx_path))
            print("  [OK] ONNX simplification successful")
        else:
            print("  [WARN] ONNX simplification failed, using original")

    # Validate ONNX export
    if validate:
        print("Validating ONNX export...")
        validate_onnx_export(simplified_model, str(onnx_path), input_shape=input_shape, device=device)

    # Print ONNX stats
    onnx_model = onnx.load(str(onnx_path))
    print(f"\nONNX Model Statistics:")
    print(f"  Operators: {len(onnx_model.graph.node)}")
    print(f"  Inputs: {len(onnx_model.graph.input)}")
    print(f"  Outputs: {len(onnx_model.graph.output)}")
    print(f"  Initializers (weights): {len(onnx_model.graph.initializer)}")

    # Count operator types
    op_types = {}
    for node in onnx_model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    print(f"\nOperator Types:")
    for op, count in sorted(op_types.items()):
        print(f"  {op}: {count}")

    print(f"\n{'='*80}")
    print(f"[SUCCESS] Export complete: {onnx_path}")
    print(f"{'='*80}\n")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX for alpha-beta-CROWN')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to PyTorch model (.pth file)')
    parser.add_argument('--output_dir', type=str, default='artifacts/abcrown_models',
                        help='Output directory for ONNX files')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no_simplify', action='store_true',
                        help='Disable ONNX simplification')
    parser.add_argument('--no_validate', action='store_true',
                        help='Disable ONNX validation')

    args = parser.parse_args()

    # Export model
    onnx_path = export_model_to_onnx(
        args.model_path,
        args.output_dir,
        device=args.device,
        simplify_onnx=not args.no_simplify,
        validate=not args.no_validate
    )

    print(f"\nNext steps:")
    print(f"1. Create verification config: see modules/alpha-beta-CROWN/complete_verifier/exp_configs/")
    print(f"2. Run verification: cd modules/alpha-beta-CROWN/complete_verifier && python abcrown.py --config <config.yaml>")


if __name__ == '__main__':
    main()
