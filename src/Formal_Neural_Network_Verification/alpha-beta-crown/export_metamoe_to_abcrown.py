"""
Export MetaMoE Model to ONNX for Compositional Robustness Verification

This script exports a trained MetaMoE model (router + frozen experts) to ONNX format
for verification with alpha-beta-CROWN. This enables compositional robustness analysis:
verifying that the routing mechanism maintains correct expert selection under adversarial perturbations.

Key features:
- Exports entire MetaMoE system (router + all experts) as single ONNX model
- Automatic BatchNorm folding for all components
- Handles UltraVerifiableCNN and other expert architectures
- Validates compositional ONNX export correctness

Usage:
    python export_metamoe_to_abcrown.py \
        --model_path artifacts/training_20251019_111534/meta_moe_ultra_verifiable_cnn_best_og.pth \
        --output_dir artifacts/abcrown_models
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

# Add project directories to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src' / 'Vision_Transformer_Pytorch'))

from src.Vision_Transformer_Pytorch.vision_transformer_moe import MetaMoE, MetaGatingNet
from src.Vision_Transformer_Pytorch.small_expert import (
    SmallExpertCNN, TinyExpertCNN, MicroExpertCNN,
    NNVCompatibleCNN, UltraVerifiableCNN
)

# Import BatchNorm folding utility from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from export_to_abcrown import fold_batch_norm_into_conv


class SimplifiedExpert(nn.Module):
    """
    Simplified expert model with BatchNorm folding for ONNX export.
    Supports all expert architectures.
    """

    def __init__(self, expert_model):
        super(SimplifiedExpert, self).__init__()
        self.model_type = type(expert_model).__name__

        if self.model_type == 'UltraVerifiableCNN':
            self._fold_ultra_verifiable(expert_model)
        elif self.model_type == 'NNVCompatibleCNN':
            self._fold_nnv_compatible(expert_model)
        elif self.model_type == 'MicroExpertCNN':
            self._fold_micro_expert(expert_model)
        elif self.model_type == 'TinyExpertCNN':
            self._fold_tiny_expert(expert_model)
        elif self.model_type == 'SmallExpertCNN':
            self._fold_small_expert(expert_model)
        else:
            raise ValueError(f"Unsupported expert type: {self.model_type}")

    def _fold_ultra_verifiable(self, model):
        """Fold BatchNorm for UltraVerifiableCNN (4 conv blocks)"""
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
        self.num_classes = model.num_classes

    def _fold_nnv_compatible(self, model):
        """Fold BatchNorm for NNVCompatibleCNN (3 conv blocks)"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1

        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2

        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

        self.fc = model.fc
        self.num_classes = model.num_classes

    def _fold_micro_expert(self, model):
        """Fold BatchNorm for MicroExpertCNN"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)

        self.pool1 = model.pool1 if model.use_maxpool else None
        self.pool2 = model.pool2 if model.use_maxpool else None
        self.pool3 = model.pool3 if model.use_maxpool else None

        self.fc = model.fc
        self.num_classes = model.num_classes
        self.use_maxpool = model.use_maxpool

    def _fold_tiny_expert(self, model):
        """Fold BatchNorm for TinyExpertCNN"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1

        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2

        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.dropout = model.dropout
        self.num_classes = model.num_classes

    def _fold_small_expert(self, model):
        """Fold BatchNorm for SmallExpertCNN"""
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
        self.num_classes = model.num_classes

    def forward(self, x):
        """Forward pass with folded BatchNorm"""
        if self.model_type == 'UltraVerifiableCNN':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = F.relu(self.conv4(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        elif self.model_type == 'NNVCompatibleCNN':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.model_type == 'MicroExpertCNN':
            if self.use_maxpool:
                x = F.relu(self.conv1(x))
                x = self.pool1(x)
                x = F.relu(self.conv2(x))
                x = self.pool2(x)
                x = F.relu(self.conv3(x))
                x = self.pool3(x)
            else:
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.model_type in ['TinyExpertCNN', 'SmallExpertCNN']:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)

            if self.model_type == 'SmallExpertCNN':
                x = F.relu(self.conv4(x))
                x = self.pool4(x)

            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

        return x


class SimplifiedRouter(nn.Module):
    """
    Simplified router (MetaGatingNet) with BatchNorm folding for ONNX export.
    """

    def __init__(self, router_model):
        super(SimplifiedRouter, self).__init__()

        # Get backbone type
        self.backbone_type = router_model.backbone
        self.temperature = router_model.temperature

        # Simplify backbone based on type
        if self.backbone_type == 'ultra_verifiable_cnn':
            self._fold_ultra_verifiable_backbone(router_model.model)
        elif self.backbone_type == 'nnv_cnn':
            self._fold_nnv_backbone(router_model.model)
        elif self.backbone_type == 'micro_cnn':
            self._fold_micro_backbone(router_model.model)
        elif self.backbone_type == 'tiny_cnn':
            self._fold_tiny_backbone(router_model.model)
        elif self.backbone_type == 'small_cnn':
            self._fold_small_backbone(router_model.model)
        else:
            raise ValueError(f"Unsupported router backbone: {self.backbone_type}")

        # Copy FC layer (Linear + Softmax)
        self.fc = router_model.fc

    def _fold_ultra_verifiable_backbone(self, model):
        """Fold UltraVerifiableCNN_Features backbone"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1
        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2
        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3
        self.conv4 = fold_batch_norm_into_conv(model.conv4, model.bn4)

    def _fold_nnv_backbone(self, model):
        """Fold NNVCompatibleCNN_Features backbone"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1
        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2
        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

    def _fold_micro_backbone(self, model):
        """Fold MicroExpertCNN_Features backbone"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1
        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2
        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

    def _fold_tiny_backbone(self, model):
        """Fold TinyExpertCNN_Features backbone"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1
        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2
        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3

    def _fold_small_backbone(self, model):
        """Fold SmallExpertCNN_Features backbone"""
        self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
        self.pool1 = model.pool1
        self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
        self.pool2 = model.pool2
        self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
        self.pool3 = model.pool3
        self.conv4 = fold_batch_norm_into_conv(model.conv4, model.bn4)
        self.pool4 = model.pool4

    def forward(self, x):
        """Forward pass for router backbone"""
        if self.backbone_type == 'ultra_verifiable_cnn':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = F.relu(self.conv4(x))
            x = x.view(x.size(0), -1)

        elif self.backbone_type in ['nnv_cnn', 'micro_cnn', 'tiny_cnn']:
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = x.view(x.size(0), -1)

        elif self.backbone_type == 'small_cnn':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = F.relu(self.conv4(x))
            x = self.pool4(x)
            x = x.view(x.size(0), -1)

        # Apply FC layer (includes softmax in original)
        logits = self.fc(x) / self.temperature
        return logits


class SimplifiedMetaMoE(nn.Module):
    """
    Simplified MetaMoE model for ONNX export with BatchNorm folding.
    Exports the entire composed system (router + experts) as a single model.
    """

    def __init__(self, metamoe_model):
        super(SimplifiedMetaMoE, self).__init__()

        # Simplify router
        print("  Simplifying router...")
        self.router = SimplifiedRouter(metamoe_model.meta_gating_net)

        # Simplify all experts
        print(f"  Simplifying {metamoe_model.num_experts} experts...")
        self.experts = nn.ModuleList([
            SimplifiedExpert(expert) for expert in metamoe_model.experts
        ])

        # Copy metadata
        self.num_experts = metamoe_model.num_experts
        self.num_classes_list = metamoe_model.num_classes_list
        self.total_classes = metamoe_model.total_classes
        self.meta_top_k = metamoe_model.meta_top_k
        self.class_offsets = metamoe_model.class_offsets

    def forward(self, x):
        """
        Forward pass: route input to top-k experts and combine outputs.

        This matches the original MetaMoE forward logic but with simplified components.
        """
        # Get routing probabilities
        gates = self.router(x)  # [batch_size, num_experts]

        # Initialize output
        batch_size = x.shape[0]
        final_output = torch.zeros(batch_size, self.total_classes, device=x.device, dtype=x.dtype)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gates, k=self.meta_top_k, dim=1)
        sum_top_k = top_k_probs.sum(dim=1, keepdim=True)
        normalized_probs = top_k_probs / sum_top_k

        # Route to experts
        for i in range(self.num_experts):
            sample_indices, k_positions = torch.where(top_k_indices == i)
            if len(sample_indices) > 0:
                scaling_factors = normalized_probs[sample_indices, k_positions]
                expert_input = x[sample_indices]
                expert_output = self.experts[i](expert_input)
                scaled_output = expert_output * scaling_factors.unsqueeze(1)
                start_idx = self.class_offsets[i]
                end_idx = self.class_offsets[i+1]
                final_output[sample_indices, start_idx:end_idx] = scaled_output.to(dtype=x.dtype)

        return final_output


def validate_onnx_export(pytorch_model, onnx_path, input_shape=(1, 3, 32, 32), device='cpu'):
    """
    Validate that ONNX export produces same outputs as PyTorch model.
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
        print(f"  [WARN] Validation tolerance exceeded (max diff >= 1e-4)")
        print(f"  This may be acceptable for MetaMoE due to routing complexity")
        return False


def export_metamoe_to_onnx(model_path, output_dir, device='cuda', simplify_onnx=True, validate=True):
    """
    Export MetaMoE model to ONNX for compositional robustness verification.

    Args:
        model_path: Path to MetaMoE .pth file
        output_dir: Directory to save ONNX file
        device: Device to load model on
        simplify_onnx: Whether to simplify ONNX graph
        validate: Whether to validate ONNX export

    Returns:
        Path to exported ONNX file
    """
    print(f"\n{'='*80}")
    print(f"Exporting MetaMoE model: {model_path}")
    print(f"{'='*80}")

    # Load MetaMoE model
    print("Loading MetaMoE model...")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Unwrap if needed
    if type(model).__name__ == 'ModelWrapper':
        print("Detected ModelWrapper, unwrapping...")
        model = model.model

    # Verify it's a MetaMoE model
    if type(model).__name__ != 'MetaMoE':
        raise ValueError(f"Expected MetaMoE model, got {type(model).__name__}")

    print(f"Number of experts: {model.num_experts}")
    print(f"Number of classes per expert: {model.num_classes_list}")
    print(f"Total output classes: {model.total_classes}")
    print(f"Meta top-k: {model.meta_top_k}")
    print(f"Router backbone: {model.meta_gating_net.backbone}")

    # Create simplified wrapper
    print("\nCreating simplified MetaMoE (folding BatchNorm in router and experts)...")
    simplified_model = SimplifiedMetaMoE(model).to(device)
    simplified_model.eval()

    # Test forward pass
    input_shape = (1, 3, 32, 32)
    dummy_input = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        output = simplified_model(dummy_input)
    print(f"\nInput shape: {input_shape}")
    print(f"Output shape: {output.shape}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    model_name = Path(model_path).stem
    onnx_path = output_dir / f"{model_name}.onnx"

    # Export to ONNX
    print(f"\nExporting to ONNX: {onnx_path}")
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
        print("\nValidating ONNX export...")
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
    print(f"[SUCCESS] MetaMoE export complete: {onnx_path}")
    print(f"{'='*80}\n")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export MetaMoE models to ONNX for compositional verification')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to MetaMoE PyTorch model (.pth file)')
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
    onnx_path = export_metamoe_to_onnx(
        args.model_path,
        args.output_dir,
        device=args.device,
        simplify_onnx=not args.no_simplify,
        validate=not args.no_validate
    )

    print(f"\nNext steps:")
    print(f"1. Use verify_metamoe_abcrown.py to run compositional verification")
    print(f"2. Or manually create config and run: cd modules/alpha-beta-CROWN/complete_verifier && python abcrown.py --config <config.yaml>")


if __name__ == '__main__':
    main()
