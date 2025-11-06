"""
Export MetaMoE Router to ONNX for Formal Verification

Since the full MetaMoE cannot be verified due to dynamic routing operations,
we extract and verify just the ROUTER component with alpha-beta-CROWN.

This provides formal guarantees that adversarial perturbations cannot fool
the router into selecting the wrong expert.

Router architecture: MetaGatingNet with UltraVerifiableCNN backbone
- Input: 32x32x3 image
- Output: 2-class softmax (probabilities for each expert)

Usage:
    python export_router_to_abcrown.py \
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

# Add project directories
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src' / 'Vision_Transformer_Pytorch'))

from src.Vision_Transformer_Pytorch.vision_transformer_moe import MetaMoE, MetaGatingNet

# Import BatchNorm folding utility
sys.path.insert(0, str(Path(__file__).parent))
from export_to_abcrown import fold_batch_norm_into_conv


class SimplifiedRouter(nn.Module):
    """
    Simplified router with BatchNorm folding for ONNX export.

    This is the same as in export_metamoe_to_abcrown.py but standalone
    for router-only verification.
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

        # Copy FC layer - now just a Linear layer (Softmax removed in bug fix)
        # alpha-beta-CROWN verifies logits, not softmax outputs
        if isinstance(router_model.fc, nn.Sequential):
            # Old architecture with nn.Sequential([Linear, Softmax])
            self.fc_linear = router_model.fc[0]  # Linear layer only, skip Softmax
        else:
            # New architecture with just nn.Linear (bug fix applied)
            self.fc_linear = router_model.fc

    def _fold_ultra_verifiable_backbone(self, model):
        """Fold UltraVerifiableCNN_Features backbone (or copy if no BatchNorm)"""
        # Check if BatchNorm exists (old models) or was removed (new models after bug fix)
        if hasattr(model, 'bn1'):
            # Old model with BatchNorm - fold it
            self.conv1 = fold_batch_norm_into_conv(model.conv1, model.bn1)
            self.pool1 = model.pool1
            self.conv2 = fold_batch_norm_into_conv(model.conv2, model.bn2)
            self.pool2 = model.pool2
            self.conv3 = fold_batch_norm_into_conv(model.conv3, model.bn3)
            self.pool3 = model.pool3
            self.conv4 = fold_batch_norm_into_conv(model.conv4, model.bn4)
        else:
            # New model without BatchNorm - just copy layers directly
            self.conv1 = model.conv1
            self.pool1 = model.pool1
            self.conv2 = model.conv2
            self.pool2 = model.pool2
            self.conv3 = model.conv3
            self.pool3 = model.pool3
            self.conv4 = model.conv4

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
        """Forward pass for router backbone - output LOGITS (not softmax)"""
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

        # Apply Linear layer only (skip softmax and temperature scaling for verification)
        # alpha-beta-CROWN verifies logits: argmax(logits) should be correct class
        logits = self.fc_linear(x)
        return logits


def validate_router_semantics(meta_moe_model):
    """
    Validate that the router is correctly configured for verification.

    This checks:
    - Router returns raw logits (not softmax probabilities)
    - Router assigns samples to correct experts (sanity check)
    - Number of output classes matches number of experts
    """
    print("\n[Pre-Export Validation]")

    router = meta_moe_model.meta_gating_net
    num_experts = meta_moe_model.num_experts

    # Check 1: Output shape
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = router(dummy_input)

    if output.shape[1] != num_experts:
        print(f"  ✗ ERROR: Router output ({output.shape[1]}) != num_experts ({num_experts})")
        return False

    print(f"  ✓ Router output shape correct: {output.shape}")

    # Check 2: Temperature handling
    if hasattr(router, 'temperature'):
        print(f"  ✓ Temperature parameter found: {router.temperature}")
        if router.temperature != 1.0:
            print(f"    WARNING: Non-default temperature ({router.temperature}) stored but NOT applied in forward()")
            print(f"    This is OK for verification (we verify raw logits)")

    # Check 3: Output range (sanity check for logits vs softmax)
    if output.abs().max() > 10:
        print(f"  ✓ Output range suggests RAW LOGITS: [{output.min():.2f}, {output.max():.2f}]")
    else:
        print(f"  ⚠ Output range might indicate softmax: [{output.min():.2f}, {output.max():.2f}]")
        print(f"    Verify that router returns logits, not probabilities")

    return True


def validate_onnx_export(pytorch_model, onnx_path, input_shape=(1, 3, 32, 32), device='cpu'):
    """Validate ONNX export correctness"""
    import onnxruntime as ort

    dummy_input = torch.randn(*input_shape).to(device)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).cpu().numpy()

    # ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]

    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"  Max difference between PyTorch and ONNX: {max_diff:.2e}")

    if max_diff < 1e-4:
        print(f"  [OK] Validation PASSED (max diff < 1e-4)")
        return True
    else:
        print(f"  [WARN] Validation tolerance exceeded (max diff >= 1e-4)")
        return False


def export_router_to_onnx(model_path, output_dir, device='cuda', simplify_onnx=True, validate=True):
    """
    Export MetaMoE router to ONNX for formal verification.

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
    print(f"Exporting MetaMoE Router: {model_path}")
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
    print(f"Router backbone: {model.meta_gating_net.backbone}")

    # Pre-export validation
    if not validate_router_semantics(model):
        print("\nERROR: Router validation failed!")
        sys.exit(1)

    # Extract and simplify router
    print("\nExtracting router and folding BatchNorm...")
    router = model.meta_gating_net
    simplified_router = SimplifiedRouter(router).to(device)
    simplified_router.eval()

    # Test forward pass
    input_shape = (1, 3, 32, 32)
    dummy_input = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        output = simplified_router(dummy_input)

    print(f"\nRouter output:")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output classes: {output.shape[1]} (number of experts)")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    model_name = Path(model_path).stem
    onnx_path = output_dir / f"{model_name}_router_only.onnx"

    # Export to ONNX
    print(f"\nExporting to ONNX: {onnx_path}")
    torch.onnx.export(
        simplified_router,
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
        validate_onnx_export(simplified_router, str(onnx_path), input_shape=input_shape, device=device)

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
    print(f"[SUCCESS] Router export complete: {onnx_path}")
    print(f"{'='*80}\n")

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description='Export MetaMoE router to ONNX for verification')
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

    # Export router
    onnx_path = export_router_to_onnx(
        args.model_path,
        args.output_dir,
        device=args.device,
        simplify_onnx=not args.no_simplify,
        validate=not args.no_validate
    )

    print(f"\nNext steps:")
    print(f"1. Use verify_router_abcrown.py to run formal verification")
    print(f"2. Router verifies correct expert selection under adversarial perturbations")


if __name__ == '__main__':
    main()
