"""
Export MetaMoE Router (MetaGatingNet) to ONNX for formal verification

This script extracts the router/gating network from a trained MetaMoE model
and exports it to ONNX format for verification with NNV.

Usage:
    # Export router from MetaMoE model
    python export_router_to_onnx.py \
        --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
        --output_dir artifacts/nnv_models

    # Test with different input size
    python export_router_to_onnx.py \
        --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
        --output_dir artifacts/nnv_models \
        --input_size "1,3,48,48"
"""

import torch
import torch.nn as nn
import onnx
import os
import sys
import argparse
from pathlib import Path

# Add Vision Transformer directory to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Vision_Transformer_Pytorch'))
sys.path.append(SRC_DIR)


class RouterWrapper(nn.Module):
    """
    Wrapper for MetaGatingNet that outputs raw logits before softmax
    This is better for verification as we can verify the argmax decision
    """
    def __init__(self, meta_gating_net, output_logits=False):
        super().__init__()
        self.backbone = meta_gating_net.model
        self.fc = meta_gating_net.fc[0]  # Linear layer only (no softmax)
        self.temperature = meta_gating_net.temperature
        self.output_logits = output_logits

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features) / self.temperature

        if self.output_logits:
            # Output raw logits for verification
            return logits
        else:
            # Output probabilities (softmax)
            return torch.softmax(logits, dim=1)


def export_router_to_onnx(meta_moe_path, output_path, input_size=(1, 3, 32, 32),
                          output_logits=True, opset_version=11):
    """
    Extract and export the router from MetaMoE to ONNX

    Args:
        meta_moe_path: Path to MetaMoE .pth model file
        output_path: Path to save router .onnx file
        input_size: Input tensor size (batch, channels, height, width)
        output_logits: If True, output logits; if False, output probabilities
        opset_version: ONNX opset version
    """
    print(f"\n{'='*80}")
    print(f"Extracting Router from: {meta_moe_path}")
    print(f"{'='*80}")

    # Load MetaMoE model
    device = torch.device('cpu')

    try:
        meta_moe = torch.load(meta_moe_path, map_location=device, weights_only=False)

        # Handle different wrapper types
        if hasattr(meta_moe, 'model'):
            meta_moe = meta_moe.model

        # Extract the router (MetaGatingNet)
        if not hasattr(meta_moe, 'meta_gating_net'):
            print("ERROR: Model does not have 'meta_gating_net' attribute.")
            print("This is not a MetaMoE model!")
            return False

        router = meta_moe.meta_gating_net
        print(f"✓ Extracted MetaGatingNet")
        print(f"  Backbone: {router.backbone}")
        print(f"  Num experts: {router.fc[0].out_features}")
        print(f"  Temperature: {router.temperature}")

    except Exception as e:
        print(f"ERROR loading MetaMoE model: {e}")
        return False

    # Wrap router for export
    router_wrapper = RouterWrapper(router, output_logits=output_logits)
    router_wrapper.eval()

    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    print(f"\nInput size: {input_size}")

    # Test forward pass
    try:
        with torch.no_grad():
            output = router_wrapper(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output type: {'logits' if output_logits else 'probabilities'}")

        # Show output statistics
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        if not output_logits:
            print(f"Sum of probabilities: {output.sum(dim=1).item():.4f} (should be ~1.0)")

    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        return False

    # Export to ONNX
    try:
        output_mode = "logits" if output_logits else "probs"

        torch.onnx.export(
            router_wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=[output_mode],
            dynamic_axes={
                'input': {0: 'batch_size'},
                output_mode: {0: 'batch_size'}
            }
        )
        print(f"\n✓ Successfully exported to: {output_path}")

        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")

        # Print model info
        print(f"\nONNX Model Info:")
        print(f"  Inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"  Outputs: {[output.name for output in onnx_model.graph.output]}")
        print(f"  Operators: {len(onnx_model.graph.node)}")

        return True

    except Exception as e:
        print(f"ERROR during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_router_info(meta_moe_path):
    """
    Extract and display router information without exporting
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Router from: {meta_moe_path}")
    print(f"{'='*80}")

    device = torch.device('cpu')

    try:
        meta_moe = torch.load(meta_moe_path, map_location=device, weights_only=False)

        if hasattr(meta_moe, 'model'):
            meta_moe = meta_moe.model

        if not hasattr(meta_moe, 'meta_gating_net'):
            print("ERROR: Not a MetaMoE model!")
            return

        router = meta_moe.meta_gating_net

        print(f"\nRouter Architecture:")
        print(f"  Type: MetaGatingNet")
        print(f"  Backbone: {router.backbone}")
        print(f"  Temperature: {router.temperature}")
        print(f"  Num experts: {router.fc[0].out_features}")

        # Get feature dimension
        feature_dim = router.model.num_features
        print(f"  Feature dimension: {feature_dim}")

        # Count parameters
        total_params = sum(p.numel() for p in router.parameters())
        trainable_params = sum(p.numel() for p in router.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Test with dummy input
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = router(dummy_input)

        print(f"\nTest Forward Pass:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values: {output[0].cpu().numpy()}")
        print(f"  Predicted expert: {output.argmax(dim=1).item()}")

        # Show expert info if available
        if hasattr(meta_moe, 'num_experts'):
            print(f"\nMetaMoE Info:")
            print(f"  Number of experts: {meta_moe.num_experts}")
            print(f"  Meta top-k: {meta_moe.meta_top_k}")
            print(f"  Total classes: {meta_moe.total_classes}")
            print(f"  Class offsets: {meta_moe.class_offsets}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Export MetaMoE router to ONNX for formal verification'
    )
    parser.add_argument('--meta_moe_path', type=str, required=True,
                       help='Path to MetaMoE .pth model file')
    parser.add_argument('--output_dir', type=str, default='artifacts/nnv_models',
                       help='Output directory for router ONNX model')
    parser.add_argument('--input_size', type=str, default='1,3,32,32',
                       help='Input size as "B,C,H,W"')
    parser.add_argument('--output_logits', action='store_true',
                       help='Output logits instead of probabilities (recommended for verification)')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--info_only', action='store_true',
                       help='Only display router info without exporting')

    args = parser.parse_args()

    # Check if MetaMoE model exists
    if not os.path.exists(args.meta_moe_path):
        print(f"ERROR: MetaMoE model not found: {args.meta_moe_path}")
        return 1

    # Info only mode
    if args.info_only:
        extract_router_info(args.meta_moe_path)
        return 0

    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine output filename
    base_name = os.path.splitext(os.path.basename(args.meta_moe_path))[0]
    output_mode = "logits" if args.output_logits else "probs"
    output_filename = f"{base_name}_router_{output_mode}.onnx"
    output_path = os.path.join(args.output_dir, output_filename)

    # Export router
    success = export_router_to_onnx(
        args.meta_moe_path,
        output_path,
        input_size,
        args.output_logits,
        args.opset_version
    )

    if success:
        print(f"\n{'='*80}")
        print("SUCCESS: Router exported successfully!")
        print(f"{'='*80}")
        print(f"\nRouter ONNX model: {output_path}")
        print(f"\nNext steps:")
        print(f"1. Load in NNV: net = onnx2nnv('{output_path}')")
        print(f"2. Verify routing stability under perturbations")
        print(f"3. Combine with expert verification for end-to-end guarantees")
        return 0
    else:
        print(f"\n{'='*80}")
        print("FAILED: Router export failed")
        print(f"{'='*80}")
        return 1


if __name__ == '__main__':
    exit(main())
