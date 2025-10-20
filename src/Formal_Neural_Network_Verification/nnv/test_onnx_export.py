"""
Test ONNX model export to verify correctness

This script loads an ONNX model and compares its outputs with the original
PyTorch model to ensure the export was successful.

Usage:
    python test_onnx_export.py --pth_model path/to/model.pth --onnx_model path/to/model.onnx
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
import sys
import argparse

# Add Vision Transformer directory to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Vision_Transformer_Pytorch'))
sys.path.append(SRC_DIR)


def test_onnx_export(pth_path, onnx_path, input_size=(1, 3, 32, 32), num_tests=10):
    """
    Test ONNX export by comparing PyTorch and ONNX outputs

    Args:
        pth_path: Path to PyTorch .pth model
        onnx_path: Path to ONNX .onnx model
        input_size: Input tensor size
        num_tests: Number of random inputs to test
    """
    print(f"\n{'='*80}")
    print("Testing ONNX Export")
    print(f"{'='*80}\n")

    # Load PyTorch model
    print(f"Loading PyTorch model: {pth_path}")
    device = torch.device('cpu')

    try:
        pth_model = torch.load(pth_path, map_location=device, weights_only=False)

        # Handle wrapper
        if hasattr(pth_model, 'model'):
            pth_model = pth_model.model

        pth_model.eval()
        print("✓ PyTorch model loaded")
    except Exception as e:
        print(f"✗ Error loading PyTorch model: {e}")
        return False

    # Verify ONNX model
    print(f"\nVerifying ONNX model: {onnx_path}")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False

    # Load ONNX runtime
    print("\nLoading ONNX Runtime session...")
    try:
        ort_session = ort.InferenceSession(onnx_path)
        print("✓ ONNX Runtime session created")

        # Print model info
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        print(f"  Input name: {input_name}")
        print(f"  Output name: {output_name}")
    except Exception as e:
        print(f"✗ Error creating ONNX Runtime session: {e}")
        return False

    # Run comparison tests
    print(f"\nRunning {num_tests} comparison tests...")
    print("-" * 80)

    all_passed = True
    max_diff_overall = 0.0

    for i in range(num_tests):
        # Generate random input
        test_input = torch.randn(input_size)

        # PyTorch inference
        with torch.no_grad():
            pth_output = pth_model(test_input).numpy()

        # ONNX inference
        ort_inputs = {input_name: test_input.numpy()}
        onnx_output = ort_session.run([output_name], ort_inputs)[0]

        # Compare outputs
        diff = np.abs(pth_output - onnx_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        max_diff_overall = max(max_diff_overall, max_diff)

        # Check tolerance (FP32 precision)
        tolerance = 1e-5
        passed = max_diff < tolerance

        status = "✓" if passed else "✗"
        print(f"Test {i+1:2d}: {status}  Max diff: {max_diff:.2e}  Mean diff: {mean_diff:.2e}")

        if not passed:
            all_passed = False
            print(f"         WARNING: Difference exceeds tolerance ({tolerance})")

    print("-" * 80)

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {num_tests}")
    print(f"Maximum difference: {max_diff_overall:.2e}")

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("ONNX export is correct and ready for NNV verification!")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("ONNX export may have issues. Check model compatibility.")

    print(f"{'='*80}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Test ONNX model export correctness')
    parser.add_argument('--pth_model', type=str, required=True,
                       help='Path to PyTorch .pth model')
    parser.add_argument('--onnx_model', type=str, required=True,
                       help='Path to ONNX .onnx model')
    parser.add_argument('--input_size', type=str, default='1,3,32,32',
                       help='Input size as "B,C,H,W"')
    parser.add_argument('--num_tests', type=int, default=10,
                       help='Number of random tests to run')

    args = parser.parse_args()

    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))

    # Check files exist
    if not os.path.exists(args.pth_model):
        print(f"Error: PyTorch model not found: {args.pth_model}")
        return 1

    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model not found: {args.onnx_model}")
        return 1

    # Run test
    success = test_onnx_export(args.pth_model, args.onnx_model, input_size, args.num_tests)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
