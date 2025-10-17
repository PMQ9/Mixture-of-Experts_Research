"""
Test that BatchNorm folding produces the same outputs as the original model
"""
import torch
import sys
import os

# Add Vision Transformer directory to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Vision_Transformer_Pytorch'))
sys.path.append(SRC_DIR)

from export_to_onnx import NNVSimplifiedWrapper

# Load the original model
model_path = "artifacts/gtsrb_ultra_verifiable_cnn.pth"
device = torch.device('cpu')

print("Loading original model...")
original_model = torch.load(model_path, map_location=device, weights_only=False)
if hasattr(original_model, 'model'):
    original_model = original_model.model
original_model.eval()

print("Creating folded model...")
folded_model = NNVSimplifiedWrapper(original_model, fold_bn=True)
folded_model.eval()

# Test with random input
print("\nTesting with random inputs...")
torch.manual_seed(42)
test_inputs = torch.randn(5, 3, 32, 32)

with torch.no_grad():
    original_outputs = original_model(test_inputs)
    folded_outputs = folded_model(test_inputs)

# Compare outputs
max_diff = torch.max(torch.abs(original_outputs - folded_outputs)).item()
mean_diff = torch.mean(torch.abs(original_outputs - folded_outputs)).item()

print(f"\nOutput comparison:")
print(f"  Max absolute difference: {max_diff:.2e}")
print(f"  Mean absolute difference: {mean_diff:.2e}")

# Check if predictions are the same
original_preds = torch.argmax(original_outputs, dim=1)
folded_preds = torch.argmax(folded_outputs, dim=1)
same_preds = torch.all(original_preds == folded_preds).item()

print(f"  Same predictions: {same_preds}")
print(f"  Original predictions: {original_preds.tolist()}")
print(f"  Folded predictions: {folded_preds.tolist()}")

# Tolerance check
if max_diff < 1e-4:
    print("\n✓ BatchNorm folding is CORRECT (differences within numerical precision)")
else:
    print(f"\n✗ WARNING: Large differences detected (max diff: {max_diff:.2e})")
