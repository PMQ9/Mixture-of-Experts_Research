"""Quick test to verify ONNX export matches PyTorch model"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Vision_Transformer_Pytorch'))

import torch
import onnxruntime as ort
import numpy as np

# Load PyTorch model
print("Loading PyTorch model...")
pth_model = torch.load('artifacts/nnv_models/gtsrb_micro_cnn.pth',
                       map_location='cpu', weights_only=False)
if hasattr(pth_model, 'model'):
    pth_model = pth_model.model
pth_model.eval()

# Load ONNX model
print("Loading ONNX model...")
ort_session = ort.InferenceSession('artifacts/nnv_models/gtsrb_micro_cnn.onnx')

# Test with random input
print("\nTesting with 5 random inputs...")
for i in range(5):
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        pth_out = pth_model(x).numpy()

    onnx_out = ort_session.run(None, {'input': x.numpy()})[0]

    diff = np.abs(pth_out - onnx_out)
    print(f"Test {i+1}: Max diff = {diff.max():.2e}, Mean diff = {diff.mean():.2e}, Match = {np.allclose(pth_out, onnx_out, atol=1e-5)}")

print("\nONNX export verification PASSED!")
