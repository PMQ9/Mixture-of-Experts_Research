import torch
import os
import sys

PRETRAINED_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'artifacts', 'nnv'))
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Vision_Transformer_Pytorch')
sys.path.append(SRC_DIR)
model = torch.load(
    os.path.join(PRETRAINED_MODEL_DIR, "meta_moe_convnext_tiny_best_og_1.pth"),
    map_location='cpu'
)

print("Model type:", type(model))
print("\nParameter names:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")