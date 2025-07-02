# inspect_model.py
import torch
import timm
from torchvision import models
from vision_transformer_moe import VisionTransformer
from train_moe import ModelWrapper

model_path = "C:\\Users\\admin\\Documents\\Project\\Mixture-of-Experts_Research\\artifacts\\results\\vit_tsrd_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = torch.load(model_path, map_location=device, weights_only=False)
print(f"Model type: {type(model)}")

# Check output shape
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    output = model(dummy_input)
    if isinstance(output, tuple):
        output = output[0]  # Handle VisionTransformer tuple output
    print(f"Output shape: {output.shape}")