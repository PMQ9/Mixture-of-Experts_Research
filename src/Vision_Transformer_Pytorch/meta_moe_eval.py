import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from torchvision import transforms
from vision_transformer_moe import MetaMoE, VisionTransformer, MetaGatingNet, VisionTransformerConfig
from config import (
    NORM_MEAN_R_UNIFIED, NORM_MEAN_G_UNIFIED, NORM_MEAN_B_UNIFIED,
    NORM_STD_R_UNIFIED, NORM_STD_G_UNIFIED, NORM_STD_B_UNIFIED
)

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser
parser = argparse.ArgumentParser(description='Evaluate MetaMoE model on a single image')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained MetaMoE model (e.g., vit_meta_moe_best.pth)')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image (JPG or PPM)')
args = parser.parse_args()

# Load the model
model = torch.load(args.model_path, map_location=DEVICE)
model = model.to(DEVICE)
model.eval()

# Define the transformation using unified normalization
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(NORM_MEAN_R_UNIFIED, NORM_MEAN_G_UNIFIED, NORM_MEAN_B_UNIFIED),
        std=(NORM_STD_R_UNIFIED, NORM_STD_G_UNIFIED, NORM_STD_B_UNIFIED)
    )
])

# Load and preprocess the input image
try:
    image = Image.open(args.image_path).convert('RGB')
except Exception as e:
    print(f"Error opening image: {e}")
    exit(1)

image = transform(image).unsqueeze(0).to(DEVICE)

# Pass the image through the model
with torch.no_grad():
    output, gates = model(image)

# Compute predictions
# Class prediction
probabilities = F.softmax(output, dim=1)
confidence, predicted_class = probabilities.max(1)
predicted_class = predicted_class.item()
confidence = confidence.item()

# Determine dataset and class ID
num_classes_gtsrb = 43  # Number of classes in GTSRB
if predicted_class < num_classes_gtsrb:
    dataset = 'GTSRB'
    class_id = predicted_class
else:
    dataset = 'PTSD'
    class_id = predicted_class - num_classes_gtsrb

# Meta_class prediction
meta_prob, predicted_meta_class = gates.max(1)
predicted_meta_class = predicted_meta_class.item()
meta_confidence = meta_prob.item()
meta_dataset = 'GTSRB' if predicted_meta_class == 0 else 'PTSD'

# Output results
print(f"Predicted meta_class: {meta_dataset} with confidence {meta_confidence:.4f}")
print(f"Predicted class ID: {class_id} in {dataset} with confidence {confidence:.4f}")
print(f"Gating weights: W_G={gates[0,0].item():.4f}, W_P={gates[0,1].item():.4f}")