import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import argparse
import logging

# Import from vision_transformer_moe
from vision_transformer_moe import MetaGatingNet, MetaMoE

# Define ModelWrapper locally (for compatibility with non-MetaMoE models)
class ModelWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x):
        output = self.model(x)
        return output  # Simplified; assumes logits for non-MetaMoE models

# Normalization parameters (as provided in the original script)
GTSRB_NORM = {'mean': [0.3403, 0.3121, 0.3214], 'std': [0.2724, 0.2608, 0.2669]}
PTSD_NORM = {'mean': [0.3403, 0.3121, 0.3214], 'std': [0.2724, 0.2608, 0.2669]}
UNIFIED_NORM = {'mean': [0.3403, 0.3121, 0.3214], 'std': [0.2724, 0.2608, 0.2669]}

def evaluate_model(model_path, image_path, model_type='gtsrb', device='cpu'):
    """
    Loads a trained .pth model and evaluates it on a given image.
    
    Parameters:
    - model_path (str): Path to the .pth model file.
    - image_path (str): Path to the input image.
    - model_type (str): Type of model ('gtsrb', 'ptsd', or 'meta').
    - device (str): Device to use ('cpu' or 'cuda').
    
    Returns:
    - predicted_class (int): The predicted class index (local class for MetaMoE).
    - expert_index (int or None): For MetaMoE, the selected expert index; otherwise None.
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load the model
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

    # Debug: Log model type
    logging.info(f"Loaded model type: {type(model)}")
    
    # Unwrap ModelWrapper if present
    if isinstance(model, ModelWrapper):
        logging.info("Unwrapping ModelWrapper")
        model = model.model
    
    # Additional debug: Check if model is MetaMoE
    logging.info(f"Is MetaMoE: {isinstance(model, MetaMoE)}")
    if isinstance(model, MetaMoE):
        logging.info(f"MetaMoE attributes - num_experts: {model.num_experts}, class_offsets: {model.class_offsets}")
    
    model.eval()

    # Determine normalization based on model type
    if model_type == 'gtsrb':
        mean, std = GTSRB_NORM['mean'], GTSRB_NORM['std']
        num_classes = 43
    elif model_type == 'ptsd':
        mean, std = PTSD_NORM['mean'], PTSD_NORM['std']
        num_classes = 43
    elif model_type == 'meta':
        mean, std = UNIFIED_NORM['mean'], UNIFIED_NORM['std']
        num_classes = 86
    else:
        raise ValueError("Invalid model_type. Choose 'gtsrb', 'ptsd', or 'meta'.")
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Load and preprocess image
    if not os.path.exists(image_path):
        logging.error(f"Image AZImage not found at {image_path}")
        raise FileNotFoundError(f"Image not found at {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logging.error(f"Failed to load image from {image_path}: {e}")
        raise
    input_tensor = transform(image).unsqueeze(0).to(device)
    logging.info(f"Image loaded and preprocessed: {image_path}")

    # Run inference
    with torch.no_grad():
        if isinstance(model, MetaMoE):
            output, gates = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
            expert_index = gates.argmax(dim=1).item()
            local_class = predicted_class - model.class_offsets[expert_index]
            logging.info(f"MetaMoE: Global Class {predicted_class}, Expert {expert_index}, Local Class {local_class}")
            return local_class, expert_index
        else:
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            predicted_class = output.argmax(dim=1).item()
            return predicted_class, None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Evaluate a trained .pth model on an image.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth model file.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--model_type', type=str, default='gtsrb', choices=['gtsrb', 'ptsd', 'meta'], help='Type of model.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference.')
    
    args = parser.parse_args()
    
    predicted_class, expert_index = evaluate_model(args.model_path, args.image_path, args.model_type, args.device)
    
    if expert_index is not None:
        print(f"Predicted Class: {predicted_class} (Expert: {expert_index})")
    else:
        print(f"Predicted Class: {predicted_class}")