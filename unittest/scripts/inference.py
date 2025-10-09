"""
Inference script for Go test suite.
Loads a model and runs inference on an image, outputting JSON results.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_model(model_path, device='cpu'):
    """Load a PyTorch model from checkpoint."""
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)


def load_image(image_path, img_size=224):
    """Load and preprocess an image."""
    try:
        img = Image.open(image_path).convert('RGB')

        # Standard normalization (ImageNet defaults)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img).unsqueeze(0)
        return img_tensor
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)


def run_inference(model, image_tensor, model_type='single', device='cpu'):
    """Run model inference and collect outputs."""
    image_tensor = image_tensor.to(device)

    start_time = time.time()

    with torch.no_grad():
        if model_type == 'meta':
            # MetaMoE model - returns (logits, router_weights)
            output = model(image_tensor)
            if isinstance(output, tuple):
                logits, router_weights = output
                router_info = {
                    f"expert_{i}": float(router_weights[0, i])
                    for i in range(router_weights.size(1))
                }
            else:
                logits = output
                router_info = {}
        else:
            # Single expert model
            logits = model(image_tensor)
            router_info = {}

    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    # Get predictions
    probs = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probs, 1)

    result = {
        "predictions": logits[0].cpu().numpy().tolist(),
        "class": int(predicted_class.item()),
        "confidence": float(confidence.item()),
        "router_info": router_info,
        "inference_ms": inference_time
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Run model inference for testing')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model_type', type=str, default='single',
                       choices=['single', 'meta'],
                       help='Type of model (single expert or meta)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size for preprocessing')
    parser.add_argument('--output_json', type=bool, default=True,
                       help='Output results as JSON')

    args = parser.parse_args()

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # Load model and image
    model = load_model(args.model_path, device=device)
    image_tensor = load_image(args.image_path, img_size=args.img_size)

    # Run inference
    result = run_inference(model, image_tensor,
                          model_type=args.model_type,
                          device=device)

    # Output results
    if args.output_json:
        print(json.dumps(result))
    else:
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {result['inference_ms']:.2f} ms")
        if result['router_info']:
            print("Router weights:")
            for expert, weight in result['router_info'].items():
                print(f"  {expert}: {weight:.4f}")


if __name__ == '__main__':
    main()
