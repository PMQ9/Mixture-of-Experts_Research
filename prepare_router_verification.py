"""
Prepare Router Verification Data for alpha-beta-CROWN

This script:
1. Loads MNIST and CIFAR10 test images
2. Creates VNNLIB specifications for robustness verification
3. Generates CSV file listing all verification instances
"""

import torch
import numpy as np
from torchvision import datasets, transforms
import sys
import os
from pathlib import Path

sys.path.insert(0, 'src/Vision_Transformer_Pytorch')
from config import UNIFIED_NORM

# Create output directories
vnnlib_dir = Path("artifacts/vnnlib_specs/router")
vnnlib_dir.mkdir(parents=True, exist_ok=True)

# Prepare transforms (UNIFIED normalization - same as training!)
transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(UNIFIED_NORM['mean'], UNIFIED_NORM['std'])
])

transform_cifar = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(UNIFIED_NORM['mean'], UNIFIED_NORM['std'])
])

# Load datasets
print("Loading datasets...")
mnist = datasets.MNIST(root='./data', train=False, download=False, transform=transform_mnist)
cifar10 = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_cifar)

def create_vnnlib_router(input_image, true_expert, epsilon, output_file):
    """
    Create VNNLIB specification for router verification.

    Property: For all perturbations within epsilon-ball around input_image,
              the router should predict true_expert

    Args:
        input_image: torch.Tensor of shape [3, 32, 32]
        true_expert: int (0 for CIFAR10, 1 for MNIST)
        epsilon: float, perturbation bound (L-infinity norm)
        output_file: str, path to save VNNLIB file
    """
    img_np = input_image.numpy()
    C, H, W = img_np.shape

    with open(output_file, 'w') as f:
        # Declare variables
        # Inputs: X_0, X_1, ..., X_3071 (flattened 3x32x32 = 3072 pixels)
        # Outputs: Y_0, Y_1 (logits for expert 0 and expert 1)

        # Input variables (flattened)
        idx = 0
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    f.write(f"(declare-const X_{idx} Real)\n")
                    idx += 1

        # Output variables
        f.write(f"(declare-const Y_0 Real)\n")  # Expert 0 logit
        f.write(f"(declare-const Y_1 Real)\n")  # Expert 1 logit
        f.write("\n")

        # Input constraints (epsilon-ball)
        # Note: Do NOT clamp to [-1, 1] - normalized images can have any range
        # depending on the normalization parameters (mean/std)
        f.write("; Input constraints\n")
        idx = 0
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    pixel_val = float(img_np[c, h, w])
                    lower = pixel_val - epsilon
                    upper = pixel_val + epsilon
                    f.write(f"(assert (<= X_{idx} {upper:.10f}))\n")
                    f.write(f"(assert (>= X_{idx} {lower:.10f}))\n")
                    idx += 1
        f.write("\n")

        # Output property: Router should predict true_expert
        # If true_expert=0: Y_0 > Y_1 (expert 0 logit > expert 1 logit)
        # If true_expert=1: Y_1 > Y_0 (expert 1 logit > expert 0 logit)
        f.write("; Output property\n")
        if true_expert == 0:
            # Negation of property: Y_1 >= Y_0 (we want to find counterexample)
            f.write("(assert (>= Y_1 Y_0))\n")
        else:  # true_expert == 1
            # Negation of property: Y_0 >= Y_1
            f.write("(assert (>= Y_0 Y_1))\n")

print("\nCreating VNNLIB specifications...")
print("="*70)

# Configuration
epsilon = 2.0 / 255.0  # L-infinity perturbation bound (2/255 is common for robustness testing)
num_mnist_samples = 10
num_cifar_samples = 10

csv_lines = []

# Process MNIST samples (should route to expert 1)
print(f"\nMNIST samples (epsilon={epsilon:.6f}):")
for i in range(num_mnist_samples):
    img, label = mnist[i * 100]  # Sample every 100th image
    vnnlib_file = vnnlib_dir / f"mnist_{i}.vnnlib"
    create_vnnlib_router(img, true_expert=1, epsilon=epsilon, output_file=str(vnnlib_file))

    csv_lines.append(f"mnist_{i},artifacts/vnnlib_specs/router/mnist_{i}.vnnlib,60")
    print(f"  Created: {vnnlib_file.name} (digit {label})")

# Process CIFAR10 samples (should route to expert 0)
print(f"\nCIFAR10 samples (epsilon={epsilon:.6f}):")
for i in range(num_cifar_samples):
    img, label = cifar10[i * 100]  # Sample every 100th image
    vnnlib_file = vnnlib_dir / f"cifar10_{i}.vnnlib"
    create_vnnlib_router(img, true_expert=0, epsilon=epsilon, output_file=str(vnnlib_file))

    csv_lines.append(f"cifar10_{i},artifacts/vnnlib_specs/router/cifar10_{i}.vnnlib,60")
    print(f"  Created: {vnnlib_file.name} (class {label})")

# Create instances CSV for alpha-beta-CROWN
csv_file = "artifacts/router_verification_instances.csv"
with open(csv_file, 'w') as f:
    for line in csv_lines:
        f.write(line + "\n")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Created {len(csv_lines)} VNNLIB specifications")
print(f"MNIST samples: {num_mnist_samples}")
print(f"CIFAR10 samples: {num_cifar_samples}")
print(f"Epsilon (L∞): {epsilon:.6f} ({epsilon*255:.1f}/255)")
print(f"\nInstances CSV: {csv_file}")
print(f"VNNLIB specs: {vnnlib_dir}/")
print("\nNext: Create YAML config and run alpha-beta-CROWN verification")
