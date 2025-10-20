"""
Generate VNNLIB Specifications for MetaMoE Router Verification

This script creates VNNLIB (.vnnlib) specification files for formal verification
of the MetaMoE router with alpha-beta-CROWN. Each spec defines:
- Input bounds: epsilon-ball around a test image
- Output property: router must select correct expert (argmax must be correct)

VNNLIB format for router verification:
- For CIFAR-10 images: Verify output[1] > output[0] (select CIFAR expert)
- For MNIST images: Verify output[0] > output[1] (select MNIST expert)

This enables formal verification of routing robustness, proving that adversarial
perturbations cannot fool the router into selecting the wrong expert.

Usage:
    # Generate specs for CIFAR-10 images
    python generate_router_vnnlib.py \
        --dataset CIFAR10 \
        --num_images 200 \
        --epsilon 0.00784 \
        --output_dir artifacts/vnnlib/router_cifar10

    # Generate specs for MNIST images
    python generate_router_vnnlib.py \
        --dataset MNIST \
        --num_images 200 \
        --epsilon 0.00784 \
        --output_dir artifacts/vnnlib/router_mnist
"""

import torch
import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src' / 'Vision_Transformer_Pytorch'))

from src.Vision_Transformer_Pytorch.config import MNIST_NORM, CIFAR10_NORM
from torchvision import datasets, transforms


def create_router_vnnlib_spec(
    image,
    expected_expert,
    epsilon,
    mean,
    std,
    output_file,
    num_experts=2
):
    """
    Create a VNNLIB specification file for router verification.

    Args:
        image: Normalized image tensor (C, H, W)
        expected_expert: Index of correct expert (0 for MNIST, 1 for CIFAR-10)
        epsilon: L-infinity perturbation bound
        mean: Normalization mean (list of 3 floats)
        std: Normalization std (list of 3 floats)
        output_file: Path to save VNNLIB file
        num_experts: Number of experts (default: 2)

    The spec verifies that the router outputs satisfy:
        output[expected_expert] > output[i] for all i != expected_expert
    """

    # Image is already normalized, we need to compute epsilon-ball in normalized space
    # Input bounds: [image - epsilon, image + epsilon] clipped to valid range

    C, H, W = image.shape
    num_inputs = C * H * W

    # Flatten image
    image_flat = image.flatten()

    # Compute input bounds in normalized space
    # The image is already normalized, so we add epsilon directly
    input_lower = (image_flat - epsilon).clamp(-10, 10)  # Reasonable bounds for normalized space
    input_upper = (image_flat + epsilon).clamp(-10, 10)

    # Write VNNLIB file
    with open(output_file, 'w') as f:
        # Declare input variables
        f.write(f"; Router verification for expert {expected_expert}\n")
        f.write(f"; Input image shape: {C}x{H}x{W}\n")
        f.write(f"; Epsilon: {epsilon}\n\n")

        for i in range(num_inputs):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables (one per expert)
        for i in range(num_experts):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Input constraints (epsilon-ball around normalized image)
        f.write("\n; Input constraints (epsilon-ball)\n")
        for i in range(num_inputs):
            f.write(f"(assert (>= X_{i} {input_lower[i]:.10f}))\n")
            f.write(f"(assert (<= X_{i} {input_upper[i]:.10f}))\n")

        # Output property: expected_expert must have highest output
        # Property is NEGATED in VNNLIB (we're looking for counterexamples)
        # So we assert: NOT(output[expected] > output[i] for all i != expected)
        # Which is: OR(output[i] >= output[expected] for some i != expected)
        f.write(f"\n; Output property: expert {expected_expert} must have highest output\n")
        f.write(f"; (Negated: looking for counterexample where wrong expert is selected)\n")

        # Build list of and clauses
        and_clauses = []
        for i in range(num_experts):
            if i != expected_expert:
                and_clauses.append(f"(and (>= Y_{i} Y_{expected_expert}))")

        # Write on single line (required by alpha-beta-CROWN parser)
        f.write(f"(assert (or {' '.join(and_clauses)}))\n")


def cleanup_vnnlib_directory(vnnlib_dir):
    """
    Clean up old VNNLIB files before generating new ones.

    Args:
        vnnlib_dir: Path to VNNLIB directory
    """
    if vnnlib_dir.exists():
        print(f"Cleaning up old VNNLIB files in {vnnlib_dir}...")
        # Remove all .vnnlib files
        vnnlib_files = list(vnnlib_dir.glob("*.vnnlib"))
        if vnnlib_files:
            for f in vnnlib_files:
                f.unlink()
            print(f"  Removed {len(vnnlib_files)} old VNNLIB files")
        else:
            print(f"  No old VNNLIB files to remove")
    else:
        print(f"Creating VNNLIB directory: {vnnlib_dir}")


def generate_router_vnnlib_specs(
    dataset_name,
    num_images,
    epsilon,
    output_dir,
    device='cpu',
    cleanup=True
):
    """
    Generate VNNLIB specs for router verification on a dataset.

    Args:
        dataset_name: 'MNIST' or 'CIFAR10'
        num_images: Number of images to generate specs for
        epsilon: L-infinity perturbation bound
        output_dir: Directory to save VNNLIB files
        device: Device for loading images
        cleanup: Whether to clean up old VNNLIB files before generation (default: True)

    Returns:
        tuple: (vnnlib_dir, csv_path) - paths to specs and index file
    """
    print(f"\n{'='*80}")
    print(f"Generating VNNLIB Specifications for Router Verification")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Number of images: {num_images}")
    print(f"Epsilon: {epsilon:.5f} ({epsilon*255:.1f}/255)")
    print(f"Output directory: {output_dir}\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old VNNLIB files
    if cleanup:
        cleanup_vnnlib_directory(output_dir)

    # Load dataset
    print("Loading dataset...")
    if dataset_name == 'MNIST':
        mean, std = MNIST_NORM['mean'], MNIST_NORM['std']
        # MetaMoE expert order: [cifar10_model, mnist_model]
        # Expert 0 = CIFAR-10, Expert 1 = MNIST
        expected_expert = 1  # MNIST images should route to Expert 1

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    elif dataset_name == 'CIFAR10':
        mean, std = CIFAR10_NORM['mean'], CIFAR10_NORM['std']
        # MetaMoE expert order: [cifar10_model, mnist_model]
        # Expert 0 = CIFAR-10, Expert 1 = MNIST
        expected_expert = 0  # CIFAR-10 images should route to Expert 0

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Limit to num_images
    num_images = min(num_images, len(test_dataset))
    test_dataset = torch.utils.data.Subset(test_dataset, range(num_images))

    print(f"Loaded {len(test_dataset)} images")
    print(f"Expected expert for all images: {expected_expert}")
    print(f"Normalization: mean={mean}, std={std}\n")

    # Generate VNNLIB files
    print("Generating VNNLIB specifications...")

    vnnlib_files = []

    for idx in tqdm(range(len(test_dataset)), desc="Creating specs"):
        image, label = test_dataset[idx]

        # Generate VNNLIB file
        vnnlib_file = output_dir / f"spec_{idx}.vnnlib"

        create_router_vnnlib_spec(
            image=image,
            expected_expert=expected_expert,
            epsilon=epsilon,
            mean=mean,
            std=std,
            output_file=vnnlib_file,
            num_experts=2
        )

        vnnlib_files.append(vnnlib_file)

    # Create CSV index file for alpha-beta-CROWN
    # CSV format: onnx_file,vnnlib_file,timeout
    csv_path = output_dir / 'instances.csv'

    print(f"\nCreating CSV index file: {csv_path}")

    # Paths must be relative to complete_verifier directory where abcrown.py runs
    # complete_verifier is at: modules/alpha-beta-CROWN/complete_verifier/
    # ONNX is at: artifacts/abcrown_models/
    # VNNLIB is at: artifacts/vnnlib/router_{dataset}/

    onnx_relative_path = "../../../artifacts/abcrown_models/meta_moe_ultra_verifiable_cnn_best_og_router_only.onnx"
    vnnlib_prefix = f"../../../artifacts/vnnlib/router_{dataset_name.lower()}/"
    timeout = 10  # 10 seconds per instance (router is small)

    with open(csv_path, 'w') as f:
        for vnnlib_file in vnnlib_files:
            # Make vnnlib path relative to complete_verifier directory
            vnnlib_relative = vnnlib_prefix + vnnlib_file.name
            f.write(f"{onnx_relative_path},{vnnlib_relative},{timeout}\n")

    print(f"\n{'='*80}")
    print(f"VNNLIB Generation Complete!")
    print(f"{'='*80}")
    print(f"Generated {len(vnnlib_files)} specification files")
    print(f"VNNLIB directory: {output_dir}")
    print(f"CSV index file: {csv_path}")
    print(f"\nNext steps:")
    print(f"1. Run alpha-beta-CROWN with:")
    print(f"   cd modules/alpha-beta-CROWN/complete_verifier")
    print(f"   python abcrown.py --config exp_configs/moe_experts/router_vnnlib_{dataset_name.lower()}.yaml")
    print(f"\n2. Or use the automated verification script")
    print(f"{'='*80}\n")

    return str(output_dir), str(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate VNNLIB specs for MetaMoE router verification'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['MNIST', 'CIFAR10'],
                        help='Dataset to generate specs for')
    parser.add_argument('--num_images', type=int, default=200,
                        help='Number of images to generate specs for')
    parser.add_argument('--epsilon', type=float, default=2/255,
                        help='L-infinity perturbation bound (default: 2/255)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: artifacts/vnnlib/router_{dataset})')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for loading images')

    args = parser.parse_args()

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"artifacts/vnnlib/router_{args.dataset.lower()}"

    # Generate VNNLIB specs
    vnnlib_dir, csv_path = generate_router_vnnlib_specs(
        dataset_name=args.dataset,
        num_images=args.num_images,
        epsilon=args.epsilon,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
