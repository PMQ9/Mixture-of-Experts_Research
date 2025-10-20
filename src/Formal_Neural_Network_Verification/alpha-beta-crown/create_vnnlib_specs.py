"""
Create VNNLIB specification files for verifying expert models.

VNNLIB format specifies:
1. Input bounds: epsilon-ball around test image
2. Output property: true label has highest score

Usage:
    python create_vnnlib_specs.py --dataset GTSRB --num_images 10 --epsilon 0.00784
"""

import torch
import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.Vision_Transformer_Pytorch.config import GTSRB_NORM, CIFAR10_NORM, MNIST_NORM

def create_vnnlib_spec(
    image,
    true_label,
    epsilon,
    num_classes,
    output_file,
    mean=None,
    std=None
):
    """
    Create VNNLIB specification file for robustness verification.

    Args:
        image: Input image tensor [C, H, W]
        true_label: Ground truth label (int)
        epsilon: Perturbation bound (in normalized space)
        num_classes: Number of output classes
        output_file: Path to save VNNLIB file
        mean: Normalization mean (optional, for denormalization)
        std: Normalization std (optional, for denormalization)
    """
    # Flatten image
    image_flat = image.flatten().cpu().numpy()
    num_inputs = len(image_flat)

    # Compute input bounds (epsilon-ball in normalized space)
    # We clip to [0, 1] range since images are normalized
    input_lower = np.clip(image_flat - epsilon, -10.0, 10.0)  # Allow wide range for normalized values
    input_upper = np.clip(image_flat + epsilon, -10.0, 10.0)

    with open(output_file, 'w') as f:
        # Declare input variables
        for i in range(num_inputs):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Input constraints (epsilon-ball)
        # Each constraint is a separate assert statement
        f.write("\n; Input constraints\n")
        for i in range(num_inputs):
            f.write(f"(assert (>= X_{i} {input_lower[i]:.10f}))\n")
            f.write(f"(assert (<= X_{i} {input_upper[i]:.10f}))\n")

        # Output property: Y_true > Y_i for all i != true
        # Negation for VNNLIB: we want to prove NO counterexample exists where Y_i >= Y_true
        # DNF format: (assert (or (and (>= Y_0 Y_true)) (and (>= Y_1 Y_true)) ...))
        # Each disjunct must be wrapped in (and ...) even if it's a single comparison
        f.write("\n; Output property: true class has highest score\n")
        f.write("(assert\n")
        f.write("    (or\n")
        for i in range(num_classes):
            if i != true_label:
                f.write(f"        (and (>= Y_{i} Y_{true_label}))\n")
        f.write("    )\n")
        f.write(")\n")


def load_dataset(dataset_name, data_dir='data', num_images=100):
    """
    Load dataset and return images and labels.

    Returns:
        images: List of tensors [C, H, W]
        labels: List of integers
    """
    if dataset_name == 'GTSRB':
        mean, std = GTSRB_NORM['mean'], GTSRB_NORM['std']
        num_classes = 43

        # Load GTSRB test dataset
        import torchvision.transforms as transforms
        from PIL import Image
        import pandas as pd

        test_dir = Path(data_dir) / 'GTSRB' / 'Test'
        csv_path = test_dir / 'testset_with_meta_class.csv'

        if not csv_path.exists():
            csv_path = test_dir / 'GT-final_test.csv'  # Fallback to original CSV

        df = pd.read_csv(csv_path, sep=';' if ';' in open(csv_path).read(100) else ',')

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        images = []
        labels = []

        for idx in range(min(num_images, len(df))):
            row = df.iloc[idx]
            img_path = test_dir / 'Images' / row['Filename'] if 'Images' not in row['Filename'] else test_dir / row['Filename']

            image = Image.open(img_path).convert('RGB')
            image = transform(image)

            images.append(image)
            labels.append(int(row['ClassId']))

    elif dataset_name == 'CIFAR10':
        import torchvision
        import torchvision.transforms as transforms

        mean, std = CIFAR10_NORM['mean'], CIFAR10_NORM['std']
        num_classes = 10

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )

        images = []
        labels = []

        for idx in range(min(num_images, len(testset))):
            image, label = testset[idx]
            images.append(image)
            labels.append(label)

    elif dataset_name == 'MNIST':
        import torchvision
        import torchvision.transforms as transforms
        from PIL import Image

        mean, std = MNIST_NORM['mean'], MNIST_NORM['std']
        num_classes = 10

        # IMPORTANT: Match the training pipeline
        # - MNIST models are trained on RGB images (grayscale converted to 3 channels)
        # - Images are resized from 28x28 to 32x32
        # - Normalization uses 3-channel mean/std
        transform = transforms.Compose([
            transforms.Resize(32),  # Resize from 28x28 to 32x32 (to match training)
            transforms.Lambda(lambda img: img.convert('RGB')),  # Convert grayscale to RGB (3 channels)
            transforms.ToTensor(),
            transforms.Normalize(mean, std)  # Use 3-channel normalization
        ])

        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

        images = []
        labels = []

        for idx in range(min(num_images, len(testset))):
            image, label = testset[idx]
            images.append(image)
            labels.append(label)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return images, labels, num_classes


def main():
    parser = argparse.ArgumentParser(description='Create VNNLIB specification files')

    parser.add_argument('--dataset', type=str, required=True, choices=['GTSRB', 'CIFAR10', 'MNIST'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to create specs for')
    parser.add_argument('--epsilon', type=float, default=0.00784313725490196,
                        help='Perturbation bound (default: 2/255)')
    parser.add_argument('--output_dir', type=str, default='artifacts/vnnlib_specs',
                        help='Directory to save VNNLIB files')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Creating VNNLIB Specification Files")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of images: {args.num_images}")
    print(f"Epsilon: {args.epsilon} ({args.epsilon * 255:.2f}/255)")
    print(f"{'='*80}\n")

    # Load dataset
    print("Loading dataset...")
    images, labels, num_classes = load_dataset(args.dataset, args.data_dir, args.num_images)
    print(f"Loaded {len(images)} images with {num_classes} classes")

    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create VNNLIB files
    print(f"\nCreating VNNLIB files in {output_dir}...")
    for idx, (image, label) in enumerate(zip(images, labels)):
        vnnlib_file = output_dir / f"image_{idx}_label_{label}.vnnlib"
        create_vnnlib_spec(
            image,
            label,
            args.epsilon,
            num_classes,
            vnnlib_file
        )

        if (idx + 1) % 10 == 0:
            print(f"  Created {idx + 1}/{len(images)} VNNLIB files...")

    print(f"\n{'='*80}")
    print(f"[SUCCESS] Created {len(images)} VNNLIB files")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Create CSV index file for alpha-beta-CROWN
    # The CSV should contain relative paths from where abcrown.py is run (complete_verifier directory)
    # Calculate relative path from complete_verifier to vnnlib files
    abcrown_dir = Path('modules/alpha-beta-CROWN/complete_verifier')
    csv_file = output_dir / f"{args.dataset.lower()}_instances.csv"

    with open(csv_file, 'w') as f:
        for idx in range(len(images)):
            vnnlib_filename = f"image_{idx}_label_{labels[idx]}.vnnlib"
            vnnlib_absolute = (output_dir / vnnlib_filename).absolute()

            # Create relative path from abcrown_dir to vnnlib file
            try:
                vnnlib_relative = os.path.relpath(vnnlib_absolute, abcrown_dir)
            except ValueError:
                # On Windows, relpath fails if paths are on different drives
                # Fall back to absolute path
                vnnlib_relative = str(vnnlib_absolute)

            f.write(f"{vnnlib_relative}\n")

    print(f"Created CSV index file: {csv_file}")
    print("\nTo run verification with these specs:")
    print(f"  1. Set 'general.csv_name: {csv_file.absolute()}' in config")
    print(f"  2. Paths in CSV are relative to modules/alpha-beta-CROWN/complete_verifier")


if __name__ == '__main__':
    main()
