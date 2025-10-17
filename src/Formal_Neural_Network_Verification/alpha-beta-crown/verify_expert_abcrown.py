"""
Verify Expert Models with alpha-beta-CROWN

This script provides a high-level interface for verifying expert models using alpha-beta-CROWN.
It handles data loading, ONNX export, and verification execution.

Usage:
    # Verify GTSRB expert
    python verify_expert_abcrown.py --model_path artifacts/gtsrb_small_cnn_best.pth --dataset GTSRB --epsilon 0.00784

    # Verify CIFAR-10 expert
    python verify_expert_abcrown.py --model_path artifacts/cifar10_tiny_cnn_best.pth --dataset CIFAR10 --epsilon 0.00784

    # Custom verification settings
    python verify_expert_abcrown.py --model_path artifacts/gtsrb_small_cnn_best.pth --dataset GTSRB --epsilon 0.00784 --num_images 50 --timeout 600
"""

import torch
import argparse
import sys
import os
from pathlib import Path
import yaml
import subprocess
import numpy as np

# Add src directory to path (go up 3 levels: alpha-beta-crown -> Formal_Neural_Network_Verification -> src)
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import export utility (from parent directory)
sys.path.insert(0, str(Path(__file__).parent.parent))
from export_to_abcrown import export_model_to_onnx


def create_vnnlib_spec(num_classes, true_label, epsilon, input_bounds_lower, input_bounds_upper, output_file):
    """
    Create VNNLIB specification file for robustness verification.

    VNNLIB format:
    - Declare input and output variables
    - Specify input bounds (epsilon-ball around test image)
    - Specify property: output[true_label] > output[i] for all i != true_label

    Args:
        num_classes: Number of output classes
        true_label: Ground truth label
        epsilon: Perturbation bound
        input_bounds_lower: Lower bounds for input (shape: [C, H, W])
        input_bounds_upper: Upper bounds for input (shape: [C, H, W])
        output_file: Path to save VNNLIB file
    """
    with open(output_file, 'w') as f:
        # Flatten input bounds
        input_lower_flat = input_bounds_lower.flatten()
        input_upper_flat = input_bounds_upper.flatten()
        num_inputs = len(input_lower_flat)

        # Declare input variables
        for i in range(num_inputs):
            f.write(f"(declare-const X_{i} Real)\n")

        # Declare output variables
        for i in range(num_classes):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Input constraints (epsilon-ball)
        f.write("\n; Input constraints\n")
        f.write("(assert (and\n")
        for i in range(num_inputs):
            f.write(f"  (>= X_{i} {input_lower_flat[i]:.10f})\n")
            f.write(f"  (<= X_{i} {input_upper_flat[i]:.10f})\n")
        f.write("))\n")

        # Output property: Y_true > Y_i for all i != true
        f.write("\n; Output property: true class has highest score\n")
        f.write("(assert (or\n")
        for i in range(num_classes):
            if i != true_label:
                f.write(f"  (>= Y_{i} Y_{true_label})\n")
        f.write("))\n")


def get_normalization_values(dataset_name):
    """
    Get normalization values for a dataset.

    Args:
        dataset_name: 'GTSRB', 'CIFAR10', or 'MNIST'

    Returns:
        (mean, std): Normalization values for the dataset
    """
    if dataset_name == 'GTSRB':
        from src.Vision_Transformer_Pytorch.config import GTSRB_NORM
        mean, std = list(GTSRB_NORM['mean']), list(GTSRB_NORM['std'])

    elif dataset_name == 'CIFAR10':
        from src.Vision_Transformer_Pytorch.config import CIFAR10_NORM
        mean, std = list(CIFAR10_NORM['mean']), list(CIFAR10_NORM['std'])

    elif dataset_name == 'MNIST':
        from src.Vision_Transformer_Pytorch.config import MNIST_NORM
        mean, std = list(MNIST_NORM['mean']), list(MNIST_NORM['std'])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return mean, std


def run_abcrown_verification(config_path):
    """
    Run alpha-beta-CROWN verification with given config.

    Args:
        config_path: Path to YAML config file

    Returns:
        Verification results (stdout from abcrown.py)
    """
    # Go up 4 levels to reach project root: alpha-beta-crown -> Formal_Neural_Network_Verification -> src -> root
    project_root = Path(__file__).parent.parent.parent.parent
    abcrown_dir = project_root / 'modules' / 'alpha-beta-CROWN' / 'complete_verifier'
    auto_lirpa_dir = project_root / 'modules' / 'alpha-beta-CROWN' / 'auto_LiRPA'

    print(f"\n{'='*80}")
    print("Running alpha-beta-CROWN verification...")
    print(f"Config: {config_path}")
    print(f"{'='*80}\n")

    # Set up environment with auto_LiRPA in PYTHONPATH
    env = os.environ.copy()
    pythonpath = str(auto_lirpa_dir)
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath

    print(f"Added to PYTHONPATH: {auto_lirpa_dir}")

    # Run abcrown.py (use absolute path for config)
    config_abs_path = Path(config_path).absolute()
    result = subprocess.run(
        [sys.executable, 'abcrown.py', '--config', str(config_abs_path)],
        cwd=str(abcrown_dir),
        env=env,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result


def create_verification_config(
    model_path,
    onnx_path,
    dataset_name,
    epsilon,
    num_images,
    timeout,
    output_path,
    mean,
    std,
    vnnlib_dir=None,
    csv_file=None
):
    """
    Create verification configuration YAML file.

    Args:
        model_path: Path to original PyTorch model
        onnx_path: Path to exported ONNX model
        dataset_name: Dataset name
        epsilon: Perturbation bound
        num_images: Number of images to verify
        timeout: Timeout per image (seconds)
        output_path: Path to save config
        mean: Normalization mean
        std: Normalization std
        vnnlib_dir: Directory containing VNNLIB files (optional)
        csv_file: CSV file listing VNNLIB files (optional)
    """
    # Determine input shape based on dataset
    if dataset_name == 'MNIST':
        input_shape = [1, 1, 28, 28]  # Grayscale, 28x28
    else:  # CIFAR10, GTSRB
        input_shape = [1, 3, 32, 32]  # RGB, 32x32

    # Use VNNLIB specs if provided, otherwise use built-in dataset loader
    if vnnlib_dir and csv_file:
        # VNNLIB-based verification (recommended for custom datasets)
        config = {
            'general': {
                'csv_name': str(Path(csv_file).absolute())
            },
            'model': {
                'onnx_path': str(Path(onnx_path).absolute()),
                'input_shape': input_shape
            },
            'specification': {
                'vnnlib_path_prefix': str(Path(vnnlib_dir).absolute()),
                'norm': float('inf'),
                'epsilon': epsilon
            },
            'data': {
                'start': 0,
                'end': num_images
            },
        }
    else:
        # Built-in dataset loader (fallback, may not work for all datasets)
        dataset_loader = 'CIFAR' if dataset_name in ['GTSRB', 'CIFAR10'] else 'MNIST'
        config = {
            'model': {
                'onnx_path': str(Path(onnx_path).absolute())
            },
            'data': {
                'dataset': dataset_loader,
                'mean': mean,
                'std': std,
                'start': 0,
                'end': num_images
            },
            'specification': {
                'norm': float('inf'),
                'epsilon': epsilon
            },
        }

    # Common solver and attack settings
    config.update({
        'attack': {
            'pgd_order': 'skip',  # Skip PGD attack (onnx2pytorch has reshape bug)
            'pgd_steps': 0,
            'pgd_restarts': 0,
            'pgd_alpha': 0.001
        },
        'solver': {
            'batch_size': 1024,
            'alpha-crown': {
                'iteration': 100,
                'lr_alpha': 0.1
            },
            'beta-crown': {
                'lr_alpha': 0.01,
                'lr_beta': 0.05,
                'iteration': 20
            }
        },
        'bab': {
            'timeout': timeout,
            'branching': {
                'method': 'kfsb',
                'reduceop': 'min',
                'candidates': 3
            },
            'cut': {
                'enabled': False
            }
        }
    })

    # Save config using custom YAML dumper that preserves .inf format
    with open(output_path, 'w') as f:
        # Create custom representer for infinity
        def represent_float(dumper, data):
            if data != data:  # NaN
                return dumper.represent_scalar('tag:yaml.org,2002:float', '.nan')
            elif data == float('inf'):
                return dumper.represent_scalar('tag:yaml.org,2002:float', '.inf')
            elif data == float('-inf'):
                return dumper.represent_scalar('tag:yaml.org,2002:float', '-.inf')
            else:
                return dumper.represent_float(data)

        yaml.add_representer(float, represent_float)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created verification config: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Verify expert models with alpha-beta-CROWN')

    # Model and data
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained PyTorch model (.pth)')
    parser.add_argument('--dataset', type=str, required=True, choices=['GTSRB', 'CIFAR10', 'MNIST'],
                        help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to dataset directory')

    # Verification settings
    parser.add_argument('--epsilon', type=float, default=0.00784313725490196,
                        help='Perturbation bound (default: 2/255 = 0.00784)')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to verify')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per image in seconds (default: 300)')

    # Export settings
    parser.add_argument('--onnx_dir', type=str, default='artifacts/abcrown_models',
                        help='Directory to save ONNX models')
    parser.add_argument('--config_dir', type=str, default='artifacts/abcrown_configs',
                        help='Directory to save verification configs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for model export (cuda or cpu)')

    # Skip export if ONNX already exists
    parser.add_argument('--skip_export', action='store_true',
                        help='Skip ONNX export if file already exists')

    # Auto-run verification without prompt
    parser.add_argument('--auto_run', action='store_true',
                        help='Run verification automatically without prompting')

    # Force recreation of VNNLIB specs
    parser.add_argument('--force_recreate', action='store_true',
                        help='Force recreation of VNNLIB specs even if they exist')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("alpha-beta-CROWN Expert Verification")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Epsilon: {args.epsilon} ({args.epsilon * 255:.2f}/255)")
    print(f"Number of images: {args.num_images}")
    print(f"Timeout: {args.timeout}s per image")
    print(f"{'='*80}\n")

    # Step 1: Export to ONNX
    model_name = Path(args.model_path).stem
    onnx_path = Path(args.onnx_dir) / f"{model_name}.onnx"

    if args.skip_export and onnx_path.exists():
        print(f"Skipping ONNX export, using existing file: {onnx_path}")
    else:
        onnx_path = export_model_to_onnx(
            args.model_path,
            args.onnx_dir,
            device=args.device,
            simplify_onnx=True,
            validate=True
        )

    # Step 2: Get normalization values for the dataset
    mean, std = get_normalization_values(args.dataset)

    print(f"\nDataset normalization:")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")

    # Step 2.5: Create VNNLIB specification files
    print(f"\nCreating VNNLIB specification files...")
    vnnlib_dir = Path('artifacts/vnnlib_specs') / args.dataset.lower()
    csv_file = vnnlib_dir / f"{args.dataset.lower()}_instances.csv"

    if args.force_recreate or not vnnlib_dir.exists() or not csv_file.exists():
        # Create VNNLIB specs using the create_vnnlib_specs script
        result = subprocess.run(
            [
                sys.executable,
                'src/Formal_Neural_Network_Verification/alpha-beta-crown/create_vnnlib_specs.py',
                '--dataset', args.dataset,
                '--num_images', str(args.num_images),
                '--epsilon', str(args.epsilon),
                '--data_dir', args.data_dir
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Failed to create VNNLIB specs:")
            print(result.stderr)
            print("Falling back to built-in dataset loader (may not work for all datasets)")
            vnnlib_dir = None
            csv_file = None
        else:
            print(result.stdout)
    else:
        print(f"Using existing VNNLIB specs: {vnnlib_dir}")

    # Step 3: Create verification config
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"{model_name}_verification.yaml"
    create_verification_config(
        args.model_path,
        onnx_path,
        args.dataset,
        args.epsilon,
        args.num_images,
        args.timeout,
        config_path,
        mean,
        std,
        vnnlib_dir=vnnlib_dir,
        csv_file=csv_file
    )

    # Step 4: Run verification
    if vnnlib_dir and csv_file:
        print(f"\nUsing VNNLIB-based verification with {args.num_images} specifications")
    else:
        print("\nNOTE: Using built-in dataset loader (may not work for custom datasets)")
        print("Consider creating VNNLIB specification files for better compatibility.\n")

    print("To run verification manually:")
    print(f"  cd modules/alpha-beta-CROWN/complete_verifier")
    print(f"  python abcrown.py --config {config_path.absolute()}")

    # Optionally run verification automatically
    if args.auto_run:
        run_verification = 'y'
    else:
        run_verification = input("\nRun verification now? (y/n): ").strip().lower()

    if run_verification == 'y':
        result = run_abcrown_verification(config_path)
        print(f"\nVerification exit code: {result.returncode}")
        if result.returncode == 0:
            print("Verification completed successfully!")
        else:
            print("Verification failed or encountered errors.")


if __name__ == '__main__':
    main()
