"""
Run Formal Router Verification with alpha-beta-CROWN

This script runs formal verification on the MetaMoE router using VNNLIB specifications.
It provides formal guarantees that adversarial perturbations cannot fool the router
into selecting the wrong expert.

Usage:
    # Verify CIFAR-10 images (200 specs)
    python run_router_formal_verification.py --dataset CIFAR10

    # Verify MNIST images (200 specs)
    python run_router_formal_verification.py --dataset MNIST

    # Verify both datasets
    python run_router_formal_verification.py --dataset BOTH
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_verification(dataset, project_root):
    """Run alpha-beta-CROWN verification for a dataset"""

    if dataset == 'CIFAR10':
        config_file = 'exp_configs/moe_experts/router_vnnlib_cifar10.yaml'
        print(f"\n{'='*80}")
        print(f"Running Formal Router Verification: CIFAR-10")
        print(f"{'='*80}\n")
    elif dataset == 'MNIST':
        config_file = 'exp_configs/moe_experts/router_vnnlib_mnist.yaml'
        print(f"\n{'='*80}")
        print(f"Running Formal Router Verification: MNIST")
        print(f"{'='*80}\n")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Set up environment
    auto_lirpa_path = project_root / 'modules' / 'alpha-beta-CROWN' / 'auto_LiRPA'
    env = os.environ.copy()
    pythonpath = str(auto_lirpa_path)
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath

    # Run abcrown
    abcrown_dir = project_root / 'modules' / 'alpha-beta-CROWN' / 'complete_verifier'

    print(f"Config: {config_file}")
    print(f"Expected expert: {'0 (CIFAR-10)' if dataset == 'CIFAR10' else '1 (MNIST)'}")
    print(f"Verification property: Router must select correct expert within epsilon-ball\n")

    result = subprocess.run(
        [sys.executable, 'abcrown.py', '--config', config_file],
        cwd=str(abcrown_dir),
        env=env
    )

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='Run formal router verification with alpha-beta-CROWN'
    )
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'MNIST', 'BOTH'],
                        help='Dataset to verify (default: CIFAR10)')

    args = parser.parse_args()

    project_root = Path(__file__).parent

    # Check if VNNLIB specs exist
    vnnlib_dirs = {
        'CIFAR10': project_root / 'artifacts' / 'vnnlib' / 'router_cifar10',
        'MNIST': project_root / 'artifacts' / 'vnnlib' / 'router_mnist'
    }

    datasets_to_run = []
    if args.dataset == 'BOTH':
        datasets_to_run = ['CIFAR10', 'MNIST']
    else:
        datasets_to_run = [args.dataset]

    for dataset in datasets_to_run:
        # Check if specs exist
        vnnlib_dir = vnnlib_dirs[dataset]
        csv_file = vnnlib_dir / 'instances.csv'

        if not csv_file.exists():
            print(f"\n[ERROR] VNNLIB specs not found for {dataset}")
            print(f"Expected: {csv_file}")
            print(f"\nPlease run:")
            print(f"  python src/Formal_Neural_Network_Verification/alpha-beta-crown/generate_router_vnnlib.py \\")
            print(f"    --dataset {dataset} --num_images 200\n")
            continue

        # Run verification
        returncode = run_verification(dataset, project_root)

        if returncode != 0:
            print(f"\n[WARNING] Verification for {dataset} exited with code {returncode}\n")

    print(f"\n{'='*80}")
    print("Formal Router Verification Complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
