"""
Run All Verification Tests and Log Results to VERIFICATION_RESULTS.md

This script automatically runs all expert and router verification tests defined in
VERIFICATION_RESULTS.md and updates the markdown file with the results.

Usage:
    python run_all_verifications.py [--skip-experts] [--skip-routers] [--dry-run]

Options:
    --skip-experts   Skip expert verification (only run router)
    --skip-routers   Skip router verification (only run experts)
    --dry-run        Print commands without executing
"""

import subprocess
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
import json
import os

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Project root (parent of paper directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Model configurations
EXPERT_CONFIGS = [
    {
        'name': 'E_0_CNN_NAT',
        'type': 'Expert',
        'dataset': 'CIFAR10',
        'training': 'Non-Robust',
        'model_path': 'paper/artifacts/E_0_CNN_NAT/cifar10_ultra_verifiable_cnn_best_og.pth',
        'epsilons': [0.00784, 0.01569, 0.03137],
        'num_images': 20,
    },
    {
        'name': 'E_0_CNN_AT',
        'type': 'Expert',
        'dataset': 'CIFAR10',
        'training': 'Robust',
        'model_path': 'paper/artifacts/E_0_CNN_AT/cifar10_ultra_verifiable_cnn_best_robust.pth',
        'epsilons': [0.00784, 0.01569, 0.03137],
        'num_images': 20,
    },
    {
        'name': 'E_1_CNN_NAT',
        'type': 'Expert',
        'dataset': 'MNIST',
        'training': 'Non-Robust',
        'model_path': 'paper/artifacts/E_1_CNN_NAT/mnist_ultra_verifiable_cnn_best_og.pth',
        'epsilons': [0.00784, 0.01569, 0.03137],
        'num_images': 20,
    },
    {
        'name': 'E_1_CNN_AT',
        'type': 'Expert',
        'dataset': 'MNIST',
        'training': 'Robust',
        'model_path': 'paper/artifacts/E_1_CNN_AT/mnist_ultra_verifiable_cnn_best_robust.pth',
        'epsilons': [0.00784, 0.01569, 0.03137],
        'num_images': 20,
    },
]

ROUTER_CONFIGS = [
    {
        'name': 'MoE_CNN_NAT',
        'type': 'Router',
        'training': 'Non-Robust',
        'model_path': 'paper/artifacts/MoE_CNN_NAT/meta_moe_ultra_verifiable_cnn_best_og.pth',
        'epsilons': [0.00784, 0.01569, 0.03137],
        'num_mnist': 50,
        'num_cifar': 50,
    },
    {
        'name': 'MoE_CNN_AT',
        'type': 'Router',
        'training': 'Robust',
        'model_path': 'paper/artifacts/MoE_CNN_AT/meta_moe_ultra_verifiable_cnn_best_og.pth',
        'epsilons': [0.00784, 0.01569, 0.03137],
        'num_mnist': 50,
        'num_cifar': 50,
    },
]


def epsilon_to_fraction(epsilon):
    """Convert epsilon float to fraction string (e.g., 0.00784 -> '2/255')"""
    epsilon_map = {
        0.00784: '2/255',
        0.01569: '4/255',
        0.03137: '8/255',
    }
    # Find closest match
    for val, frac in sorted(epsilon_map.items()):
        if abs(val - epsilon) < 1e-5:
            return frac
    return f'{epsilon:.5f}'


def run_expert_verification(config, epsilon, dry_run=False):
    """Run expert verification and return results"""
    model_path = PROJECT_ROOT / config['model_path']
    dataset = config['dataset']
    num_images = config['num_images']
    timeout = 300

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py'),
        '--model_path', str(model_path),
        '--dataset', dataset,
        '--epsilon', str(epsilon),
        '--num_images', str(num_images),
        '--timeout', str(timeout),
    ]

    print(f"\n{'='*80}")
    print(f"Running: {config['name']} @ eps={epsilon_to_fraction(epsilon)}")
    print(f"{'='*80}")
    print(' '.join(cmd))

    if dry_run:
        print("[DRY RUN - NOT EXECUTING]")
        return None

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 300, cwd=PROJECT_ROOT)

        # Parse results from output
        output = result.stdout + result.stderr
        verified = falsified = timeout_count = unknown = avg_time = 0

        # Parse verification results
        if 'Verified:' in output:
            match = re.search(r'Verified:\s*(\d+)', output)
            if match:
                verified = int(match.group(1))

        if 'Falsified:' in output:
            match = re.search(r'Falsified:\s*(\d+)', output)
            if match:
                falsified = int(match.group(1))

        if 'Timeout:' in output:
            match = re.search(r'Timeout:\s*(\d+)', output)
            if match:
                timeout_count = int(match.group(1))

        if 'Unknown:' in output:
            match = re.search(r'Unknown:\s*(\d+)', output)
            if match:
                unknown = int(match.group(1))

        if 'Average time' in output:
            match = re.search(r'Average time.*:\s*([\d.]+)', output)
            if match:
                avg_time = float(match.group(1))

        return {
            'verified': verified,
            'falsified': falsified,
            'timeout': timeout_count,
            'unknown': unknown,
            'avg_time': avg_time,
            'success': result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Verification timeout (>{timeout + 300}s)")
        return {
            'verified': 0,
            'falsified': 0,
            'timeout': 1,
            'unknown': 0,
            'avg_time': 0,
            'success': False,
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return {
            'verified': 0,
            'falsified': 0,
            'timeout': 0,
            'unknown': num_images,
            'avg_time': 0,
            'success': False,
        }


def run_router_verification(config, epsilon, dry_run=False):
    """Run router verification and return results"""
    model_path = PROJECT_ROOT / config['model_path']
    num_mnist = config['num_mnist']
    num_cifar = config['num_cifar']
    timeout = 300

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'verify_all_router_samples.py'),
        '--model_path', str(model_path),
        '--num_mnist', str(num_mnist),
        '--num_cifar', str(num_cifar),
        '--epsilon', str(epsilon),
        '--timeout', str(timeout),
    ]

    print(f"\n{'='*80}")
    print(f"Running: {config['name']} @ eps={epsilon_to_fraction(epsilon)}")
    print(f"{'='*80}")
    print(' '.join(cmd))

    if dry_run:
        print("[DRY RUN - NOT EXECUTING]")
        return None

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 600, cwd=PROJECT_ROOT)

        # Parse results from output
        output = result.stdout + result.stderr
        verified = falsified = timeout_count = unknown = avg_time = 0

        # Parse verification results
        if 'Verified:' in output:
            match = re.search(r'Verified:\s*(\d+)', output)
            if match:
                verified = int(match.group(1))

        if 'Falsified:' in output:
            match = re.search(r'Falsified:\s*(\d+)', output)
            if match:
                falsified = int(match.group(1))

        if 'Timeout:' in output:
            match = re.search(r'Timeout:\s*(\d+)', output)
            if match:
                timeout_count = int(match.group(1))

        if 'Unknown:' in output:
            match = re.search(r'Unknown:\s*(\d+)', output)
            if match:
                unknown = int(match.group(1))

        if 'Average time' in output:
            match = re.search(r'Average time.*:\s*([\d.]+)', output)
            if match:
                avg_time = float(match.group(1))

        return {
            'verified': verified,
            'falsified': falsified,
            'timeout': timeout_count,
            'unknown': unknown,
            'avg_time': avg_time,
            'success': result.returncode == 0,
        }

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Verification timeout (>{timeout + 600}s)")
        return {
            'verified': 0,
            'falsified': 0,
            'timeout': 1,
            'unknown': 0,
            'avg_time': 0,
            'success': False,
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        return {
            'verified': 0,
            'falsified': 0,
            'timeout': 0,
            'unknown': num_mnist + num_cifar,
            'avg_time': 0,
            'success': False,
        }


def update_markdown_results(all_results):
    """Update VERIFICATION_RESULTS.md with new results"""
    md_path = Path(__file__).parent / 'VERIFICATION_RESULTS.md'

    if not md_path.exists():
        print(f"ERROR: {md_path} not found!")
        return

    with open(md_path, 'r') as f:
        lines = f.readlines()

    # Update results in tables
    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for epsilon values in result tables
        for model_name, results in all_results.items():
            for epsilon, result in results.items():
                if result is None:
                    i += 1
                    continue

                epsilon_frac = epsilon_to_fraction(epsilon)
                if epsilon_frac in line:
                    # Found the line with this epsilon
                    # Parse the line and update values
                    parts = line.strip().split('|')
                    if len(parts) >= 6:  # Should have at least: |epsilon|verified|falsified|timeout|unknown|time|
                        # Update the values
                        if '|' in line:
                            # Reconstruct the line with new values
                            new_line = f"| {epsilon_frac} ({epsilon:.5f}) | {result['verified']} | {result['falsified']} | {result['timeout']} | {result['unknown']} | {result['avg_time']:.2f} | |"
                            lines[i] = new_line + '\n'

        i += 1

    # Update last run timestamp
    for i, line in enumerate(lines):
        if line.startswith('- Last run:'):
            lines[i] = f'- Last run: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
            break

    with open(md_path, 'w') as f:
        f.writelines(lines)

    print(f"\n✓ Updated: {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Run all verification tests')
    parser.add_argument('--skip-experts', action='store_true', help='Skip expert verification')
    parser.add_argument('--skip-routers', action='store_true', help='Skip router verification')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to run (e.g., E_0_CNN_NAT,MoE_CNN_NAT)')

    args = parser.parse_args()

    all_results = {}

    # Filter configurations if models specified
    experts = EXPERT_CONFIGS if not args.skip_experts else []
    routers = ROUTER_CONFIGS if not args.skip_routers else []

    if args.models:
        model_names = set(args.models.split(','))
        experts = [e for e in experts if e['name'] in model_names]
        routers = [r for r in routers if r['name'] in model_names]

    print("\n" + "="*80)
    print("VERIFICATION BATCH RUN")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Experts to verify: {len(experts)}")
    print(f"Routers to verify: {len(routers)}")
    print(f"Dry run: {args.dry_run}")

    # Run expert verifications
    total_expert_tests = len(experts) * 3  # 3 epsilon values per expert
    current_test = 0

    for config in experts:
        all_results[config['name']] = {}
        for epsilon in config['epsilons']:
            current_test += 1
            print(f"\n[{current_test}/{total_expert_tests + len(routers)*3}] Expert test: {config['name']} @ eps={epsilon_to_fraction(epsilon)}")

            result = run_expert_verification(config, epsilon, args.dry_run)
            all_results[config['name']][epsilon] = result

            if result and not result['success']:
                print(f"⚠ Warning: Test may have failed")

    # Run router verifications
    total_router_tests = len(routers) * 3
    for config in routers:
        all_results[config['name']] = {}
        for epsilon in config['epsilons']:
            current_test += 1
            print(f"\n[{current_test}/{total_expert_tests + total_router_tests}] Router test: {config['name']} @ eps={epsilon_to_fraction(epsilon)}")

            result = run_router_verification(config, epsilon, args.dry_run)
            all_results[config['name']][epsilon] = result

            if result and not result['success']:
                print(f"⚠ Warning: Test may have failed")

    # Update markdown
    if not args.dry_run:
        update_markdown_results(all_results)

    print("\n" + "="*80)
    print("VERIFICATION BATCH RUN COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: paper/VERIFICATION_RESULTS.md")


if __name__ == '__main__':
    main()
