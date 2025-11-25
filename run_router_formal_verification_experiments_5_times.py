#!/usr/bin/env python3
"""
Run alpha-beta-CROWN verification on both NRT and RT MetaMoE routers (5 runs each).

This script runs the verification pipeline twice:
1. NRT (Non-Robust Training) model: 5 runs
2. RT (Robust Training) model: 5 runs

Results are collected separately and formatted as LaTeX tables for the paper.
Total: 10 runs (5 per model)

Usage (default: 10 MNIST + 10 CIFAR10, 5 runs per model):
    python run_router_formal_verification_experiments_5_times.py

With custom sample counts:
    python run_router_formal_verification_experiments_5_times.py --num_mnist 20 --num_cifar 20

With custom number of runs per model:
    python run_router_formal_verification_experiments_5_times.py --num_runs 3
"""

import subprocess
import sys
from pathlib import Path
import json
import numpy as np

def run_verification_for_model(model_path, model_name, num_mnist, num_cifar, num_runs=5):
    """
    Run the verification script for a specific model.
    Returns aggregated results.
    """
    project_root = Path(__file__).parent.resolve()
    script = project_root / "run_router_formal_verification.py"

    print("\n" + "="*80)
    print(f"RUNNING VERIFICATION FOR {model_name} MODEL ({num_runs} runs)")
    print("="*80)
    print(f"Model: {model_path}")

    # Build command
    cmd = [
        sys.executable,
        str(script),
        "--model_path", str(model_path),
        "--num_mnist", str(num_mnist),
        "--num_cifar", str(num_cifar),
        "--num_runs", str(num_runs)
    ]

    # Run verification
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: Verification failed for {model_name}")
        return None

    # Read the JSON results file
    results_file = project_root / "artifacts/router_verification_results_5runs.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Save a copy with model-specific name
        model_results_file = project_root / f"artifacts/router_verification_{model_name.lower()}_results.json"
        with open(model_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {model_results_file}")

        return results
    else:
        print(f"\nWARNING: Results file not found: {results_file}")
        return None


def main():
    project_root = Path(__file__).parent.resolve()

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Run verification on both NRT and RT MetaMoE routers'
    )
    parser.add_argument('--num_mnist', type=int, default=10,
                        help='Number of MNIST samples (default: 10)')
    parser.add_argument('--num_cifar', type=int, default=10,
                        help='Number of CIFAR10 samples (default: 10)')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs per model (default: 5)')

    args = parser.parse_args()

    # Define models
    nrt_model = project_root / "paper/artifacts/MoE_CNN_NAT/meta_moe_ultra_verifiable_cnn_best_NRT.pth"
    rt_model = project_root / "paper/artifacts/MoE_CNN_AT/meta_moe_ultra_verifiable_cnn_best_RT_eps0.03137.pth"

    # Verify models exist
    if not nrt_model.exists():
        print(f"ERROR: NRT model not found: {nrt_model}")
        sys.exit(1)
    if not rt_model.exists():
        print(f"ERROR: RT model not found: {rt_model}")
        sys.exit(1)

    print("\n" + "="*80)
    print("METAMOE ROUTER VERIFICATION - BOTH MODELS")
    print("="*80)
    print(f"\nNRT Model: {nrt_model.name}")
    print(f"RT Model: {rt_model.name}")
    print(f"MNIST samples: {args.num_mnist}")
    print(f"CIFAR10 samples: {args.num_cifar}")
    print(f"Runs per model: {args.num_runs}")
    print(f"Total runs: {args.num_runs * 2}")

    # Run verification for NRT
    nrt_results = run_verification_for_model(
        str(nrt_model), "NRT", args.num_mnist, args.num_cifar, args.num_runs
    )

    if nrt_results is None:
        print("\nERROR: NRT verification failed")
        sys.exit(1)

    # Run verification for RT
    rt_results = run_verification_for_model(
        str(rt_model), "RT", args.num_mnist, args.num_cifar, args.num_runs
    )

    if rt_results is None:
        print("\nERROR: RT verification failed")
        sys.exit(1)

    print("\n" + "="*80)
    print("ALL VERIFICATION COMPLETE")
    print("="*80)
    print("\nResults Summary:")
    print(f"  NRT runs: {args.num_runs}")
    print(f"  RT runs: {args.num_runs}")
    print(f"  Total: {args.num_runs * 2} runs")

    # Generate LaTeX table content
    nrt_table = _format_latex_table(nrt_results, "NRT")
    rt_table = _format_latex_table(rt_results, "RT")

    # Save to single text file
    output_file = project_root / "artifacts/router_verification_results.txt"
    with open(output_file, 'w') as f:
        f.write("MetaMoE Router Verification - 5 Runs Per Model\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"NRT Model: {Path(str(nrt_model)).name}\n")
        f.write(f"RT Model: {Path(str(rt_model)).name}\n")
        f.write(f"MNIST samples per run: {args.num_mnist}\n")
        f.write(f"CIFAR10 samples per run: {args.num_cifar}\n")
        f.write(f"Runs per model: {args.num_runs}\n")
        f.write(f"Total runs: {args.num_runs * 2}\n\n")

        f.write("=" * 80 + "\n")
        f.write("NRT TABLE (Non-Robust Training Router)\n")
        f.write("=" * 80 + "\n\n")
        for line in nrt_table:
            f.write(line + "\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("RT TABLE (Robust Training Router)\n")
        f.write("=" * 80 + "\n\n")
        for line in rt_table:
            f.write(line + "\n")

    print(f"\nResults saved to: {output_file}")

    # Print formatted tables
    print("\n" + "="*80)
    print("LATEX TABLES FOR PAPER")
    print("="*80)

    print("\n\nNRT TABLE (Non-Robust Training):")
    print("-" * 80)
    for line in nrt_table:
        print(line)

    print("\n\nRT TABLE (Robust Training):")
    print("-" * 80)
    for line in rt_table:
        print(line)

    print("\n" + "="*80)
    print(f"Copy the above tables directly into paper.tex")
    print(f"All results saved to: {output_file}")
    print("="*80)


def _format_latex_table(results, training_type):
    """Format LaTeX table rows from results dict."""
    table_rows = []
    table_rows.append("% " + "="*76)
    table_rows.append(f"% Router Verification Results - {training_type}")
    table_rows.append("% " + "="*76)

    for epsilon in sorted(results.keys()):
        epsilon_frac = epsilon.replace('_', '/')
        first_dataset = True

        for dataset in ["MNIST", "CIFAR10"]:
            if dataset not in results[epsilon]:
                continue

            data = results[epsilon][dataset]
            verified = data['verified']
            falsified = data['falsified']
            unknown = data['unknown']
            success_rate = data['success_rate']
            avg_time = data['avg_time']

            if first_dataset:
                epsilon_col = f"{epsilon_frac}\n({float(epsilon.replace('_', '/')):.5f})"
                first_dataset = False
            else:
                epsilon_col = ""

            # Format row
            row = f"{epsilon_col} & {dataset} & "
            row += f"{verified['mean']:.0f} ± {verified['std']:.2f} & "
            row += f"{falsified['mean']:.0f} ± {falsified['std']:.2f} & "
            row += f"{unknown['mean']:.0f} ± {unknown['std']:.2f} & "
            row += f"{success_rate['mean']:.1f} ± {success_rate['std']:.2f} & "
            row += f"{avg_time['mean']:.2f} ± {avg_time['std']:.2f} \\\\"

            table_rows.append(row)

    return table_rows


if __name__ == '__main__':
    main()
