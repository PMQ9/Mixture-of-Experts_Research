#!/usr/bin/env python3
"""
Run alpha-beta-CROWN verification on both NRT and RT MetaMoE routers (5 runs each).

This script runs the verification pipeline twice:
1. NRT (Non-Robust Training) model: 5 runs
2. RT (Robust Training) model: 5 runs

Results are collected separately and formatted as LaTeX tables for the paper.
Total: 20 runs (5 per model)

Usage (default: 20 MNIST + 20 CIFAR10, 5 runs per model):
    python run_router_formal_verification_experiments_5_times.py

With custom sample counts:
    python run_router_formal_verification_experiments_5_times.py --num_mnist 20 --num_cifar 20

With custom number of runs per model:
    python run_router_formal_verification_experiments_5_times.py --num_runs 3
"""

import subprocess
import sys
from pathlib import Path

def run_verification_for_model(model_path, model_name, num_mnist, num_cifar, num_runs=5):
    """
    Run the verification script multiple times for a specific model.
    Returns aggregated results.
    """
    project_root = Path(__file__).parent.resolve()
    script = project_root / "run_router_formal_verification.py"

    print("\n" + "="*80)
    print(f"RUNNING VERIFICATION FOR {model_name} MODEL ({num_runs} runs)")
    print("="*80)
    print(f"Model: {model_path}")

    all_results = []

    # Run verification num_runs times
    for run_num in range(1, num_runs + 1):
        print(f"\n--- Run {run_num}/{num_runs} ---")

        # Build command (no --num_runs argument for original script)
        cmd = [
            sys.executable,
            str(script),
            "--model_path", str(model_path),
            "--num_mnist", str(num_mnist),
            "--num_cifar", str(num_cifar)
        ]

        # Run verification
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"\nWARNING: Run {run_num} verification failed for {model_name}")
            # Continue with other runs instead of aborting
            continue

        # Read the results file from this run
        results_file = project_root / "artifacts/router_verification_results.txt"
        if results_file.exists():
            with open(results_file, 'r') as f:
                run_results = f.read()
            all_results.append(run_results)
            print(f"Results captured for run {run_num}")
        else:
            print(f"WARNING: Results file not found after run {run_num}")

    if not all_results:
        print(f"\nERROR: No successful runs for {model_name}")
        return None

    # Save combined results
    combined_file = project_root / f"artifacts/router_verification_{model_name.lower()}_all_runs.txt"
    with open(combined_file, 'w') as f:
        f.write(f"Combined Results - {model_name} ({num_runs} runs)\n")
        f.write("="*80 + "\n\n")
        for i, results in enumerate(all_results, 1):
            f.write(f"\n--- RUN {i} ---\n")
            f.write(results)
            f.write("\n" + "="*80 + "\n")

    print(f"\nAll runs results saved to: {combined_file}")
    return all_results


def main():
    project_root = Path(__file__).parent.resolve()

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Run verification on both NRT and RT MetaMoE routers'
    )
    parser.add_argument('--num_mnist', type=int, default=20,
                        help='Number of MNIST samples (default: 20)')
    parser.add_argument('--num_cifar', type=int, default=20,
                        help='Number of CIFAR10 samples (default: 20)')
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

    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(f"\nNRT Results saved to: artifacts/router_verification_nrt_all_runs.txt")
    print(f"RT Results saved to: artifacts/router_verification_rt_all_runs.txt")
    print("\nNote: Aggregate the results manually or parse the text output files to generate LaTeX tables")
    print("="*80)

if __name__ == '__main__':
    main()
