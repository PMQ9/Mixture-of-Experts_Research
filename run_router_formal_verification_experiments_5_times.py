#!/usr/bin/env python3
"""
Run alpha-beta-CROWN verification on NRT and RT MetaMoE routers across all epsilon values (5 runs each).

This script runs the verification pipeline for all 3 epsilon values:
1. NRT (Non-Robust Training) model: 5 runs per epsilon
2. RT (Robust Training) model: 5 runs per epsilon

Epsilon values tested:
- ε = 2/255 = 0.00784
- ε = 4/255 = 0.01569
- ε = 8/255 = 0.03137

Results are collected separately and formatted as LaTeX tables for the paper.
Total: 60 runs (5 runs × 3 epsilons × 2 models)

Usage (default: 20 MNIST + 20 CIFAR10, 5 runs per model per epsilon):
    python run_router_formal_verification_experiments_5_times.py

With custom sample counts:
    python run_router_formal_verification_experiments_5_times.py --num_mnist 20 --num_cifar 20

With custom number of runs per model:
    python run_router_formal_verification_experiments_5_times.py --num_runs 3
"""

import subprocess
import sys
from pathlib import Path

def run_verification_for_model(model_path, model_name, num_mnist, num_cifar, epsilon, num_runs=5):
    """
    Run the verification script multiple times for a specific model and epsilon value.
    Returns aggregated results.
    """
    project_root = Path(__file__).parent.resolve()
    script = project_root / "run_router_formal_verification.py"

    print("\n" + "="*80)
    print(f"RUNNING VERIFICATION FOR {model_name} MODEL (ε={epsilon}, {num_runs} runs)")
    print("="*80)
    print(f"Model: {model_path}")

    all_results = []

    # Run verification num_runs times
    for run_num in range(1, num_runs + 1):
        print(f"\n--- Run {run_num}/{num_runs} ---")

        # Build command with epsilon parameter
        cmd = [
            sys.executable,
            str(script),
            "--model_path", str(model_path),
            "--num_mnist", str(num_mnist),
            "--num_cifar", str(num_cifar),
            "--epsilon", str(epsilon)
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

    # Create safe filename from model_name (remove special characters)
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "")

    # Save combined results
    combined_file = project_root / f"artifacts/router_verification_{safe_name}_all_runs.txt"
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
        description='Run verification on both NRT and RT MetaMoE routers across all epsilon values'
    )
    parser.add_argument('--num_mnist', type=int, default=20,
                        help='Number of MNIST samples (default: 20)')
    parser.add_argument('--num_cifar', type=int, default=20,
                        help='Number of CIFAR10 samples (default: 20)')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs per model per epsilon (default: 5)')

    args = parser.parse_args()

    # Define epsilon values (perturbation bounds for verification)
    epsilons = [
        (0.00784, "2/255"),    # 2/255
        (0.01569, "4/255"),    # 4/255
        (0.03137, "8/255")     # 8/255
    ]

    # Model paths (single RT model for all epsilon tests)
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
    print("METAMOE ROUTER VERIFICATION - ALL EPSILON VALUES")
    print("="*80)
    print(f"\nNRT Model: {nrt_model.name}")
    print(f"RT Model: {rt_model.name}")
    print(f"MNIST samples: {args.num_mnist}")
    print(f"CIFAR10 samples: {args.num_cifar}")
    print(f"Runs per model per epsilon: {args.num_runs}")
    print(f"Total epsilon values: {len(epsilons)}")
    print(f"Total runs: {args.num_runs * len(epsilons) * 2} (5 runs × {len(epsilons)} epsilons × 2 models)")
    print(f"\nEpsilon values (perturbation bounds):")
    for eps_val, eps_frac in epsilons:
        print(f"  - ε = {eps_frac} = {eps_val}")

    all_results = {}

    # Run verification for all epsilon values
    for eps_val, eps_frac in epsilons:
        print(f"\n{'='*80}")
        print(f"EPSILON = {eps_frac} ({eps_val})")
        print(f"{'='*80}")

        # Run verification for NRT with this epsilon
        nrt_key = f"NRT_eps{eps_val}"
        nrt_results = run_verification_for_model(
            str(nrt_model), f"NRT (ε={eps_frac})", args.num_mnist, args.num_cifar, eps_val, args.num_runs
        )

        if nrt_results is None:
            print(f"\nERROR: NRT verification failed for epsilon {eps_frac}")
            continue

        all_results[nrt_key] = nrt_results

        # Run verification for RT with this epsilon
        rt_key = f"RT_eps{eps_val}"
        rt_results = run_verification_for_model(
            str(rt_model), f"RT (ε={eps_frac})", args.num_mnist, args.num_cifar, eps_val, args.num_runs
        )

        if rt_results is None:
            print(f"\nERROR: RT verification failed for epsilon {eps_frac}")
            continue

        all_results[rt_key] = rt_results

    if not all_results:
        print("\nERROR: No successful verification runs")
        sys.exit(1)

    print("\n" + "="*80)
    print("ALL VERIFICATION COMPLETE")
    print("="*80)
    print("\nResults Summary:")
    print(f"  Epsilon values tested: {len(epsilons)}")
    print(f"  Runs per model per epsilon: {args.num_runs}")
    print(f"  Total successful runs: {len(all_results) * args.num_runs}")

    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    for eps_val, eps_frac in epsilons:
        safe_eps = str(eps_val).replace(".", "_")
        nrt_key = f"NRT_eps{eps_val}"
        rt_key = f"RT_eps{eps_val}"
        if nrt_key in all_results:
            print(f"\nNRT (ε={eps_frac}) Results saved to: artifacts/router_verification_nrt_eps{safe_eps}_all_runs.txt")
        if rt_key in all_results:
            print(f"RT (ε={eps_frac}) Results saved to: artifacts/router_verification_rt_eps{safe_eps}_all_runs.txt")
    print("\nNote: Aggregate the results manually or parse the text output files to generate LaTeX tables")
    print("="*80)

if __name__ == '__main__':
    main()
