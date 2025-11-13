#!/usr/bin/env python3
"""
Script to run expert formal verification for MNIST and CIFAR10 with NRT and AT
3 epsilon values × 5 runs each, total 60 runs
Records: % verified/falsified/timeout/unknown, average verification time
"""

import subprocess
import os
import time
import re
from pathlib import Path
from collections import defaultdict
import statistics

# Repository root
REPO_ROOT = Path(__file__).parent
PAPER_ARTIFACTS_DIR = REPO_ROOT / "paper" / "artifacts"
RESULTS_FILE = REPO_ROOT / "expert_verification_results.txt"

# Verification parameters
EPSILON_VALUES = [
    (2/255, "2/255"),
    (4/255, "4/255"),
    (8/255, "8/255"),
]
NUM_RUNS = 5
NUM_IMAGES = 20
TIMEOUT = 300

# Model configurations
MODELS = {
    "MNIST_NRT": {
        "dataset": "MNIST",
        "model_path": PAPER_ARTIFACTS_DIR / "E_1_CNN_NAT" / "mnist_ultra_verifiable_cnn_best_og.pth",
        "label": "MNIST NRT",
    },
    "MNIST_AT": {
        "dataset": "MNIST",
        "model_path": PAPER_ARTIFACTS_DIR / "E_1_CNN_AT" / "mnist_ultra_verifiable_cnn_best_robust.pth",
        "label": "MNIST AT",
    },
    "CIFAR10_NRT": {
        "dataset": "CIFAR10",
        "model_path": PAPER_ARTIFACTS_DIR / "E_0_CNN_NAT" / "cifar10_ultra_verifiable_cnn_best_og.pth",
        "label": "CIFAR10 NRT",
    },
    "CIFAR10_AT": {
        "dataset": "CIFAR10",
        "model_path": PAPER_ARTIFACTS_DIR / "E_0_CNN_AT" / "cifar10_ultra_verifiable_cnn_best_robust.pth",
        "label": "CIFAR10 AT",
    },
}

def run_verification(model_path, dataset, epsilon, num_images=NUM_IMAGES, timeout=TIMEOUT):
    """Run a single verification session"""
    cmd = [
        "python",
        "src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py",
        "--model_path", str(model_path),
        "--dataset", dataset,
        "--epsilon", str(epsilon),
        "--num_images", str(num_images),
        "--timeout", str(timeout),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout * num_images + 60  # Add buffer for overhead
        )

        if result.returncode != 0:
            error_output = result.stderr if result.stderr else result.stdout
            return None, f"Verification failed with return code {result.returncode}. Error: {error_output[:200]}"

        return result.stdout + result.stderr, None

    except subprocess.TimeoutExpired:
        return None, "Verification timeout"
    except Exception as e:
        return None, f"Exception: {str(e)}"

def parse_verification_output(output):
    """Extract verification statistics from alpha-beta-CROWN output"""
    stats = {
        "verified": 0,
        "falsified": 0,
        "timeout": 0,
        "unknown": 0,
        "avg_time": 0.0,
    }

    # Look for verification summary section
    lines = output.split('\n')

    for i, line in enumerate(lines):
        # Find the summary section (marked by "Summary")
        if "Summary" in line and "#" in line:
            # Parse the next few lines for statistics
            for summary_line in lines[i:i+10]:
                # Parse: "Problem instances count: 2 , total verified (safe/unsat): 2 , total falsified (unsafe/sat): 0 , timeout: 0"
                # Use non-greedy matching [^:]* to avoid matching too far

                if "total verified" in summary_line:
                    # Match "total verified (..." up to the colon, then extract digits
                    match = re.search(r'total verified[^:]*:\s*(\d+)', summary_line)
                    if match:
                        stats["verified"] = int(match.group(1))

                if "total falsified" in summary_line:
                    # Match "total falsified (..." up to the colon, then extract digits
                    match = re.search(r'total falsified[^:]*:\s*(\d+)', summary_line)
                    if match:
                        stats["falsified"] = int(match.group(1))

                # For timeout, look for "timeout: N" but not in compound phrases
                if "timeout:" in summary_line:
                    # Match the timeout count (may be on same line as verified/falsified)
                    match = re.search(r',\s*timeout:\s*(\d+)', summary_line)
                    if match:
                        stats["timeout"] = int(match.group(1))
                    else:
                        # Fallback if format is different
                        match = re.search(r'timeout:\s*(\d+)', summary_line)
                        if match:
                            stats["timeout"] = int(match.group(1))

                # Parse: "mean time for ALL instances (total 2):4.630319451132541"
                if "mean time for ALL instances" in summary_line:
                    match = re.search(r'mean time for ALL instances[^:]*:\s*([\d.]+)', summary_line)
                    if match:
                        stats["avg_time"] = float(match.group(1))
            break

    return stats

def main():
    # Initialize results file
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write("Expert Formal Verification Experiments\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: 3 epsilon × 5 runs × 2 training modes × 2 datasets = 60 runs\n")
        f.write(f"Images per run: {NUM_IMAGES}\n")
        f.write(f"=" * 100 + "\n\n")

    # Store results for statistics calculation
    results_by_config = defaultdict(list)
    total_runs = 0
    successful_runs = 0
    failed_runs = []

    # Generate all experiment configurations
    experiments = []
    for model_key, model_config in MODELS.items():
        for epsilon, epsilon_label in EPSILON_VALUES:
            experiments.append({
                "model_key": model_key,
                "model_config": model_config,
                "epsilon": epsilon,
                "epsilon_label": epsilon_label,
            })

    total_experiments = len(experiments) * NUM_RUNS
    run_count = 0

    # Run all verification experiments
    for exp in experiments:
        model_key = exp["model_key"]
        model_config = exp["model_config"]
        epsilon = exp["epsilon"]
        epsilon_label = exp["epsilon_label"]
        dataset = model_config["dataset"]
        model_path = model_config["model_path"]
        label = model_config["label"]

        config_label = f"{label} (ε={epsilon_label})"
        config_key = f"{model_key}_eps{epsilon_label.replace('/', '_')}"

        # Check if model exists
        if not model_path.exists():
            error_msg = f"Model not found: {model_path}"
            print(f"\n✗ {config_label}: {error_msg}")
            failed_runs.append(f"{config_label}: {error_msg}")
            with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{config_label}\n")
                f.write(f"ERROR: {error_msg}\n\n")
            continue

        print(f"\n{'='*100}")
        print(f"Running verification for: {config_label}")
        print(f"{'='*100}")

        run_results = []

        for run_num in range(1, NUM_RUNS + 1):
            run_count += 1
            run_label = f"{config_label} - Run {run_num}/{NUM_RUNS}"

            print(f"\n[{run_count}/{total_experiments}] {run_label}")

            # Run verification
            output, error = run_verification(model_path, dataset, epsilon, NUM_IMAGES, TIMEOUT)

            if error:
                print(f"✗ {run_label} FAILED: {error}")
                failed_runs.append(f"{run_label}: {error}")
                with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"{run_label}\n")
                    f.write(f"ERROR: {error}\n\n")
            else:
                successful_runs += 1
                stats = parse_verification_output(output)
                run_results.append(stats)

                print(f"✓ {run_label} completed")
                print(f"  Verified: {stats['verified']}, Falsified: {stats['falsified']}, "
                      f"Timeout: {stats['timeout']}, Unknown: {stats['unknown']}")
                print(f"  Avg time: {stats['avg_time']:.2f}s")

                with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"{run_label}\n")
                    f.write(f"  Verified: {stats['verified']}\n")
                    f.write(f"  Falsified: {stats['falsified']}\n")
                    f.write(f"  Timeout: {stats['timeout']}\n")
                    f.write(f"  Unknown: {stats['unknown']}\n")
                    f.write(f"  Avg verification time: {stats['avg_time']:.2f}s\n\n")

            total_runs += 1

        # Calculate statistics for this configuration
        if run_results:
            results_by_config[config_key] = run_results
            print(f"\n{'-'*100}")
            print(f"Statistics for {config_label}:")
            print(f"{'-'*100}")

            verified_list = [r["verified"] for r in run_results]
            falsified_list = [r["falsified"] for r in run_results]
            timeout_list = [r["timeout"] for r in run_results]
            unknown_list = [r["unknown"] for r in run_results]
            time_list = [r["avg_time"] for r in run_results]

            stats_summary = f"""
  Verified:     {statistics.mean(verified_list):.1f} ± {statistics.stdev(verified_list) if len(verified_list) > 1 else 0:.1f}
  Falsified:    {statistics.mean(falsified_list):.1f} ± {statistics.stdev(falsified_list) if len(falsified_list) > 1 else 0:.1f}
  Timeout:      {statistics.mean(timeout_list):.1f} ± {statistics.stdev(timeout_list) if len(timeout_list) > 1 else 0:.1f}
  Unknown:      {statistics.mean(unknown_list):.1f} ± {statistics.stdev(unknown_list) if len(unknown_list) > 1 else 0:.1f}
  Avg time (s): {statistics.mean(time_list):.2f} ± {statistics.stdev(time_list) if len(time_list) > 1 else 0:.2f}
            """
            print(stats_summary)

            with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
                f.write(f"Summary for {config_label}:\n")
                f.write(f"  Verified:     {statistics.mean(verified_list):.1f} ± {statistics.stdev(verified_list) if len(verified_list) > 1 else 0:.1f}\n")
                f.write(f"  Falsified:    {statistics.mean(falsified_list):.1f} ± {statistics.stdev(falsified_list) if len(falsified_list) > 1 else 0:.1f}\n")
                f.write(f"  Timeout:      {statistics.mean(timeout_list):.1f} ± {statistics.stdev(timeout_list) if len(timeout_list) > 1 else 0:.1f}\n")
                f.write(f"  Unknown:      {statistics.mean(unknown_list):.1f} ± {statistics.stdev(unknown_list) if len(unknown_list) > 1 else 0:.1f}\n")
                f.write(f"  Avg time (s): {statistics.mean(time_list):.2f} ± {statistics.stdev(time_list) if len(time_list) > 1 else 0:.2f}\n")
                f.write(f"\n" + "="*100 + "\n\n")

    # Final summary
    print(f"\n{'#'*100}")
    print(f"# All verification experiments completed!")
    print(f"# Successful runs: {successful_runs}/{total_runs}")
    print(f"# Results saved to: {RESULTS_FILE}")
    print(f"{'#'*100}\n")

    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}):")
        for failed in failed_runs:
            print(f"  - {failed}")
        print()

    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(f"Experiments completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successful runs: {successful_runs}/{total_runs}\n")

        if failed_runs:
            f.write(f"\nFailed runs ({len(failed_runs)}):\n")
            for failed in failed_runs:
                f.write(f"  - {failed}\n")

        f.write("="*100 + "\n")

if __name__ == "__main__":
    main()
