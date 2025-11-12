#!/usr/bin/env python3
"""
Script to run training experiments for MNIST and CIFAR10 with NAT and AT
5 runs each, total 20 runs
"""

import subprocess
import os
import time
from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
TRAINING_LOG = ARTIFACTS_DIR / "training_log.txt"
SUMMARY_FILE = REPO_ROOT / "test_summary.txt"

def extract_results_from_log(log_path):
    """Extract key results from training_log.txt"""
    if not log_path.exists():
        return "ERROR: training_log.txt not found!\n"

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Find the results section (starts from "Training completed")
    results = []
    capture = False

    for line in lines:
        # Start capturing from "Training completed"
        if "Training completed." in line:
            capture = True

        if capture:
            results.append(line.rstrip())

        # Stop after "Clean Accuracy" or if we hit "All artifacts moved"
        if "Clean Accuracy:" in line or "Adversarial Accuracy:" in line:
            break

    return '\n'.join(results) + '\n'

def run_training(dataset, adv_training=False):
    """Run a single training session"""
    cmd = [
        "python", "train.py",
        "--dataset", dataset,
        "--model_arch", "ultra_verifiable_cnn",
        "--epochs", "200",
        "--test_start_epoch", "0",
        "--art_attack"
    ]

    if adv_training:
        cmd.append("--adv_training")

    print(f"\n{'='*60}")
    print(f"Running command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        return False

    return True

def main():
    # Initialize summary file
    with open(SUMMARY_FILE, 'w') as f:
        f.write(f"Training Experiments Summary\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*80 + "\n\n")

    experiments = [
        ("MNIST", False, "MNIST NAT", 5),
        ("MNIST", True, "MNIST AT", 5),
        ("CIFAR10", False, "CIFAR10 NAT", 5),
        ("CIFAR10", True, "CIFAR10 AT", 5),
    ]

    total_runs = 0
    successful_runs = 0

    for dataset, adv_training, label, num_runs in experiments:
        for run_num in range(1, num_runs + 1):
            total_runs += 1
            run_label = f"Run {run_num}: {label}"

            print(f"\n{'#'*80}")
            print(f"# Starting {run_label} (Total progress: {total_runs}/20)")
            print(f"{'#'*80}\n")

            # Run training
            success = run_training(dataset, adv_training)

            if success:
                successful_runs += 1
                # Wait a moment for file system to sync
                time.sleep(2)

                # Extract results
                results = extract_results_from_log(TRAINING_LOG)

                # Append to summary file
                with open(SUMMARY_FILE, 'a') as f:
                    f.write("="*80 + "\n")
                    f.write(f"{run_label}\n")
                    f.write("="*80 + "\n")
                    f.write(results)
                    f.write("\n")

                print(f"\n✓ {run_label} completed successfully")
                print(f"Results appended to {SUMMARY_FILE}")
            else:
                print(f"\n✗ {run_label} FAILED")
                with open(SUMMARY_FILE, 'a') as f:
                    f.write("="*80 + "\n")
                    f.write(f"{run_label}\n")
                    f.write("="*80 + "\n")
                    f.write("ERROR: Training failed!\n\n")

    # Final summary
    print(f"\n{'#'*80}")
    print(f"# All experiments completed!")
    print(f"# Successful: {successful_runs}/{total_runs}")
    print(f"# Summary saved to: {SUMMARY_FILE}")
    print(f"{'#'*80}\n")

    with open(SUMMARY_FILE, 'a') as f:
        f.write("="*80 + "\n")
        f.write(f"Experiments completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Successful runs: {successful_runs}/{total_runs}\n")
        f.write("="*80 + "\n")

if __name__ == "__main__":
    main()
