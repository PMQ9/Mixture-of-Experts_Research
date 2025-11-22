#!/usr/bin/env python3
"""
Script to run MoE training experiments with different expert combinations.

Experiment structure:
- MoE with AT Router (20 runs):
  - 5 runs: CIFAR10-AT + MNIST-AT experts
  - 5 runs: CIFAR10-NAT + MNIST-NAT experts
  - 5 runs: CIFAR10-AT + MNIST-NAT experts
  - 5 runs: CIFAR10-NAT + MNIST-AT experts

- MoE with NAT Router (20 runs):
  - Same 4 expert combinations

Total: 40 runs
"""

import subprocess
import os
import time
from pathlib import Path

# Repository root (navigate from paper/scripts/ back to root)
REPO_ROOT = Path(__file__).parent.parent.parent
PAPER_ARTIFACTS_DIR = REPO_ROOT / "paper" / "artifacts"
SUMMARY_FILE = REPO_ROOT / "test_summary_moe.txt"

# Expert model paths
CIFAR10_AT = PAPER_ARTIFACTS_DIR / "E_0_CNN_AT" / "cifar10_ultra_verifiable_cnn_best_robust.pth"
CIFAR10_NAT = PAPER_ARTIFACTS_DIR / "E_0_CNN_NAT" / "cifar10_ultra_verifiable_cnn_best_og.pth"
MNIST_AT = PAPER_ARTIFACTS_DIR / "E_1_CNN_AT" / "mnist_ultra_verifiable_cnn_best_robust.pth"
MNIST_NAT = PAPER_ARTIFACTS_DIR / "E_1_CNN_NAT" / "mnist_ultra_verifiable_cnn_best_og.pth"

# Training parameters
MODEL_ARCH = "ultra_verifiable_cnn"
GATING_BACKBONE = "ultra_verifiable_cnn"
EPOCHS = 50
TEST_START_EPOCH = 0


def find_latest_training_log():
    """Find the most recently created training_* folder and return path to its training_log.txt"""
    training_folders = list(REPO_ROOT.glob("artifacts/training_*"))

    if not training_folders:
        return None

    # Sort by modification time, most recent first
    latest_folder = max(training_folders, key=lambda p: p.stat().st_mtime)
    log_path = latest_folder / "training_log.txt"

    return log_path if log_path.exists() else None


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

        # Stop after "Clean Accuracy" or "Adversarial Accuracy" or other completion markers
        if "Accuracy:" in line and ("Clean" in line or "Adversarial" in line):
            break

    return '\n'.join(results) + '\n'


def run_training(cifar10_model_path, mnist_model_path, adv_gating=False):
    """Run a single MoE training session"""
    cmd = [
        "python", "train.py",
        "--meta_moe",
        "--model_arch", MODEL_ARCH,
        "--gating_backbone", GATING_BACKBONE,
        "--cifar10_model_path", str(cifar10_model_path),
        "--mnist_model_path", str(mnist_model_path),
        "--epochs", str(EPOCHS),
        "--test_start_epoch", str(TEST_START_EPOCH),
        "--art_attack"
    ]

    if adv_gating:
        cmd.append("--adv_gating_train")

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
        f.write(f"MoE Training Experiments Summary\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*80 + "\n\n")

    # Define experiments
    # Format: (cifar10_path, mnist_path, adv_gating, label, num_runs)
    experiments = [
        # MoE with AT Router (20 runs)
        (CIFAR10_AT, MNIST_AT, True, "MoE AT Router with CIFAR10-AT + MNIST-AT", 5),
        (CIFAR10_NAT, MNIST_NAT, True, "MoE AT Router with CIFAR10-NAT + MNIST-NAT", 5),
        (CIFAR10_AT, MNIST_NAT, True, "MoE AT Router with CIFAR10-AT + MNIST-NAT", 5),
        (CIFAR10_NAT, MNIST_AT, True, "MoE AT Router with CIFAR10-NAT + MNIST-AT", 5),
        # MoE with NAT Router (20 runs)
        (CIFAR10_AT, MNIST_AT, False, "MoE NAT Router with CIFAR10-AT + MNIST-AT", 5),
        (CIFAR10_NAT, MNIST_NAT, False, "MoE NAT Router with CIFAR10-NAT + MNIST-NAT", 5),
        (CIFAR10_AT, MNIST_NAT, False, "MoE NAT Router with CIFAR10-AT + MNIST-NAT", 5),
        (CIFAR10_NAT, MNIST_AT, False, "MoE NAT Router with CIFAR10-NAT + MNIST-AT", 5),
    ]

    total_runs = 0
    successful_runs = 0

    for cifar10_path, mnist_path, adv_gating, label, num_runs in experiments:
        for run_num in range(1, num_runs + 1):
            total_runs += 1
            run_label = f"Run {run_num}: {label}"

            print(f"\n{'#'*80}")
            print(f"# Starting {run_label} (Total progress: {total_runs}/40)")
            print(f"{'#'*80}\n")

            # Run training
            success = run_training(cifar10_path, mnist_path, adv_gating)

            if success:
                successful_runs += 1
                # Wait a moment for file system to sync and archive to complete
                time.sleep(2)

                # Find and extract results from the latest training log
                log_path = find_latest_training_log()
                if log_path:
                    results = extract_results_from_log(log_path)
                    print(f"Reading results from: {log_path}")
                else:
                    results = "ERROR: Could not find training_log.txt in artifacts folder!\n"

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
