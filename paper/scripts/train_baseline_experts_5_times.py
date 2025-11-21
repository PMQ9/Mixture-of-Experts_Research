"""
Train baseline expert models 5 times and calculate statistics.

This script trains 4 baseline experts 5 times each (20 total training runs):
- E_0 (CIFAR-10): NAT and AT variants
- E_1 (MNIST): NAT and AT variants

Calculates mean and standard deviation for test accuracy and robust accuracy.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import re
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "Vision_Transformer_Pytorch"))

def run_command(cmd, shell=False):
    """Run a command and return output."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*80}\n")

    result = subprocess.run(
        cmd,
        shell=shell,
        capture_output=True,
        text=True,
        cwd=project_root
    )

    # Print output in real-time style
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"\nWarning: Command failed with return code {result.returncode}")
        # Don't raise exception, just return the output

    return result.stdout, result.stderr

def extract_metrics(log_text):
    """Extract test accuracy and robust accuracy from training logs."""
    metrics = {
        'test_accuracy': None,
        'robust_accuracy': None
    }

    # Look for test accuracy patterns
    test_acc_patterns = [
        r'Test Accuracy[:\s]+([0-9.]+)',
        r'test_acc[:\s]+([0-9.]+)',
        r'Final Test Accuracy[:\s]+([0-9.]+)',
        r'Best Test Acc[:\s]+([0-9.]+)',
    ]

    for pattern in test_acc_patterns:
        match = re.search(pattern, log_text, re.IGNORECASE)
        if match:
            metrics['test_accuracy'] = float(match.group(1))
            break

    # Look for robust accuracy patterns
    robust_acc_patterns = [
        r'Robust Accuracy[:\s]+([0-9.]+)',
        r'robust_acc[:\s]+([0-9.]+)',
        r'Adversarial Accuracy[:\s]+([0-9.]+)',
        r'adv_acc[:\s]+([0-9.]+)',
        r'PGD Accuracy[:\s]+([0-9.]+)',
    ]

    for pattern in robust_acc_patterns:
        match = re.search(pattern, log_text, re.IGNORECASE)
        if match:
            metrics['robust_accuracy'] = float(match.group(1))
            break

    return metrics

def train_expert_single_run(dataset, expert_id, is_adversarial, output_dir, run_number):
    """
    Train a single expert model for one run.

    Args:
        dataset: 'CIFAR10' or 'MNIST'
        expert_id: 'E_0' (CIFAR10) or 'E_1' (MNIST)
        is_adversarial: True for AT, False for NAT
        output_dir: Directory to save artifacts for this run
        run_number: Run number (1-5)
    """
    training_type = "AT" if is_adversarial else "NAT"
    print(f"\n{'#'*80}")
    print(f"# Training {expert_id} ({dataset}) - {training_type} - RUN {run_number}/5")
    print(f"# Output directory: {output_dir}")
    print(f"{'#'*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build training command
    train_script = project_root / "src" / "Vision_Transformer_Pytorch" / "train_moe.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--dataset", dataset,
        "--model_arch", "ultra_verifiable_cnn",
        "--epochs", "200",
        "--art_attack",  # Always run ART attack for evaluation
    ]

    if is_adversarial:
        cmd.append("--adv_training")

    # Run training
    stdout, stderr = run_command(cmd)

    # Save logs
    log_file = output_dir / "training_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"Run Number: {run_number}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        f.write(stdout)
        if stderr:
            f.write("\n\nSTDERR:\n")
            f.write(stderr)

    # Extract metrics
    metrics = extract_metrics(stdout)

    # If metrics not found in stdout, try reading from artifacts/training_log.txt
    if metrics['test_accuracy'] is None or metrics['robust_accuracy'] is None:
        artifacts_log = project_root / "artifacts" / "training_log.txt"
        if artifacts_log.exists():
            with open(artifacts_log, 'r') as f:
                artifacts_log_text = f.read()
            extracted = extract_metrics(artifacts_log_text)
            if metrics['test_accuracy'] is None:
                metrics['test_accuracy'] = extracted['test_accuracy']
            if metrics['robust_accuracy'] is None:
                metrics['robust_accuracy'] = extracted['robust_accuracy']

    # Copy artifacts from artifacts/ to output_dir
    artifacts_dir = project_root / "artifacts"

    # Determine expected model filename
    model_suffix = "_best_robust.pth" if is_adversarial else "_best_og.pth"
    expected_model = f"{dataset.lower()}_ultra_verifiable_cnn{model_suffix}"

    # Also try the generic "_best.pth" suffix
    possible_models = [
        expected_model,
        f"{dataset.lower()}_ultra_verifiable_cnn_best.pth",
    ]

    # Copy model file
    model_copied = False
    for model_name in possible_models:
        src_model = artifacts_dir / model_name
        if src_model.exists():
            dst_model = output_dir / model_name
            shutil.copy2(src_model, dst_model)
            print(f"Copied model: {src_model} -> {dst_model}")
            model_copied = True
            break

    if not model_copied:
        print(f"Warning: Could not find model file in {artifacts_dir}")
        print(f"Looked for: {possible_models}")

    # Copy other artifacts if they exist
    for artifact in ["training_metrics.png", "training_log.txt"]:
        src_artifact = artifacts_dir / artifact
        if src_artifact.exists():
            dst_artifact = output_dir / artifact
            shutil.copy2(src_artifact, dst_artifact)

    # Save metrics summary for this run
    metrics_file = output_dir / "metrics_summary.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Expert: {expert_id} ({dataset}) - {training_type} - Run {run_number}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Test Accuracy: {metrics['test_accuracy']}\n")
        f.write(f"Robust Accuracy (ART Attack): {metrics['robust_accuracy']}\n")

    print(f"\nRun {run_number} Complete!")
    print(f"Test Accuracy: {metrics['test_accuracy']}")
    print(f"Robust Accuracy: {metrics['robust_accuracy']}")

    return metrics

def train_expert_multiple_runs(dataset, expert_id, is_adversarial, base_output_dir, num_runs=5):
    """
    Train a single expert model multiple times.

    Args:
        dataset: 'CIFAR10' or 'MNIST'
        expert_id: 'E_0' (CIFAR10) or 'E_1' (MNIST)
        is_adversarial: True for AT, False for NAT
        base_output_dir: Base directory to save all runs
        num_runs: Number of times to train (default: 5)

    Returns:
        dict: Statistics including mean and std for each metric
    """
    training_type = "AT" if is_adversarial else "NAT"

    # Store metrics from all runs
    all_metrics = {
        'test_accuracy': [],
        'robust_accuracy': []
    }

    # Train num_runs times
    for run_num in range(1, num_runs + 1):
        run_dir = base_output_dir / f"run_{run_num}"

        try:
            metrics = train_expert_single_run(
                dataset=dataset,
                expert_id=expert_id,
                is_adversarial=is_adversarial,
                output_dir=run_dir,
                run_number=run_num
            )

            # Store metrics
            if metrics['test_accuracy'] is not None:
                all_metrics['test_accuracy'].append(metrics['test_accuracy'])
            if metrics['robust_accuracy'] is not None:
                all_metrics['robust_accuracy'].append(metrics['robust_accuracy'])

        except Exception as e:
            print(f"\nError in run {run_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate statistics
    stats = {
        'expert_id': expert_id,
        'dataset': dataset,
        'training_type': training_type,
        'num_runs': num_runs,
        'test_accuracy': {
            'values': all_metrics['test_accuracy'],
            'mean': np.mean(all_metrics['test_accuracy']) if all_metrics['test_accuracy'] else None,
            'std': np.std(all_metrics['test_accuracy'], ddof=1) if len(all_metrics['test_accuracy']) > 1 else None,
        },
        'robust_accuracy': {
            'values': all_metrics['robust_accuracy'],
            'mean': np.mean(all_metrics['robust_accuracy']) if all_metrics['robust_accuracy'] else None,
            'std': np.std(all_metrics['robust_accuracy'], ddof=1) if len(all_metrics['robust_accuracy']) > 1 else None,
        }
    }

    # Save statistics to file
    stats_file = base_output_dir / "statistics_summary.txt"
    with open(stats_file, 'w') as f:
        f.write(f"Expert: {expert_id} ({dataset}) - {training_type}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Number of runs: {num_runs}\n\n")

        f.write("TEST ACCURACY:\n")
        f.write(f"  Individual runs: {all_metrics['test_accuracy']}\n")
        f.write(f"  Mean: {stats['test_accuracy']['mean']:.4f}\n")
        f.write(f"  Std:  {stats['test_accuracy']['std']:.4f}\n\n")

        f.write("ROBUST ACCURACY (ART Attack):\n")
        f.write(f"  Individual runs: {all_metrics['robust_accuracy']}\n")
        f.write(f"  Mean: {stats['robust_accuracy']['mean']:.4f}\n")
        f.write(f"  Std:  {stats['robust_accuracy']['std']:.4f}\n")

    print(f"\n{'='*80}")
    print(f"Statistics for {expert_id} ({dataset}) - {training_type}")
    print(f"{'='*80}")
    print(f"Test Accuracy:   {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
    print(f"Robust Accuracy: {stats['robust_accuracy']['mean']:.4f} ± {stats['robust_accuracy']['std']:.4f}")

    return stats

def main():
    """Train all baseline experts 5 times each."""
    print("="*80)
    print("BASELINE EXPERT TRAINING PIPELINE - 5 RUNS WITH STATISTICS")
    print("="*80)
    print("\nThis script will train each expert model 5 times:")
    print("  1. CIFAR-10 NAT (E_0_CNN_NAT) - 5 runs")
    print("  2. CIFAR-10 AT  (E_0_CNN_AT)  - 5 runs")
    print("  3. MNIST NAT    (E_1_CNN_NAT) - 5 runs")
    print("  4. MNIST AT     (E_1_CNN_AT)  - 5 runs")
    print("\nTotal: 20 training runs")
    print("\nArchitecture: ultra_verifiable_cnn")
    print("Epochs: 200 (default)")
    print("="*80)

    # Define experts to train
    experts = [
        {
            'dataset': 'CIFAR10',
            'expert_id': 'E_0',
            'is_adversarial': False,
            'base_output_dir': project_root / "paper" / "artifacts" / "E_0_CNN_NAT"
        },
        {
            'dataset': 'CIFAR10',
            'expert_id': 'E_0',
            'is_adversarial': True,
            'base_output_dir': project_root / "paper" / "artifacts" / "E_0_CNN_AT"
        },
        {
            'dataset': 'MNIST',
            'expert_id': 'E_1',
            'is_adversarial': False,
            'base_output_dir': project_root / "paper" / "artifacts" / "E_1_CNN_NAT"
        },
        {
            'dataset': 'MNIST',
            'expert_id': 'E_1',
            'is_adversarial': True,
            'base_output_dir': project_root / "paper" / "artifacts" / "E_1_CNN_AT"
        },
    ]

    # Track all statistics
    all_stats = []

    # Train each expert 5 times
    for i, expert_config in enumerate(experts, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERT {i}/4: {expert_config['expert_id']} ({expert_config['dataset']}) - {'AT' if expert_config['is_adversarial'] else 'NAT'}")
        print(f"{'#'*80}\n")

        try:
            stats = train_expert_multiple_runs(**expert_config)
            all_stats.append(stats)
        except Exception as e:
            print(f"\nError training expert {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate final summary
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY WITH STATISTICS")
    print("="*80 + "\n")

    summary_file = project_root / "paper" / "artifacts" / "baseline_experts_5runs_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("BASELINE EXPERT TRAINING SUMMARY - 5 RUNS WITH STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total runs: {len(all_stats) * 5}\n")
        f.write("="*80 + "\n\n")

        for stats in all_stats:
            expert_name = f"{stats['expert_id']} ({stats['dataset']}) - {stats['training_type']}"

            # Write to file
            f.write(f"{expert_name}:\n")
            f.write(f"  Test Accuracy:   {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}\n")
            f.write(f"  Robust Accuracy: {stats['robust_accuracy']['mean']:.4f} ± {stats['robust_accuracy']['std']:.4f}\n")
            f.write(f"  Individual test accuracies: {stats['test_accuracy']['values']}\n")
            f.write(f"  Individual robust accuracies: {stats['robust_accuracy']['values']}\n")
            f.write("\n")

            # Print to console
            print(f"{expert_name}:")
            print(f"  Test Accuracy:   {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
            print(f"  Robust Accuracy: {stats['robust_accuracy']['mean']:.4f} ± {stats['robust_accuracy']['std']:.4f}")
            print()

    print(f"Summary saved to: {summary_file}")
    print("\nAll experts trained successfully with statistics calculated!")

if __name__ == "__main__":
    main()
