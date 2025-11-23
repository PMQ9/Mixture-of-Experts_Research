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
import platform
import select

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "Vision_Transformer_Pytorch"))

# Import pty only on Unix-like systems
if platform.system() != "Windows":
    import pty

def clear_screen():
    """Clear the terminal screen."""
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def run_command(cmd, shell=False):
    """Run a command and stream output in real-time with proper tqdm support."""
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*80}\n")

    # Set environment variables for real-time output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    # Use pty on Unix-like systems for proper tqdm display
    if platform.system() != "Windows":
        return _run_command_pty(cmd, env)
    else:
        return _run_command_windows(cmd, env)

def _run_command_pty(cmd, env):
    """Run command with pty for proper terminal emulation (Unix/Linux/Mac)."""
    import fcntl
    import termios
    import struct

    output_lines = []

    def read_output(fd):
        """Read and handle output from the pty."""
        try:
            data = os.read(fd, 1024).decode('utf-8', errors='replace')
            if data:
                # Print to terminal (preserves tqdm formatting)
                sys.stdout.write(data)
                sys.stdout.flush()
                # Save to log
                output_lines.append(data)
            return data
        except OSError:
            return None

    # Create pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    # Set terminal size to prevent wrapping issues
    size = struct.pack('HHHH', 24, 80, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, size)

    # Start process with pty
    process = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=project_root,
        env=env,
        close_fds=True
    )

    os.close(slave_fd)  # Close slave in parent process

    # Read output from master
    try:
        while True:
            # Check if process is still running
            if process.poll() is not None:
                # Process finished, read remaining output
                while True:
                    data = read_output(master_fd)
                    if not data:
                        break
                break

            # Wait for data with timeout
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                data = read_output(master_fd)
                if not data:
                    break
    finally:
        os.close(master_fd)

    # Wait for process to complete
    returncode = process.wait()

    # Combine all output
    full_output = ''.join(output_lines)

    if returncode != 0:
        print(f"\nWarning: Command failed with return code {returncode}")

    return full_output, ""

def _run_command_windows(cmd, env):
    """Run command on Windows - output goes directly to terminal."""
    # Don't pipe stdout/stderr - let them go directly to terminal
    # This allows tqdm to work properly with single-bar updates
    process = subprocess.Popen(
        cmd,
        shell=False,
        cwd=project_root,
        env=env
    )

    # Wait for process to complete
    returncode = process.wait()

    # Read log from artifacts directory after training completes
    full_output = ""
    artifacts_log = project_root / "artifacts" / "training_log.txt"
    if artifacts_log.exists():
        with open(artifacts_log, 'r', encoding='utf-8', errors='replace') as f:
            full_output = f.read()

    if returncode != 0:
        print(f"\nWarning: Command failed with return code {returncode}")

    return full_output, ""

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
        r'Test acc[:\s]+([0-9.]+)%',  # Handle percentage format
    ]

    for pattern in test_acc_patterns:
        match = re.search(pattern, log_text, re.IGNORECASE)
        if match:
            acc_value = float(match.group(1))
            # If value is > 1, assume it's a percentage and convert
            if acc_value > 1:
                acc_value = acc_value / 100.0
            metrics['test_accuracy'] = acc_value
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
            acc_value = float(match.group(1))
            # If value is > 1, assume it's a percentage and convert
            if acc_value > 1:
                acc_value = acc_value / 100.0
            metrics['robust_accuracy'] = acc_value
            break

    # If robust accuracy not found, try to extract from ART attack success rate
    # Success rate of attack: X% means robust accuracy is (100 - X)%
    if metrics['robust_accuracy'] is None:
        art_pattern = r'Success rate of attack[:\s]+([0-9.]+)%'
        # Find all matches and use the last one (most recent)
        matches = list(re.finditer(art_pattern, log_text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            attack_success_rate = float(last_match.group(1))
            robust_accuracy = 100.0 - attack_success_rate
            metrics['robust_accuracy'] = robust_accuracy / 100.0  # Convert to 0-1 range

    return metrics

def train_expert_single_run(dataset, expert_id, is_adversarial, output_dir, run_number, total_run_number=None):
    """
    Train a single expert model for one run.

    Args:
        dataset: 'CIFAR10' or 'MNIST'
        expert_id: 'E_0' (CIFAR10) or 'E_1' (MNIST)
        is_adversarial: True for AT, False for NAT
        output_dir: Directory to save artifacts for this run
        run_number: Run number within this expert (1-5)
        total_run_number: Total run number across all experts (1-20)
    """
    # Clear screen before starting new training run
    clear_screen()

    training_type = "AT" if is_adversarial else "NAT"

    # Show overall progress if total_run_number is provided
    if total_run_number is not None:
        print(f"\n{'='*80}")
        print(f"OVERALL PROGRESS: RUN {total_run_number}/20")
        print(f"{'='*80}")

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
        "-u",  # Unbuffered mode for real-time output
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
            with open(artifacts_log, 'r', encoding='utf-8', errors='replace') as f:
                artifacts_log_text = f.read()
            extracted = extract_metrics(artifacts_log_text)
            if metrics['test_accuracy'] is None:
                metrics['test_accuracy'] = extracted['test_accuracy']
            if metrics['robust_accuracy'] is None:
                metrics['robust_accuracy'] = extracted['robust_accuracy']

    # If still not found, check most recent timestamped training folder
    if metrics['test_accuracy'] is None or metrics['robust_accuracy'] is None:
        artifacts_dir = project_root / "artifacts"
        training_dirs = sorted(artifacts_dir.glob("training_*"), reverse=True)
        for training_dir in training_dirs[:3]:  # Check 3 most recent
            timestamped_log = training_dir / "training_log.txt"
            if timestamped_log.exists():
                with open(timestamped_log, 'r', encoding='utf-8', errors='replace') as f:
                    timestamped_log_text = f.read()
                extracted = extract_metrics(timestamped_log_text)
                if metrics['test_accuracy'] is None:
                    metrics['test_accuracy'] = extracted['test_accuracy']
                if metrics['robust_accuracy'] is None:
                    metrics['robust_accuracy'] = extracted['robust_accuracy']
                if metrics['test_accuracy'] is not None and metrics['robust_accuracy'] is not None:
                    break  # Found both metrics, stop searching

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

    # Copy model file - check both root artifacts/ and timestamped folders
    model_copied = False

    # First try root artifacts directory
    for model_name in possible_models:
        src_model = artifacts_dir / model_name
        if src_model.exists():
            dst_model = output_dir / model_name
            shutil.copy2(src_model, dst_model)
            print(f"Copied model: {src_model} -> {dst_model}")
            model_copied = True
            break

    # If not found, look in timestamped training directories
    if not model_copied:
        training_dirs = sorted(artifacts_dir.glob("training_*"), reverse=True)
        for training_dir in training_dirs[:3]:  # Check 3 most recent
            for model_name in possible_models:
                src_model = training_dir / model_name
                if src_model.exists():
                    dst_model = output_dir / model_name
                    shutil.copy2(src_model, dst_model)
                    print(f"Copied model from timestamped folder: {src_model} -> {dst_model}")
                    model_copied = True
                    break
            if model_copied:
                break

    if not model_copied:
        print(f"Warning: Could not find model file in {artifacts_dir} or timestamped folders")
        print(f"Looked for: {possible_models}")

    # Copy other artifacts if they exist (check root first, then timestamped folders)
    for artifact in ["training_metrics.png", "training_log.txt"]:
        src_artifact = artifacts_dir / artifact
        if src_artifact.exists():
            dst_artifact = output_dir / artifact
            shutil.copy2(src_artifact, dst_artifact)
        else:
            # Try timestamped folders
            training_dirs = sorted(artifacts_dir.glob("training_*"), reverse=True)
            for training_dir in training_dirs[:3]:  # Check 3 most recent
                src_artifact = training_dir / artifact
                if src_artifact.exists():
                    dst_artifact = output_dir / artifact
                    shutil.copy2(src_artifact, dst_artifact)
                    break

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

def train_expert_multiple_runs(dataset, expert_id, is_adversarial, base_output_dir, num_runs=5, starting_run_number=1):
    """
    Train a single expert model multiple times.

    Args:
        dataset: 'CIFAR10' or 'MNIST'
        expert_id: 'E_0' (CIFAR10) or 'E_1' (MNIST)
        is_adversarial: True for AT, False for NAT
        base_output_dir: Base directory to save all runs
        num_runs: Number of times to train (default: 5)
        starting_run_number: Starting total run number for progress display (default: 1)

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
        total_run_num = starting_run_number + run_num - 1

        try:
            metrics = train_expert_single_run(
                dataset=dataset,
                expert_id=expert_id,
                is_adversarial=is_adversarial,
                output_dir=run_dir,
                run_number=run_num,
                total_run_number=total_run_num
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
        if stats['test_accuracy']['mean'] is not None:
            f.write(f"  Mean: {stats['test_accuracy']['mean']:.4f}\n")
        else:
            f.write(f"  Mean: N/A (no data)\n")
        if stats['test_accuracy']['std'] is not None:
            f.write(f"  Std:  {stats['test_accuracy']['std']:.4f}\n\n")
        else:
            f.write(f"  Std:  N/A (no data)\n\n")

        f.write("ROBUST ACCURACY (ART Attack):\n")
        f.write(f"  Individual runs: {all_metrics['robust_accuracy']}\n")
        if stats['robust_accuracy']['mean'] is not None:
            f.write(f"  Mean: {stats['robust_accuracy']['mean']:.4f}\n")
        else:
            f.write(f"  Mean: N/A (no data)\n")
        if stats['robust_accuracy']['std'] is not None:
            f.write(f"  Std:  {stats['robust_accuracy']['std']:.4f}\n")
        else:
            f.write(f"  Std:  N/A (no data)\n")

    print(f"\n{'='*80}")
    print(f"Statistics for {expert_id} ({dataset}) - {training_type}")
    print(f"{'='*80}")
    if stats['test_accuracy']['mean'] is not None and stats['test_accuracy']['std'] is not None:
        print(f"Test Accuracy:   {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
    else:
        print(f"Test Accuracy:   N/A (no data)")
    if stats['robust_accuracy']['mean'] is not None and stats['robust_accuracy']['std'] is not None:
        print(f"Robust Accuracy: {stats['robust_accuracy']['mean']:.4f} ± {stats['robust_accuracy']['std']:.4f}")
    else:
        print(f"Robust Accuracy: N/A (no data)")

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
    num_runs_per_expert = 5
    for i, expert_config in enumerate(experts, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERT {i}/4: {expert_config['expert_id']} ({expert_config['dataset']}) - {'AT' if expert_config['is_adversarial'] else 'NAT'}")
        print(f"{'#'*80}\n")

        # Calculate starting run number for this expert
        starting_run = (i - 1) * num_runs_per_expert + 1

        try:
            stats = train_expert_multiple_runs(
                **expert_config,
                num_runs=num_runs_per_expert,
                starting_run_number=starting_run
            )
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
            if stats['test_accuracy']['mean'] is not None and stats['test_accuracy']['std'] is not None:
                f.write(f"  Test Accuracy:   {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}\n")
            else:
                f.write(f"  Test Accuracy:   N/A (no data)\n")
            if stats['robust_accuracy']['mean'] is not None and stats['robust_accuracy']['std'] is not None:
                f.write(f"  Robust Accuracy: {stats['robust_accuracy']['mean']:.4f} ± {stats['robust_accuracy']['std']:.4f}\n")
            else:
                f.write(f"  Robust Accuracy: N/A (no data)\n")
            f.write(f"  Individual test accuracies: {stats['test_accuracy']['values']}\n")
            f.write(f"  Individual robust accuracies: {stats['robust_accuracy']['values']}\n")
            f.write("\n")

            # Print to console
            print(f"{expert_name}:")
            if stats['test_accuracy']['mean'] is not None and stats['test_accuracy']['std'] is not None:
                print(f"  Test Accuracy:   {stats['test_accuracy']['mean']:.4f} ± {stats['test_accuracy']['std']:.4f}")
            else:
                print(f"  Test Accuracy:   N/A (no data)")
            if stats['robust_accuracy']['mean'] is not None and stats['robust_accuracy']['std'] is not None:
                print(f"  Robust Accuracy: {stats['robust_accuracy']['mean']:.4f} ± {stats['robust_accuracy']['std']:.4f}")
            else:
                print(f"  Robust Accuracy: N/A (no data)")
            print()

    print(f"Summary saved to: {summary_file}")
    print("\nAll experts trained successfully with statistics calculated!")

if __name__ == "__main__":
    main()
