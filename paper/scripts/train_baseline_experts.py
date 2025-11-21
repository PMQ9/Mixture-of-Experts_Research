"""
Train baseline expert models for the research paper.

This script trains 4 baseline experts:
- E_0 (CIFAR-10): NAT and AT variants
- E_1 (MNIST): NAT and AT variants

Each expert uses ultra_verifiable_cnn architecture with 200 epochs (default).
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import re
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
        print(f"\nError: Command failed with return code {returncode}")
        raise subprocess.CalledProcessError(returncode, cmd)

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
        print(f"\nError: Command failed with return code {returncode}")
        raise subprocess.CalledProcessError(returncode, cmd)

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
    ]

    for pattern in robust_acc_patterns:
        match = re.search(pattern, log_text, re.IGNORECASE)
        if match:
            metrics['robust_accuracy'] = float(match.group(1))
            break

    return metrics

def train_expert(dataset, expert_id, is_adversarial, output_dir):
    """
    Train a single expert model.

    Args:
        dataset: 'CIFAR10' or 'MNIST'
        expert_id: 'E_0' (CIFAR10) or 'E_1' (MNIST)
        is_adversarial: True for AT, False for NAT
        output_dir: Directory to save artifacts
    """
    # Clear screen before starting new training run
    clear_screen()

    training_type = "AT" if is_adversarial else "NAT"
    print(f"\n{'#'*80}")
    print(f"# Training {expert_id} ({dataset}) - {training_type}")
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
        f.write(stdout)
        if stderr:
            f.write("\n\nSTDERR:\n")
            f.write(stderr)

    # Extract metrics
    metrics = extract_metrics(stdout)

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
            print(f"Copied artifact: {artifact}")

    # Save metrics summary
    metrics_file = output_dir / "metrics_summary.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"Expert: {expert_id} ({dataset}) - {training_type}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Test Accuracy: {metrics['test_accuracy']}\n")
        f.write(f"Robust Accuracy (ART Attack): {metrics['robust_accuracy']}\n")

    print(f"\n{training_type} Expert Training Complete!")
    print(f"Test Accuracy: {metrics['test_accuracy']}")
    print(f"Robust Accuracy: {metrics['robust_accuracy']}")

    return metrics

def main():
    """Train all baseline experts."""
    print("="*80)
    print("BASELINE EXPERT TRAINING PIPELINE")
    print("="*80)
    print("\nThis script will train 4 expert models:")
    print("  1. CIFAR-10 NAT (E_0_CNN_NAT)")
    print("  2. CIFAR-10 AT  (E_0_CNN_AT)")
    print("  3. MNIST NAT    (E_1_CNN_NAT)")
    print("  4. MNIST AT     (E_1_CNN_AT)")
    print("\nArchitecture: ultra_verifiable_cnn")
    print("Epochs: 200 (default)")
    print("="*80)

    # Define experts to train
    experts = [
        {
            'dataset': 'CIFAR10',
            'expert_id': 'E_0',
            'is_adversarial': False,
            'output_dir': project_root / "paper" / "artifacts" / "E_0_CNN_NAT"
        },
        {
            'dataset': 'CIFAR10',
            'expert_id': 'E_0',
            'is_adversarial': True,
            'output_dir': project_root / "paper" / "artifacts" / "E_0_CNN_AT"
        },
        {
            'dataset': 'MNIST',
            'expert_id': 'E_1',
            'is_adversarial': False,
            'output_dir': project_root / "paper" / "artifacts" / "E_1_CNN_NAT"
        },
        {
            'dataset': 'MNIST',
            'expert_id': 'E_1',
            'is_adversarial': True,
            'output_dir': project_root / "paper" / "artifacts" / "E_1_CNN_AT"
        },
    ]

    # Track all results
    all_results = []

    # Train each expert
    for i, expert_config in enumerate(experts, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERT {i}/4")
        print(f"{'#'*80}\n")

        try:
            metrics = train_expert(**expert_config)
            all_results.append({
                'expert_id': expert_config['expert_id'],
                'dataset': expert_config['dataset'],
                'training_type': 'AT' if expert_config['is_adversarial'] else 'NAT',
                'metrics': metrics
            })
        except Exception as e:
            print(f"\nError training expert {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate final summary
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80 + "\n")

    summary_file = project_root / "paper" / "artifacts" / "baseline_experts_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("BASELINE EXPERT TRAINING SUMMARY\n")
        f.write("="*80 + "\n\n")

        for result in all_results:
            line = f"{result['expert_id']} ({result['dataset']}) - {result['training_type']}: "
            line += f"Test Acc = {result['metrics']['test_accuracy']}, "
            line += f"Robust Acc = {result['metrics']['robust_accuracy']}\n"
            f.write(line)
            print(line.strip())

    print(f"\nSummary saved to: {summary_file}")
    print("\nAll experts trained successfully!")

if __name__ == "__main__":
    main()
