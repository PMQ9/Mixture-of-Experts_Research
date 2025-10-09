"""
Automated benchmarking script for training and evaluating models across datasets.
Runs multiple training iterations, tests with adversarial attacks, and collects results.

Usage:
    python benchmarks/auto_benchmark.py --epochs 100 --runs 3 --datasets GTSRB CIFAR10 MNIST
"""

import argparse
import subprocess
import os
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Default configurations
DEFAULT_EPOCHS = 100
DEFAULT_RUNS = 3
DEFAULT_DATASETS = ['GTSRB', 'CIFAR10', 'MNIST']
DEFAULT_MODEL_ARCH = 'small_cnn'
DEFAULT_ATTACK_MODES = ['PGD']

# Paths
RESULTS_DIR = Path(__file__).parent.parent / 'artifacts' / 'benchmark_results'
TRAIN_SCRIPT = Path(__file__).parent.parent / 'src' / 'Vision_Transformer_Pytorch' / 'train_moe.py'


def parse_args():
    parser = argparse.ArgumentParser(description='Automated benchmark testing')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Number of epochs per training run (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS,
                        help=f'Number of training runs per configuration (default: {DEFAULT_RUNS})')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                        help=f'Datasets to benchmark (default: {" ".join(DEFAULT_DATASETS)})')
    parser.add_argument('--model_arch', type=str, default=DEFAULT_MODEL_ARCH,
                        help=f'Model architecture (default: {DEFAULT_MODEL_ARCH})')
    parser.add_argument('--attack_modes', nargs='+', default=DEFAULT_ATTACK_MODES,
                        help=f'Attack modes for adversarial testing (default: {" ".join(DEFAULT_ATTACK_MODES)})')
    parser.add_argument('--skip_clean_training', action='store_true',
                        help='Skip clean (non-adversarial) training')
    parser.add_argument('--skip_adv_training', action='store_true',
                        help='Skip adversarial training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory for results')
    return parser.parse_args()


def setup_results_dir(output_dir=None):
    """Create results directory with timestamp"""
    if output_dir:
        results_path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = RESULTS_DIR / f'run_{timestamp}'

    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def run_training(dataset, model_arch, epochs, device, adv_training=False, run_num=1):
    """Run a single training job"""
    print(f"\n{'='*80}")
    print(f"Training: {dataset} | Arch: {model_arch} | Adv: {adv_training} | Run: {run_num}/{epochs} epochs")
    print(f"{'='*80}\n")

    cmd = [
        'python', str(TRAIN_SCRIPT),
        '--dataset', dataset,
        '--model_arch', model_arch,
        '--epochs', str(epochs),
        '--device', device
    ]

    if adv_training:
        cmd.append('--adv_training')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed!")
        print(e.stdout)
        print(e.stderr)
        return False, e.stdout


def run_adversarial_test(dataset, model_arch, device, attack_mode='PGD', adv_trained=False):
    """Run adversarial robustness testing"""
    print(f"\n{'='*80}")
    print(f"Testing: {dataset} | Attack: {attack_mode} | Adv Trained: {adv_trained}")
    print(f"{'='*80}\n")

    # Find the most recent model
    model_suffix = 'robust' if adv_trained else 'best'
    model_pattern = f"{dataset.lower()}_{model_arch}*_{model_suffix}.pth"

    cmd = [
        'python', str(TRAIN_SCRIPT),
        '--dataset', dataset,
        '--model_arch', model_arch,
        '--art_attack',
        '--art_attack_mode', attack_mode,
        '--device', device
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Adversarial test failed!")
        print(e.stdout)
        print(e.stderr)
        return False, e.stdout


def extract_accuracy(output_text, metric_name='Test Accuracy'):
    """Extract accuracy from training output"""
    # Try different patterns
    patterns = [
        rf'{metric_name}:\s*(\d+\.\d+)%',
        rf'{metric_name}:\s*(\d+\.\d+)',
        rf'Best Accuracy:\s*(\d+\.\d+)',
        rf'Final.*?accuracy:\s*(\d+\.\d+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output_text, re.IGNORECASE)
        if matches:
            # Return the last (best) match
            return float(matches[-1])

    return None


def extract_adv_accuracy(output_text, attack_mode='PGD'):
    """Extract adversarial test accuracy"""
    patterns = [
        rf'{attack_mode}.*?accuracy:\s*(\d+\.\d+)%',
        rf'{attack_mode}.*?accuracy:\s*(\d+\.\d+)',
        rf'Adversarial accuracy:\s*(\d+\.\d+)%',
        rf'Adversarial accuracy:\s*(\d+\.\d+)',
        rf'Robust accuracy:\s*(\d+\.\d+)%',
        rf'Robust accuracy:\s*(\d+\.\d+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output_text, re.IGNORECASE)
        if matches:
            return float(matches[-1])

    return None


def run_benchmark(args):
    """Main benchmark loop"""
    results_dir = setup_results_dir(args.output_dir)
    print(f"\nResults will be saved to: {results_dir}\n")

    # Store all results
    all_results = []

    # Training configurations
    training_configs = []
    if not args.skip_clean_training:
        training_configs.append(('NAT', False))  # Natural (clean) training
    if not args.skip_adv_training:
        training_configs.append(('AT', True))    # Adversarial training

    for dataset in args.datasets:
        dataset_results = {
            'dataset': dataset,
            'model_arch': args.model_arch,
            'epochs': args.epochs,
            'runs': args.runs
        }

        for train_type, adv_training in training_configs:
            # Storage for multiple runs
            clean_accs = []
            adv_accs = {attack: [] for attack in args.attack_modes}

            # Run multiple training iterations
            for run_num in range(1, args.runs + 1):
                success, output = run_training(
                    dataset=dataset,
                    model_arch=args.model_arch,
                    epochs=args.epochs,
                    device=args.device,
                    adv_training=adv_training,
                    run_num=run_num
                )

                if not success:
                    print(f"⚠️  Training run {run_num} failed for {dataset} ({train_type})")
                    continue

                # Extract clean test accuracy
                clean_acc = extract_accuracy(output)
                if clean_acc is not None:
                    clean_accs.append(clean_acc)
                    print(f"✓ Clean accuracy: {clean_acc:.2f}%")

                # Run adversarial tests
                for attack_mode in args.attack_modes:
                    success_adv, output_adv = run_adversarial_test(
                        dataset=dataset,
                        model_arch=args.model_arch,
                        device=args.device,
                        attack_mode=attack_mode,
                        adv_trained=adv_training
                    )

                    if success_adv:
                        adv_acc = extract_adv_accuracy(output_adv, attack_mode)
                        if adv_acc is not None:
                            adv_accs[attack_mode].append(adv_acc)
                            print(f"✓ {attack_mode} accuracy: {adv_acc:.2f}%")

            # Calculate statistics
            if clean_accs:
                dataset_results[f'{train_type}_Clean_mean'] = np.mean(clean_accs)
                dataset_results[f'{train_type}_Clean_std'] = np.std(clean_accs)
                dataset_results[f'{train_type}_Clean_runs'] = clean_accs

            for attack_mode in args.attack_modes:
                if adv_accs[attack_mode]:
                    dataset_results[f'{train_type}_{attack_mode}_mean'] = np.mean(adv_accs[attack_mode])
                    dataset_results[f'{train_type}_{attack_mode}_std'] = np.std(adv_accs[attack_mode])
                    dataset_results[f'{train_type}_{attack_mode}_runs'] = adv_accs[attack_mode]

        all_results.append(dataset_results)

    # Save results
    save_results(all_results, results_dir, args)

    return all_results, results_dir


def save_results(results, results_dir, args):
    """Save benchmark results to files"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save raw JSON results
    json_path = results_dir / 'results_raw.json'
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': vars(args),
            'results': results
        }, f, indent=2)
    print(f"\n✓ Raw results saved to: {json_path}")

    # Create summary table (like the reference table)
    summary_data = []
    for res in results:
        row = {'Dataset': res['dataset']}

        # Add columns for each training type and test condition
        for train_type in ['NAT', 'AT']:
            for test_type in ['Clean'] + args.attack_modes:
                key = f'{train_type}_{test_type}_mean'
                if key in res:
                    # Format as percentage with std
                    mean_val = res[key]
                    std_key = f'{train_type}_{test_type}_std'
                    std_val = res.get(std_key, 0)
                    row[f'{train_type}_{test_type}'] = f"{mean_val:.2f}% (±{std_val:.2f})"
                else:
                    row[f'{train_type}_{test_type}'] = 'N/A'

        summary_data.append(row)

    # Save summary as CSV
    df = pd.DataFrame(summary_data)
    csv_path = results_dir / 'results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Summary table saved to: {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")

    # Save configuration
    config_path = results_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Configuration saved to: {config_path}")


def run_analysis(results_dir):
    """Run the analyze_benchmark script on the results"""
    analyze_script = Path(__file__).parent / 'analyze_benchmark.py'

    print(f"\n{'='*80}")
    print("RUNNING ANALYSIS")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            ['python', str(analyze_script), str(results_dir)],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Analysis failed. You can run it manually later:")
        print(f"  python benchmarks/analyze_benchmark.py {results_dir}")
        return False


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print("AUTOMATED BENCHMARK TESTING")
    print(f"{'='*80}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Model Architecture: {args.model_arch}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Runs per configuration: {args.runs}")
    print(f"Attack modes: {', '.join(args.attack_modes)}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    confirm = input("Start benchmark? (y/n): ")
    if confirm.lower() != 'y':
        print("Benchmark cancelled.")
        return

    # Run benchmark and get results
    results, results_dir = run_benchmark(args)

    print("\n✓ Benchmark completed!")

    # Automatically run analysis
    run_analysis(results_dir)


if __name__ == '__main__':
    main()
