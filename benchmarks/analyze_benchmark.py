"""
Analyze benchmark results from log files and generate summary table.

Usage:
    python benchmarks/analyze_benchmark.py <results_directory>
    python benchmarks/analyze_benchmark.py artifacts/benchmark_results/run_20250108_143000
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict


def extract_test_accuracy(log_file):
    """Extract test accuracy from training log"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Patterns to match accuracy values
        patterns = [
            r'Best Accuracy:\s*(\d+\.\d+)',
            r'Test Accuracy:\s*(\d+\.\d+)%',
            r'Test Accuracy:\s*(\d+\.\d+)',
            r'Final.*?accuracy:\s*(\d+\.\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Return the best (last) match
                return float(matches[-1])

        return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def extract_adversarial_accuracy(log_file):
    """Extract adversarial test accuracy from log"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Patterns for adversarial accuracy
        patterns = [
            r'PGD.*?accuracy:\s*(\d+\.\d+)%',
            r'PGD.*?accuracy:\s*(\d+\.\d+)',
            r'FGSM.*?accuracy:\s*(\d+\.\d+)%',
            r'FGSM.*?accuracy:\s*(\d+\.\d+)',
            r'Adversarial accuracy:\s*(\d+\.\d+)%',
            r'Adversarial accuracy:\s*(\d+\.\d+)',
            r'Robust accuracy:\s*(\d+\.\d+)%',
            r'Robust accuracy:\s*(\d+\.\d+)',
            r'Attack success rate.*?accuracy:\s*(\d+\.\d+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return float(matches[-1])

        return None
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def parse_log_filename(filename):
    """Parse dataset, training type, and run number from log filename"""
    # Expected format: DATASET_TRAININGTYPE_runN.log
    # Example: GTSRB_NAT_run1.log, CIFAR10_AT_run2.log
    match = re.match(r'(\w+)_(NAT|AT)_run(\d+)\.log', filename)
    if match:
        dataset = match.group(1)
        training_type = match.group(2)
        run_num = int(match.group(3))
        return dataset, training_type, run_num
    return None, None, None


def analyze_results(results_dir):
    """Analyze all log files in results directory"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return None

    # Find all log files
    log_files = list(results_path.glob('*.log'))

    if not log_files:
        print(f"Error: No log files found in {results_dir}")
        return None

    print(f"Found {len(log_files)} log files")

    # Organize results by dataset and training type
    results = defaultdict(lambda: defaultdict(lambda: {'clean': [], 'adv': []}))

    for log_file in log_files:
        dataset, training_type, run_num = parse_log_filename(log_file.name)

        if not dataset:
            print(f"Warning: Could not parse filename: {log_file.name}")
            continue

        print(f"Processing: {log_file.name}...")

        # Extract accuracies
        clean_acc = extract_test_accuracy(log_file)
        adv_acc = extract_adversarial_accuracy(log_file)

        if clean_acc is not None:
            results[dataset][training_type]['clean'].append(clean_acc)
            print(f"  Clean accuracy: {clean_acc:.2f}%")
        else:
            print(f"  Warning: Could not extract clean accuracy")

        if adv_acc is not None:
            results[dataset][training_type]['adv'].append(adv_acc)
            print(f"  Adversarial accuracy: {adv_acc:.2f}%")
        else:
            print(f"  Warning: Could not extract adversarial accuracy")

    return results


def create_summary_table(results):
    """Create summary table like the reference format"""
    summary_data = []

    # Dataset order
    dataset_order = ['GTSRB', 'CIFAR10', 'MNIST', 'CIFAR', 'TSRD', 'BTSD', 'ETSD']

    for dataset in dataset_order:
        if dataset not in results:
            continue

        row = {'Expert': dataset}

        # NAT_Clean
        if 'NAT' in results[dataset] and results[dataset]['NAT']['clean']:
            clean_accs = results[dataset]['NAT']['clean']
            row['NAT_Clean'] = f"{np.mean(clean_accs):.2f}%"
        else:
            row['NAT_Clean'] = 'N/A'

        # NAT_Adv
        if 'NAT' in results[dataset] and results[dataset]['NAT']['adv']:
            adv_accs = results[dataset]['NAT']['adv']
            row['NAT_Adv'] = f"{np.mean(adv_accs):.2f}%"
        else:
            row['NAT_Adv'] = 'N/A'

        # AT_Clean
        if 'AT' in results[dataset] and results[dataset]['AT']['clean']:
            clean_accs = results[dataset]['AT']['clean']
            row['AT_Clean'] = f"{np.mean(clean_accs):.2f}%"
        else:
            row['AT_Clean'] = 'N/A'

        # AT_Adv
        if 'AT' in results[dataset] and results[dataset]['AT']['adv']:
            adv_accs = results[dataset]['AT']['adv']
            row['AT_Adv'] = f"{np.mean(adv_accs):.2f}%"
        else:
            row['AT_Adv'] = 'N/A'

        summary_data.append(row)

    return pd.DataFrame(summary_data)


def create_detailed_table(results):
    """Create detailed table with mean ± std"""
    detailed_data = []

    dataset_order = ['GTSRB', 'CIFAR10', 'MNIST', 'CIFAR', 'TSRD', 'BTSD', 'ETSD']

    for dataset in dataset_order:
        if dataset not in results:
            continue

        row = {'Dataset': dataset}

        # NAT_Clean
        if 'NAT' in results[dataset] and results[dataset]['NAT']['clean']:
            clean_accs = results[dataset]['NAT']['clean']
            row['NAT_Clean'] = f"{np.mean(clean_accs):.2f}% (±{np.std(clean_accs):.2f})"
            row['NAT_Clean_runs'] = len(clean_accs)
        else:
            row['NAT_Clean'] = 'N/A'
            row['NAT_Clean_runs'] = 0

        # NAT_Adv
        if 'NAT' in results[dataset] and results[dataset]['NAT']['adv']:
            adv_accs = results[dataset]['NAT']['adv']
            row['NAT_Adv'] = f"{np.mean(adv_accs):.2f}% (±{np.std(adv_accs):.2f})"
            row['NAT_Adv_runs'] = len(adv_accs)
        else:
            row['NAT_Adv'] = 'N/A'
            row['NAT_Adv_runs'] = 0

        # AT_Clean
        if 'AT' in results[dataset] and results[dataset]['AT']['clean']:
            clean_accs = results[dataset]['AT']['clean']
            row['AT_Clean'] = f"{np.mean(clean_accs):.2f}% (±{np.std(clean_accs):.2f})"
            row['AT_Clean_runs'] = len(clean_accs)
        else:
            row['AT_Clean'] = 'N/A'
            row['AT_Clean_runs'] = 0

        # AT_Adv
        if 'AT' in results[dataset] and results[dataset]['AT']['adv']:
            adv_accs = results[dataset]['AT']['adv']
            row['AT_Adv'] = f"{np.mean(adv_accs):.2f}% (±{np.std(adv_accs):.2f})"
            row['AT_Adv_runs'] = len(adv_accs)
        else:
            row['AT_Adv'] = 'N/A'
            row['AT_Adv_runs'] = 0

        detailed_data.append(row)

    return pd.DataFrame(detailed_data)


def save_results(summary_df, detailed_df, results_dir):
    """Save results to CSV files"""
    results_path = Path(results_dir)

    # Save summary table
    summary_path = results_path / 'summary_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary table saved to: {summary_path}")

    # Save detailed table
    detailed_path = results_path / 'detailed_table.csv'
    detailed_df.to_csv(detailed_path, index=False)
    print(f"✓ Detailed table saved to: {detailed_path}")


def find_latest_results_dir():
    """Find the latest run_* directory in benchmark_results"""
    benchmark_results_dir = Path(__file__).parent.parent / 'artifacts' / 'benchmark_results'

    if not benchmark_results_dir.exists():
        return None

    # Find all run_* directories
    run_dirs = list(benchmark_results_dir.glob('run_*'))

    if not run_dirs:
        return None

    # Sort by modification time, return the latest
    latest_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return latest_dir


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('results_dir', nargs='?', default=None,
                        help='Path to benchmark results directory (default: latest run_* folder)')
    parser.add_argument('--output', '-o', help='Custom output filename prefix')
    args = parser.parse_args()

    # If no results_dir provided, find the latest
    if args.results_dir is None:
        args.results_dir = find_latest_results_dir()
        if args.results_dir is None:
            print("Error: No benchmark results found in artifacts/benchmark_results/")
            print("Please run benchmarks/auto_benchmark.py first or specify a results directory.")
            return
        print(f"Auto-detected latest results: {args.results_dir}")

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS ANALYSIS")
    print(f"{'='*80}\n")

    # Analyze results
    results = analyze_results(args.results_dir)

    if not results:
        print("No results to analyze.")
        return

    # Create summary tables
    print(f"\n{'='*80}")
    print("SUMMARY TABLE (Average Accuracy)")
    print(f"{'='*80}\n")

    summary_df = create_summary_table(results)
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("DETAILED TABLE (Mean ± Std)")
    print(f"{'='*80}\n")

    detailed_df = create_detailed_table(results)
    print(detailed_df.to_string(index=False))

    # Save results
    save_results(summary_df, detailed_df, args.results_dir)

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
