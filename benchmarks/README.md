# Automated Benchmarking

Automates training and evaluation across datasets with configurable epochs and runs.

## Quick Start

```bash
# Default: 100 epochs, 3 runs per config
python benchmarks/auto_benchmark.py

# Custom configuration
python benchmarks/auto_benchmark.py --epochs 50 --runs 5

# Analysis runs automatically after benchmark completes
# Or run manually (auto-detects latest run)
python benchmarks/analyze_benchmark.py
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Epochs per training run |
| `--runs` | 3 | Runs per configuration |
| `--datasets` | GTSRB CIFAR10 MNIST | Datasets to benchmark |
| `--model_arch` | small_cnn | Model architecture |
| `--attack_modes` | PGD FGSM | Adversarial attack modes |
| `--skip_clean_training` | - | Skip NAT training |
| `--skip_adv_training` | - | Skip AT training |

## Output

Results saved to `artifacts/benchmark_results/run_TIMESTAMP/`:
- Individual log files per run
- `summary_table.csv` - Average accuracies
- `detailed_table.csv` - Mean ± std with run counts

Example output table:

```
Expert  NAT_Clean  NAT_Adv  AT_Clean  AT_Adv
GTSRB   95.54%     3.50%    91.47%    48.81%
CIFAR   91.90%     0.54%    84.38%    44.98%
MNIST   99.46%     21.34%   99.38%    98.72%
```

## Examples

```bash
# Quick test
python benchmarks/auto_benchmark.py --epochs 20 --runs 1

# Full benchmark with 5 runs
python benchmarks/auto_benchmark.py --epochs 100 --runs 5

# Single dataset
python benchmarks/auto_benchmark.py --datasets GTSRB --skip_clean_training

# Windows batch alternative
benchmarks\auto_benchmark.bat --epochs 100 --runs 3
```

## What It Does

1. Trains each dataset with NAT (natural) and AT (adversarial training)
2. Runs multiple iterations per configuration
3. Tests with adversarial attacks (PGD/FGSM)
4. Automatically analyzes results and generates summary tables
5. Outputs mean ± std accuracies in CSV format
