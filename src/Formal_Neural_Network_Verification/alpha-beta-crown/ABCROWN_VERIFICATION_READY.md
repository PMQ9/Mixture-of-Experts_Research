# alpha-beta-CROWN Verification Setup

This project includes automated setup scripts for alpha-beta-CROWN neural network verification.

## One-Command Setup

Run this on any machine to set up alpha-beta-CROWN:

```bash
python src/Formal_Neural_Network_Verification/alpha-beta-crown/setup_abcrown.py
```

This will:
- Check Python and PyTorch versions
- Initialize git submodules
- Install dependencies (auto_LiRPA, ONNX, etc.)
- Verify installation
- Create necessary directories

## Usage

After setup, verify a trained model:

```bash
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py     --model_path artifacts/gtsrb_small_cnn_best.pth     --dataset GTSRB     --epsilon 0.00784     --num_images 10     --auto_run
```

## Requirements

- Python 3.9+ (3.11 recommended)
- PyTorch 2.0+ (install first: `pip install torch torchvision`)
- Git (for submodule initialization)

## Setup Scripts

Three setup scripts are provided:

1. **setup_abcrown.py** (recommended) - Cross-platform Python script
2. **setup_abcrown.sh** - Linux/Mac shell script
3. **setup_abcrown.bat** - Windows batch script

All scripts support:
- `--test` - Run test verification after setup
- `--skip-deps` - Skip dependency installation
- `--check-only` - Check installation status (Python script only)