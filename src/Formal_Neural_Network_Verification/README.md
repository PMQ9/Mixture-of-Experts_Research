# Formal Neural Network Verification

This directory contains tools and scripts for formally verifying both **individual experts** and the complete **MetaMoE system** using NNV (Neural Network Verification Tool).

## 📁 Directory Structure

```
Formal_Neural_Network_Verification/
├── README.md                           # This file
├── QUICKSTART.md                       # 5-minute quick start guide
├── NNV_SETUP_GUIDE.md                  # Detailed NNV setup and usage
├── MOE_VERIFICATION_GUIDE.md          # ⭐ MetaMoE compositional verification guide
│
├── export_to_onnx.py                   # Export individual experts to ONNX
├── export_router_to_onnx.py           # Export MetaMoE router to ONNX
├── test_onnx_export.py                # Validate ONNX export correctness
│
├── verify_expert_nnv.m                # Verify individual expert (Level 1)
├── verify_router_nnv.m                # Verify MetaMoE router (Level 2)
├── verify_metamoe_compositional.m     # ⭐ Full MetaMoE verification (Level 3)
├── quick_verify_example.m             # Quick test script
│
├── File_Conversion/                    # Legacy conversion scripts (not functional)
│   ├── pth_to_mat.py
│   ├── onnx_to_mat.py
│   └── check_model_architecture.py
└── sample_moe_io.py                   # Sample I/O testing
```

## 🚀 Quick Start

### 1. Install NNV (One-time setup)

```matlab
% In MATLAB, navigate to NNV and run:
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
install  % or startup_nnv for path-only setup
```

### 2. Export a Model to ONNX

```bash
# Export a trained expert to ONNX format
python export_to_onnx.py \
    --model_path ../artifacts/results/gtsrb_micro_cnn_best.pth \
    --output_dir ../artifacts/nnv_models
```

### 3. Run Verification

```matlab
% In MATLAB:
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% Quick test
quick_verify_example

% Or full verification with real dataset
verify_expert_nnv
```

## 📚 Documentation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup and first verification
- **[NNV_SETUP_GUIDE.md](NNV_SETUP_GUIDE.md)** - Comprehensive NNV guide for individual experts

### MetaMoE Verification ⭐
- **[MOE_VERIFICATION_GUIDE.md](MOE_VERIFICATION_GUIDE.md)** - Complete guide for verifying the full MetaMoE system
  - Router verification
  - Compositional verification
  - End-to-end guarantees

## 🔧 Tools Overview

### Python Scripts

#### `export_to_onnx.py`
Converts PyTorch `.pth` models to ONNX format compatible with NNV.

**Usage**:
```bash
# Single model
python export_to_onnx.py --model_path <path_to_pth> --output_dir <output_dir>

# Batch export
python export_to_onnx.py --models_dir artifacts/results --filter micro_cnn

# Custom input size
python export_to_onnx.py --model_path model.pth --input_size "1,3,32,32"
```

**Arguments**:
- `--model_path`: Path to single .pth file
- `--models_dir`: Directory containing multiple .pth files
- `--output_dir`: Output directory (default: `artifacts/nnv_models`)
- `--input_size`: Input tensor size as "B,C,H,W" (auto-detected if not provided)
- `--opset_version`: ONNX opset version (default: 11)
- `--filter`: Only export models containing this substring

### MATLAB Scripts

#### `verify_expert_nnv.m`
Comprehensive verification script with dataset integration and visualization.

**Features**:
- Loads ONNX models via NNV
- Integrates with GTSRB/CIFAR-10/MNIST datasets
- Supports multiple verification methods
- Generates output range visualizations
- Saves results to file

**Configuration** (edit top of file):
```matlab
onnx_model_path = '../../artifacts/nnv_models/gtsrb_micro_cnn_best.onnx';
dataset_name = 'GTSRB';
epsilon = 2/255;
reachMethod = 'approx-star';
test_image_idx = 1;
```

#### `quick_verify_example.m`
Minimal script for quick testing without dataset dependencies.

**Features**:
- Uses synthetic input
- Fast approximate verification
- Good for sanity checks
- No dataset required

## 🎯 Verification Methods

| Method | Type | Speed | Precision | Use Case |
|--------|------|-------|-----------|----------|
| **exact-star** | Sound & Complete | 🐌 Slow | ✓✓✓ Exact | Critical verification, small models |
| **approx-star** | Sound (over-approx) | 🚀 Fast | ✓✓ Good | General purpose, quick checks |
| **abs-dom** | Abstract Domains | ⚡ Very Fast | ✓ Coarse | Initial screening |

## 📊 Recommended Architectures for Verification

| Architecture | Parameters | Verification | Recommended |
|--------------|------------|--------------|-------------|
| **MicroExpertCNN** | ~67K | Fast (minutes) | ✅ Best for verification |
| **TinyExpertCNN** | ~620K | Moderate (hours) | ⚠️ Use approx methods |
| **SmallExpertCNN** | ~1.5M | Slow (days) | ❌ Challenging |
| ConvNeXt-Tiny | ~28M | Very slow | ❌ Not practical |

**Recommendation**: Use **micro_cnn** for verification tasks.

## 🔬 Verification Workflow

```
┌─────────────────┐
│  Train Expert   │  train_moe.py --model_arch micro_cnn
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Export to ONNX  │  export_to_onnx.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Load in NNV    │  onnx2nnv() in MATLAB
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Define Input Set│  ImageStar(lb, ub)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verify Robust   │  net.verify_robustness()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Results │  Certified robust / Not robust / Unknown
└─────────────────┘
```

## 🧪 Example: Complete Verification

### Train an Adversarially Robust Model
```bash
python src/Vision_Transformer_Pytorch/train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES \
    --trades_beta 6.0
```

### Export to ONNX
```bash
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best_robust.pth \
    --output_dir artifacts/nnv_models
```

### Verify in MATLAB
```matlab
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% Edit verify_expert_nnv.m parameters, then run:
verify_expert_nnv
```

## 📈 Understanding Results

### Verification Outcomes

| Result | Interpretation | Next Steps |
|--------|----------------|------------|
| ✅ **Verified Robust** (`res = 1`) | All perturbed inputs correctly classified | Increase epsilon to find limits |
| ❌ **Not Robust** (`res = 0`) | Found adversarial example | Train with adversarial training |
| ❓ **Unknown** (`res = -1`) | Timeout or approximation | Try exact method or smaller epsilon |

### Output Visualization

The verification script generates plots showing:
- **Reachable output ranges** for each class
- **Original prediction** (red X)
- **True class** (green line)

**Robust if**: True class has highest *minimum* output value across all inputs in the set.

## ⚙️ Common Parameters

### Epsilon Values (L∞ perturbation)
```matlab
epsilon = 1/255;    % 0.004 - Very robust (tight bound)
epsilon = 2/255;    % 0.008 - Standard for MNIST
epsilon = 8/255;    % 0.031 - Standard for CIFAR/GTSRB
epsilon = 16/255;   % 0.063 - Challenging (loose bound)
```

### Dataset Normalization
```matlab
% GTSRB
mean = [0.3337, 0.3064, 0.3171];
std = [0.2672, 0.2564, 0.2629];

% CIFAR-10
mean = [0.4914, 0.4822, 0.4465];
std = [0.2023, 0.1994, 0.2010];

% MNIST
mean = 0.1307;
std = 0.3081;
```

## 🐛 Troubleshooting

### Problem: "Undefined function 'onnx2nnv'"
**Solution**: Run `startup_nnv` in MATLAB from NNV directory.

### Problem: ONNX import fails
**Solution**:
- Use simpler architecture (micro_cnn)
- Try different ONNX opset: `--opset_version 9`
- Check model with: `netron model.onnx`

### Problem: Out of memory
**Solution**:
- Use `'approx-star'` instead of `'exact-star'`
- Reduce epsilon
- Reduce model size

### Problem: Takes too long
**Solution**:
- Use approximate methods
- Set timeout: `reachOptions.timeout = 300`
- Verify on fewer images

## 📖 Additional Resources

### NNV Resources
- **GitHub**: https://github.com/verivital/nnv
- **Tutorial**: `modules/nnv_moe/code/nnv/examples/Tutorial/`
- **Paper**: [NNV 2.0 (CAV 2023)](https://link.springer.com/chapter/10.1007/978-3-031-37703-7_19)

### Related Tools
- **α,β-CROWN**: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **ERAN**: https://github.com/eth-sri/eran
- **Marabou**: https://github.com/NeuralNetworkVerification/Marabou

### Example Scripts
- MNIST verification: `modules/nnv_moe/code/nnv/examples/Tutorial/NN/MNIST/verify.m`
- GTSRB verification: `modules/nnv_moe/code/nnv/examples/Tutorial/NN/GTSRB/verify_robust_1.m`

## 🤝 Contributing

When adding new verification scripts:
1. Document parameters clearly
2. Include example usage
3. Add error handling
4. Generate visualizations
5. Update this README

## 🎯 Verification Levels

| Level | What | Scripts | Use Case |
|-------|------|---------|----------|
| **Level 1** ⭐ | Individual Expert | `verify_expert_nnv.m` | Component testing, expert robustness |
| **Level 2** ⭐⭐ | Router Only | `verify_router_nnv.m` | Routing stability, misrouting detection |
| **Level 3** ⭐⭐⭐ | **Full MetaMoE** | `verify_metamoe_compositional.m` | **End-to-end certification** |

See **[MOE_VERIFICATION_GUIDE.md](MOE_VERIFICATION_GUIDE.md)** for compositional verification details.

## 📝 Notes

- **File_Conversion/** contains legacy scripts for direct PyTorch→MATLAB conversion. These are not fully functional. Use ONNX workflow instead.
- For **complete MetaMoE verification**, use compositional approach (Level 3) - verifies both router and experts together.
- NNV supports many layer types, but custom/complex layers may require manual handling.

## 📧 Support

For questions:
- NNV issues: https://github.com/verivital/nnv/issues
- Project issues: See main repository README
