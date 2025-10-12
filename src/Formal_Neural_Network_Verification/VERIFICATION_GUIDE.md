# Formal Verification Guide: MetaMoE System

**Complete guide for formally verifying Mixture-of-Experts models using NNV**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup: NNV Installation](#setup-nnv-installation)
3. [Training for Verification](#training-for-verification)
4. [Export to ONNX](#export-to-onnx)
5. [Verification Levels](#verification-levels)
6. [Parameters & Configuration](#parameters--configuration)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

**5 steps to verify your first model:**

```bash
# 1. Train micro expert (10-20 min)
python train_moe.py --dataset GTSRB --model_arch micro_cnn --epochs 50

# 2. Export to ONNX (5 sec)
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best.pth \
    --output_dir artifacts/nnv_models
```

```matlab
% 3. Install NNV in MATLAB (one-time)
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv

% 4. Run verification
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')
verify_expert_nnv  % Edit config section first

% Expected: ✓ VERIFIED ROBUST or ✗ NOT ROBUST
```

---

## Setup: NNV Installation

### Prerequisites
- **MATLAB** R2023a+ with Deep Learning Toolbox
- **Python** 3.10+ with PyTorch
- **ONNX Support**: Install via MATLAB Add-On Manager (search "ONNX")

### Install NNV

```matlab
% Navigate to NNV
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')

% Option 1: Full install (first time)
install

% Option 2: Path-only (faster, recommended)
startup_nnv

% Test installation
help onnx2nnv  % Should display help text
```

---

## Training for Verification

### Recommended Architecture

Use **micro_cnn** (~67K params) for fast verification:

| Model | Params | Verification Time | Robustness |
|-------|--------|-------------------|------------|
| **micro_cnn** ✅ | 67K | Minutes | Good |
| tiny_cnn | 620K | Hours | Better |
| small_cnn | 1.5M | Days | Best |

### Training Commands

**Individual Expert (Standard)**:
```bash
python train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --batch_size 128
```

**Individual Expert (Adversarially Robust)**:
```bash
python train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES \
    --trades_beta 6.0
```

**MetaMoE System**:
```bash
# Train experts first (GTSRB, CIFAR10, MNIST), then:
python train_moe.py \
    --meta_moe \
    --gtsrb_model_path artifacts/results/gtsrb_micro_cnn_best.pth \
    --cifar10_model_path artifacts/results/cifar10_micro_cnn_best.pth \
    --mnist_model_path artifacts/results/mnist_micro_cnn_best.pth \
    --epochs 100 \
    --meta_top_k 1 \
    --adv_gating_train \
    --at_mode TRADES
```

---

## Export to ONNX

### Export Expert Models

```bash
# Single model
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best.pth \
    --output_dir artifacts/nnv_models

# Batch export (all micro_cnn models)
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --models_dir artifacts/results \
    --filter micro_cnn \
    --output_dir artifacts/nnv_models
```

### Export Router (MetaMoE)

```bash
python src/Formal_Neural_Network_Verification/export_router_to_onnx.py \
    --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
    --output_dir artifacts/nnv_models \
    --output_logits  # Recommended for better verification
```

### Validate Export

```bash
python src/Formal_Neural_Network_Verification/test_onnx_export.py \
    --pth_model artifacts/results/gtsrb_micro_cnn_best.pth \
    --onnx_model artifacts/nnv_models/gtsrb_micro_cnn_best.onnx

# Expected: ✓ ALL TESTS PASSED
```

---

## Verification Levels

### Level 1: Individual Expert ⭐

**Verifies**: Expert robustness on its designated dataset
**Assumes**: Perfect routing (router is correct)
**Use case**: Component testing, understanding expert limits

```matlab
% Edit verify_expert_nnv.m configuration:
onnx_model_path = '..\..\artifacts\nnv_models\gtsrb_micro_cnn_best.onnx';
dataset_name = 'GTSRB';
epsilon = 2/255;
reachMethod = 'approx-star';
test_image_idx = 1;

% Run
verify_expert_nnv
```

**Output**: `res = 1` (robust), `0` (not robust), `-1` (unknown)

---

### Level 2: Router Only ⭐⭐

**Verifies**: Router maintains correct routing under perturbations
**Assumes**: Expert will classify correctly
**Use case**: Detecting misrouting attacks

```matlab
% Edit verify_router_nnv.m configuration:
router_onnx_path = '..\..\artifacts\nnv_models\meta_moe_router_logits.onnx';
dataset_name = 'GTSRB';
epsilon = 8/255;  % Can use larger epsilon for router
num_test_images = 10;

% Run
verify_router_nnv
```

**Output**: Certified routing accuracy (% of images with stable routing)

---

### Level 3: Compositional MetaMoE ⭐⭐⭐ (Strongest)

**Verifies**: Complete end-to-end robustness
**Guarantees**: Both router AND expert are robust
**Use case**: Safety-critical deployment, certification

```matlab
% Edit verify_metamoe_compositional.m configuration:
router_onnx = '..\..\artifacts\nnv_models\meta_moe_router_logits.onnx';
expert_onnx = '..\..\artifacts\nnv_models\gtsrb_micro_cnn_best.onnx';
dataset_name = 'GTSRB';
epsilon = 2/255;
num_test_images = 50;

% Run
verify_metamoe_compositional
```

**Output**:
```
Certified MetaMoE Accuracy: 65%
  Router robust: 85%
  Expert robust: 70%
  Both robust: 65%
```

**Logic**: `Router_robust ∧ Expert_robust ⟹ MetaMoE_robust`

---

## Parameters & Configuration

### Epsilon (ε) - Perturbation Bound

```matlab
% L-infinity norm: max per-pixel change
epsilon = 1/255;    % 0.004 - Very tight (high security)
epsilon = 2/255;    % 0.008 - Standard (MNIST)
epsilon = 8/255;    % 0.031 - Standard (CIFAR/GTSRB)
epsilon = 16/255;   % 0.063 - Challenging (stress test)
```

**Recommendation**:
- **Experts**: 1-2/255 (tighter)
- **Router**: 4-8/255 (can be looser)

### Verification Methods

```matlab
% Fast approximate (over-approximates)
reachOptions.reachMethod = 'approx-star';  % Recommended

% Slow but exact
reachOptions.reachMethod = 'exact-star';   % Small models only

% Very fast, coarse
reachOptions.reachMethod = 'abs-dom';      % Initial screening
```

| Method | Speed | Precision | Use When |
|--------|-------|-----------|----------|
| `approx-star` | Fast | Good | General use ✅ |
| `exact-star` | Slow | Exact | Critical verification |
| `abs-dom` | Very Fast | Coarse | Quick check |

### Dataset Normalization

```matlab
% GTSRB
meanNorm = [0.3337, 0.3064, 0.3171];
stdNorm = [0.2672, 0.2564, 0.2629];

% CIFAR-10
meanNorm = [0.4914, 0.4822, 0.4465];
stdNorm = [0.2023, 0.1994, 0.2010];

% MNIST
meanNorm = 0.1307;
stdNorm = 0.3081;
```

---

## Interpreting Results

### Expert Verification

| Result | Meaning | Next Step |
|--------|---------|-----------|
| `res = 1` ✅ | **Verified robust** - All inputs in ε-ball correctly classified | Increase epsilon |
| `res = 0` ❌ | **Not robust** - Found adversarial example | Train with `--adv_training` |
| `res = -1` ❓ | **Unknown** - Timeout/approximation limit | Use exact method or smaller ε |

### Router Verification

**Certified Routing Accuracy**: % of images maintaining correct routing

| Accuracy | Assessment | Action |
|----------|------------|--------|
| >85% | Good | Router is robust |
| 60-85% | Moderate | Consider `--adv_gating_train` |
| <60% | Poor | Retrain with adversarial gating |

### Compositional Verification (MetaMoE)

```
========================================
COMPOSITIONAL VERIFICATION SUMMARY
========================================
Images tested: 50
----------------------------------------
Router robust: 42 (84%)
Expert robust: 35 (70%)
Both robust: 32 (64%)  ← Certified MetaMoE Accuracy
========================================
```

**Interpreting**:
- **Certified MetaMoE Accuracy = 64%**: 32 out of 50 images verified end-to-end
- **Router > Expert**: Expert is bottleneck → Train experts with `--adv_training`
- **Expert > Router**: Router is bottleneck → Train MetaMoE with `--adv_gating_train`

**Expected Performance**:

| Training | Epsilon | Certified Acc |
|----------|---------|---------------|
| Standard | 2/255 | 15-30% |
| Adversarial Experts | 2/255 | 40-60% |
| Adversarial Both | 2/255 | 60-80% |

---

## Troubleshooting

### "Undefined function 'onnx2nnv'"

```matlab
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
```

### ONNX Export Fails

- Use simpler architecture: `--model_arch micro_cnn`
- Try different opset: `--opset_version 9` or `13`
- Visualize model: `netron model.onnx`

### Verification Takes Forever

```matlab
% Use approximate method
reachOptions.reachMethod = 'approx-star';

% Set timeout (5 minutes)
reachOptions.timeout = 300;

% Test fewer images
num_test_images = 10;
```

### Out of Memory

1. Use approximate method: `'approx-star'`
2. Reduce epsilon: `1/255` instead of `8/255`
3. Use smaller model: `micro_cnn` instead of `small_cnn`
4. Enable parallel computing:
   ```matlab
   reachOptions.numCores = 4;
   ```

### Low Certified Accuracy

**Checklist**:
1. Is clean accuracy >90%? If not, train longer
2. Are models adversarially trained? Use `--adv_training`
3. Is epsilon too large? Try smaller values (1-2/255)
4. For MetaMoE: Which component fails more?
   - Router: Use `--adv_gating_train`
   - Expert: Use `--adv_training` on experts

---

## Complete Workflow Example

### End-to-End: Train → Export → Verify MetaMoE

```bash
# ========== STEP 1: TRAIN EXPERTS ==========
# GTSRB
python train_moe.py --dataset GTSRB --model_arch micro_cnn \
    --epochs 50 --adv_training --at_mode TRADES

# CIFAR-10
python train_moe.py --dataset CIFAR10 --model_arch micro_cnn \
    --epochs 50 --adv_training --at_mode TRADES

# MNIST
python train_moe.py --dataset MNIST --model_arch micro_cnn \
    --epochs 50 --adv_training --at_mode TRADES

# ========== STEP 2: TRAIN METAMOE ==========
python train_moe.py --meta_moe \
    --gtsrb_model_path artifacts/results/gtsrb_micro_cnn_best_robust.pth \
    --cifar10_model_path artifacts/results/cifar10_micro_cnn_best_robust.pth \
    --mnist_model_path artifacts/results/mnist_micro_cnn_best_robust.pth \
    --epochs 100 --adv_gating_train --at_mode TRADES

# ========== STEP 3: EXPORT ==========
# Export experts
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --models_dir artifacts/results --filter micro_cnn \
    --output_dir artifacts/nnv_models

# Export router
python src/Formal_Neural_Network_Verification/export_router_to_onnx.py \
    --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
    --output_dir artifacts/nnv_models --output_logits
```

```matlab
% ========== STEP 4: VERIFY IN MATLAB ==========
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% Edit verify_metamoe_compositional.m with paths and parameters

% Run compositional verification
verify_metamoe_compositional

% Expected output:
% Certified MetaMoE Accuracy: 65-75% (with adversarial training)
```

---

## Best Practices

### 1. Start Small
- Use **micro_cnn** architecture
- Test on **1-10 images** initially
- Use **small epsilon** (1-2/255)
- Use **approx-star** method

### 2. Train for Verification
Models that verify better:
- Use adversarial training: `--adv_training`
- Use TRADES loss: `--at_mode TRADES --trades_beta 6.0`
- For MetaMoE: `--adv_gating_train`

### 3. Verify Incrementally
```
1. Verify individual experts → Identify weak experts
2. Verify router → Check routing stability
3. Compositional verification → End-to-end guarantee
4. Identify bottleneck → Retrain weak component
5. Re-verify → Iterate
```

### 4. Epsilon Selection Strategy
```
Start: ε = 1/255 (should verify)
      ↓
Increase to ε = 2/255
      ↓
Increase to ε = 4/255
      ↓
Find limit where verification fails
```

### 5. Batch Verification
For certified accuracy on dataset:
```matlab
num_test_images = 100;  % Test on 100 images
% ... run verification
certified_accuracy = sum(results == 1) / num_test_images;
fprintf('Certified Robust Accuracy: %.2f%%\n', certified_accuracy * 100);
```

---

## Key Scripts Reference

| Script | Purpose | Level |
|--------|---------|-------|
| `export_to_onnx.py` | Export experts to ONNX | - |
| `export_router_to_onnx.py` | Export MetaMoE router | - |
| `test_onnx_export.py` | Validate ONNX correctness | - |
| `verify_expert_nnv.m` | Verify individual expert | 1 ⭐ |
| `verify_router_nnv.m` | Verify router stability | 2 ⭐⭐ |
| `verify_metamoe_compositional.m` | **Full MetaMoE verification** | 3 ⭐⭐⭐ |
| `quick_verify_example.m` | Quick sanity check | - |

---

## Additional Resources

### NNV Documentation
- GitHub: https://github.com/verivital/nnv
- Examples: `modules/nnv_moe/code/nnv/examples/Tutorial/`
- Paper: [NNV 2.0 (CAV 2023)](https://link.springer.com/chapter/10.1007/978-3-031-37703-7_19)

### Alternative Tools
- **α,β-CROWN**: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **ERAN**: https://github.com/eth-sri/eran
- **Marabou**: https://github.com/NeuralNetworkVerification/Marabou

---

## Summary

**Verification Workflow**:
```
Train → Export → Verify → Analyze → Iterate
```

**Three Levels**:
1. **Level 1** - Expert only (assumes perfect routing)
2. **Level 2** - Router only (assumes expert correct)
3. **Level 3** - Compositional (strongest guarantee)

**Key Takeaways**:
- Use **micro_cnn** for practical verification
- Train with **adversarial training** for better robustness
- Start with **small epsilon**, increase gradually
- **Compositional verification** gives end-to-end guarantees
- Identify **bottlenecks** and iterate

**Ready to verify?**
```matlab
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')
verify_expert_nnv  % Start here!
```
