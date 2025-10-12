# MetaMoE Formal Verification Guide

This guide explains how to formally verify your complete Mixture-of-Experts (MetaMoE) system using compositional verification with NNV.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Verification Levels](#verification-levels)
3. [Complete Workflow](#complete-workflow)
4. [Router Verification](#router-verification)
5. [Expert Verification](#expert-verification)
6. [Compositional Verification](#compositional-verification)
7. [Interpreting Results](#interpreting-results)
8. [Best Practices](#best-practices)

---

## Overview

### What is Compositional Verification?

Your MetaMoE system has two main components:

```
┌─────────────┐
│   Input x   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Router             │  Routes to correct expert
│  (MetaGatingNet)    │  Expert 0: GTSRB
└──────┬──────────────┘  Expert 1: CIFAR-10
       │                 Expert 2: MNIST
       ▼
┌─────────────────────┐
│  Selected Expert    │  Classifies within dataset
│  (MicroExpertCNN)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│   Output y  │
└─────────────┘
```

**Compositional verification** verifies each component separately, then combines the results to guarantee end-to-end robustness.

### Why Compositional?

**Problem**: Verifying the entire MetaMoE monolithically is:
- ❌ Extremely complex (dynamic routing)
- ❌ Hard to express in ONNX
- ❌ Computationally intractable

**Solution**: Compositional reasoning:
- ✅ Verify router independently
- ✅ Verify experts independently
- ✅ Combine with logic: `Router_robust ∧ Expert_robust ⟹ MetaMoE_robust`

---

## Verification Levels

### Level 1: Expert-Only Verification ⭐

**What it verifies**: Individual experts are robust on their designated datasets

**Guarantees**:
- IF correctly routed to expert
- THEN classification is robust

**Limitations**:
- Assumes perfect routing
- No guarantee on router behavior

**Use case**: Understanding expert robustness, component testing

**Scripts**: `verify_expert_nnv.m`

---

### Level 2: Router-Only Verification ⭐⭐

**What it verifies**: Router maintains correct routing under perturbations

**Guarantees**:
- For input x from dataset D
- All x' in ε-ball still routed to expert for D

**Limitations**:
- Assumes expert will classify correctly
- No guarantee on expert behavior

**Use case**: Detecting potential misrouting attacks

**Scripts**: `verify_router_nnv.m`

---

### Level 3: Compositional Verification ⭐⭐⭐ **(Strongest)**

**What it verifies**: Complete end-to-end MetaMoE robustness

**Guarantees**:
- Router stays stable: `Router(x') = correct_expert`
- Expert stays correct: `Expert(x') = correct_class`
- Combined: `MetaMoE(x') = correct_class`

**Limitations**:
- More computationally expensive (2x verification time)
- Conservative (both must pass)

**Use case**: Safety-critical deployment, certification

**Scripts**: `verify_metamoe_compositional.m`

---

## Complete Workflow

### Prerequisites

1. **Trained MetaMoE model** with frozen experts
2. **NNV installed** in MATLAB
3. **Dataset** properly structured

### Step-by-Step Process

```bash
# ========== STEP 1: Train Individual Experts ==========
# Train each expert separately (preferably with adversarial training)

# GTSRB expert
python train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES

# CIFAR-10 expert
python train_moe.py \
    --dataset CIFAR10 \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES

# MNIST expert
python train_moe.py \
    --dataset MNIST \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES

# ========== STEP 2: Train MetaMoE ==========
# Train router with adversarial gating (recommended)

python train_moe.py \
    --meta_moe \
    --gtsrb_model_path artifacts/results/gtsrb_micro_cnn_best_robust.pth \
    --cifar10_model_path artifacts/results/cifar10_micro_cnn_best_robust.pth \
    --mnist_model_path artifacts/results/mnist_micro_cnn_best_robust.pth \
    --epochs 100 \
    --meta_top_k 1 \
    --adv_gating_train \
    --at_mode TRADES

# ========== STEP 3: Export Models to ONNX ==========

# Export experts
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --models_dir artifacts/results \
    --filter micro_cnn \
    --output_dir artifacts/nnv_models

# Export router
python src/Formal_Neural_Network_Verification/export_router_to_onnx.py \
    --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
    --output_dir artifacts/nnv_models \
    --output_logits

# ========== STEP 4: Verify in MATLAB ==========
# See MATLAB sections below
```

---

## Router Verification

### Purpose

Verify that the router correctly identifies the dataset under input perturbations.

### Export Router

```bash
# Display router info first (optional)
python src/Formal_Neural_Network_Verification/export_router_to_onnx.py \
    --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
    --info_only

# Export router (outputs logits for better verification)
python src/Formal_Neural_Network_Verification/export_router_to_onnx.py \
    --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
    --output_dir artifacts/nnv_models \
    --output_logits
```

**Output**: `meta_moe_small_cnn_best_router_logits.onnx`

### Verify Router in MATLAB

```matlab
% Open MATLAB
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% Edit verify_router_nnv.m parameters:
% - router_onnx_path: Path to router ONNX
% - dataset_name: 'GTSRB', 'CIFAR10', or 'MNIST'
% - epsilon: Perturbation bound (e.g., 8/255)
% - num_test_images: Number of images to test

% Run verification
verify_router_nnv
```

### Expected Results

**Good router** (ε = 8/255):
- Certified routing accuracy: 80-95%
- Most images maintain correct routing

**Poor router** (ε = 8/255):
- Certified routing accuracy: <50%
- Misrouting under perturbations

### Key Metrics

- **Certified Routing Accuracy**: % of images with verified stable routing
- **Average verification time**: Time per image
- **Output ranges**: Separation between expert logits

---

## Expert Verification

### Purpose

Verify that each expert correctly classifies inputs from its designated dataset under perturbations.

### Export Expert

```bash
# Export GTSRB expert
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best.pth \
    --output_dir artifacts/nnv_models
```

### Verify Expert in MATLAB

```matlab
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% Edit verify_expert_nnv.m parameters:
% - onnx_model_path: Path to expert ONNX
% - dataset_name: 'GTSRB'
% - epsilon: 2/255 (smaller for experts)
% - test_image_idx: Which image to test

% Run verification
verify_expert_nnv
```

### Expected Results

**Good expert** with adversarial training (ε = 2/255):
- Verification result: Robust (res = 1)
- Clean prediction matches true class
- Output ranges well-separated

**Standard expert** without adversarial training (ε = 2/255):
- Verification result: Not robust (res = 0)
- May find adversarial examples

### Key Metrics

- **Verification result**: Robust / Not Robust / Unknown
- **Output separation**: Gap between true class and others
- **Verification time**: Usually 1-5 minutes per image

---

## Compositional Verification

### Purpose

Verify complete end-to-end MetaMoE robustness by combining router and expert verification.

### Requirements

1. **Router ONNX**: Exported with `--output_logits`
2. **Expert ONNX**: For the dataset being tested
3. **Both models trained** (preferably with adversarial training)

### Run Compositional Verification

```matlab
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% Edit verify_metamoe_compositional.m parameters:
% - router_onnx: Path to router
% - expert_onnx: Path to expert for this dataset
% - dataset_name: 'GTSRB', 'CIFAR10', or 'MNIST'
% - epsilon: Perturbation bound (e.g., 2/255)
% - num_test_images: How many to test (e.g., 10-100)

% Run compositional verification
verify_metamoe_compositional
```

### What It Does

For each test image:

1. ✓ **Check clean predictions**
   - Router predicts correct expert
   - Expert predicts correct class
   - Skip if either is wrong

2. ✓ **Verify router stability**
   - Creates ε-ball around input
   - Verifies router keeps same routing decision

3. ✓ **Verify expert robustness**
   - Same ε-ball
   - Verifies expert keeps same classification

4. ✓ **Combine results**
   - **Both robust** → End-to-end robust ✅
   - **Router only** → Expert is bottleneck ⚠️
   - **Expert only** → Router is bottleneck ⚠️
   - **Neither** → Both need improvement ❌

---

## Interpreting Results

### Compositional Verification Output

```
========================================
COMPOSITIONAL VERIFICATION SUMMARY
========================================
Dataset: GTSRB
Epsilon: 0.0078
Method: approx-star
Images tested: 10
----------------------------------------
Component Analysis:
  Router robust: 8 (80.0%)
  Expert robust: 7 (70.0%)
----------------------------------------
Combined Analysis:
  ✓ Both robust (END-TO-END): 6 (60.0%)
  ⚠ Router only: 2 (20.0%)
  ⚠ Expert only: 1 (10.0%)
  ✗ Neither robust: 1 (10.0%)
========================================

🎯 Certified MetaMoE Accuracy: 60.00%
```

### What This Means

**Certified MetaMoE Accuracy: 60%**
- Out of 10 images with correct clean predictions
- 6 are verified robust end-to-end
- For these 6, **all** inputs in ε-ball are correctly classified by MetaMoE

**Router robust: 80%**
- 8 images maintain correct routing under perturbations

**Expert robust: 70%**
- 7 images maintain correct classification by expert

**Router only: 20% (2 images)**
- Router is stable but expert fails
- **Bottleneck**: Expert needs improvement
- **Action**: Train expert with adversarial training

**Expert only: 10% (1 image)**
- Expert is robust but router fails
- **Bottleneck**: Router needs improvement
- **Action**: Train router with adversarial gating

### Performance Benchmarks

| Configuration | Epsilon | Certified Accuracy | Notes |
|---------------|---------|-------------------|-------|
| Standard training | 2/255 | 10-30% | Poor robustness |
| Adversarial experts | 2/255 | 40-60% | Expert bottleneck removed |
| Adversarial router + experts | 2/255 | 60-80% | Best performance |
| Adversarial + larger models | 2/255 | 70-90% | Slower verification |

### Failure Mode Analysis

1. **Expert is the bottleneck** (Router robust > Expert robust)
   ```
   Router only: High
   Expert only: Low
   ```
   **Fix**:
   - Train experts with `--adv_training`
   - Use TRADES: `--at_mode TRADES --trades_beta 6.0`
   - Increase expert capacity (tiny_cnn instead of micro_cnn)

2. **Router is the bottleneck** (Expert robust > Router robust)
   ```
   Router only: Low
   Expert only: High
   ```
   **Fix**:
   - Train MetaMoE with `--adv_gating_train`
   - Increase routing temperature
   - Use more robust backbone for router

3. **Both weak** (Neither robust: High)
   ```
   Both router and expert fail
   ```
   **Fix**:
   - Train both with adversarial training
   - Reduce epsilon (verify at smaller perturbations)
   - Check if clean accuracy is high first

---

## Best Practices

### 1. Start with Individual Components

Before compositional verification:
- ✅ Verify experts independently
- ✅ Verify router independently
- ✅ Ensure clean accuracy is high (>90%)

### 2. Use Adversarial Training

For verifiable models:
```bash
# Experts
python train_moe.py --dataset GTSRB --model_arch micro_cnn \
    --adv_training --at_mode TRADES --trades_beta 6.0

# MetaMoE router
python train_moe.py --meta_moe \
    --adv_gating_train --at_mode TRADES --trades_beta 6.0
```

### 3. Choose Appropriate Epsilon

| Epsilon | Strength | Use Case |
|---------|----------|----------|
| 1/255 | Very tight | High-security applications |
| 2/255 | Standard | Typical robustness testing |
| 4/255 | Moderate | Challenging test |
| 8/255 | Loose | Stress testing, routers |

**Router** can handle larger ε (4-8/255) than **experts** (1-2/255)

### 4. Use Appropriate Architecture

| Model | Params | Verification | Robustness |
|-------|--------|--------------|------------|
| **micro_cnn** | 67K | ⚡ Fast (minutes) | ⭐⭐ Good |
| **tiny_cnn** | 620K | 🐌 Slow (hours) | ⭐⭐⭐ Better |
| small_cnn | 1.5M | ❌ Very slow | ⭐⭐⭐ Best |

**Recommendation**: Use **micro_cnn** for verification tasks.

### 5. Verification Method Selection

```matlab
% Fast initial screening
reachMethod = 'approx-star';  % Over-approximate, fast

% Critical verification
reachMethod = 'exact-star';   % Sound & complete, slow
```

### 6. Batch Verification

For certified accuracy:
```matlab
num_test_images = 100;  % Test on 100 images
% Run verify_metamoe_compositional.m
```

Calculate:
```
Certified Robust Accuracy = (Number verified robust) / (Total tested)
```

### 7. Iterative Improvement

```
1. Train baseline MetaMoE
2. Run compositional verification
3. Identify bottleneck (router vs expert)
4. Retrain weak component with adversarial training
5. Re-verify
6. Repeat until desired certified accuracy achieved
```

---

## Example: Complete Verification Workflow

### Scenario
Verify GTSRB expert in MetaMoE system at ε = 2/255

```bash
# ========== TRAINING ==========

# 1. Train GTSRB expert with adversarial training
python train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES \
    --trades_beta 6.0

# 2. Train CIFAR-10 and MNIST experts similarly
# ... (repeat for other datasets)

# 3. Train MetaMoE with adversarial gating
python train_moe.py \
    --meta_moe \
    --gtsrb_model_path artifacts/results/gtsrb_micro_cnn_best_robust.pth \
    --cifar10_model_path artifacts/results/cifar10_micro_cnn_best_robust.pth \
    --mnist_model_path artifacts/results/mnist_micro_cnn_best_robust.pth \
    --epochs 100 \
    --adv_gating_train \
    --at_mode TRADES

# ========== EXPORT ==========

# 4. Export GTSRB expert
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best_robust.pth \
    --output_dir artifacts/nnv_models

# 5. Export router
python src/Formal_Neural_Network_Verification/export_router_to_onnx.py \
    --meta_moe_path artifacts/results/meta_moe_small_cnn_best.pth \
    --output_dir artifacts/nnv_models \
    --output_logits
```

```matlab
% ========== VERIFY IN MATLAB ==========

% 6. Initialize NNV
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% 7. Edit verify_metamoe_compositional.m
% Set:
%   router_onnx = '...\meta_moe_small_cnn_best_router_logits.onnx'
%   expert_onnx = '...\gtsrb_micro_cnn_best_robust.onnx'
%   dataset_name = 'GTSRB'
%   epsilon = 2/255
%   num_test_images = 50

% 8. Run compositional verification
verify_metamoe_compositional

% 9. Analyze results
% Check:
%   - Certified MetaMoE Accuracy
%   - Bottleneck analysis
%   - Visualization plots
```

### Expected Results

With adversarial training:
```
Certified MetaMoE Accuracy: 65-75%
Router robust: 85-90%
Expert robust: 70-80%
```

Without adversarial training:
```
Certified MetaMoE Accuracy: 15-30%
Router robust: 60-70%
Expert robust: 20-40%
```

---

## Troubleshooting

### Problem: Low certified accuracy

**Check**:
1. Is clean accuracy high (>90%)? If not, train longer
2. Is router robust? If not, use `--adv_gating_train`
3. Are experts robust? If not, use `--adv_training`
4. Is epsilon too large? Try smaller values

### Problem: Router verification takes too long

**Solutions**:
- Use `'approx-star'` instead of `'exact-star'`
- Test fewer images initially
- Use smaller router backbone (though this may hurt accuracy)

### Problem: Router exported but verification fails

**Check**:
- Is `--output_logits` used? (Recommended)
- Are meta-class labels correct in CSV files?
- Does router output size match number of experts?

### Problem: "Both robust" is zero

**Likely causes**:
1. Models not trained with adversarial training
2. Epsilon too large
3. Clean predictions are incorrect (check dataset labels)

**Fix**:
- Start with small epsilon (1/255)
- Ensure adversarial training for both router and experts
- Verify clean accuracy first

---

## Additional Resources

### Scripts Reference

| Script | Purpose | Level |
|--------|---------|-------|
| `export_to_onnx.py` | Export experts | - |
| `export_router_to_onnx.py` | Export router | - |
| `verify_expert_nnv.m` | Verify expert | 1 ⭐ |
| `verify_router_nnv.m` | Verify router | 2 ⭐⭐ |
| `verify_metamoe_compositional.m` | Full verification | 3 ⭐⭐⭐ |

### Related Guides

- [QUICKSTART.md](QUICKSTART.md) - Quick setup
- [NNV_SETUP_GUIDE.md](NNV_SETUP_GUIDE.md) - Detailed NNV guide
- [README.md](README.md) - Tool reference

### Papers

- **NNV**: [CAV 2023](https://link.springer.com/chapter/10.1007/978-3-031-37703-7_19)
- **Compositional Verification**: [FORMATS 2022](https://link.springer.com/chapter/10.1007/978-3-031-15839-1_3)

---

## Summary

🎯 **Key Takeaways**:

1. **Compositional verification** = Router verification + Expert verification
2. **Use adversarial training** for both router and experts
3. **Start small**: Test on few images, small epsilon
4. **Identify bottlenecks**: Which component fails most?
5. **Iterate**: Train weak component, re-verify, repeat

✅ **Strongest guarantee**: `verify_metamoe_compositional.m` with both components trained adversarially

🚀 **Ready to verify your MetaMoE? Start with compositional verification!**
