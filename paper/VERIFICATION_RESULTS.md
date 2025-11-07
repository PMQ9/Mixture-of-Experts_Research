# Alpha-Beta-CROWN Verification Results

This document tracks the formal verification results for the MoE architecture using alpha-beta-CROWN.

**Paper**: Formal Verification of Compositional Robustness and Scalability of Heterogeneous Mixture-of-Experts
**Date Started**: 2025-11-06

---

## Models to Verify

| Model | Type | Training | Dataset | Path | Status |
|-------|------|----------|---------|------|--------|
| E_0_CNN_NAT | Expert | Non-Robust | CIFAR-10 | E_0_CNN_NAT/cifar10_ultra_verifiable_cnn_best_og.pth | Pending |
| E_0_CNN_AT | Expert | Robust | CIFAR-10 | E_0_CNN_AT/cifar10_ultra_verifiable_cnn_best_robust.pth | Pending |
| E_1_CNN_NAT | Expert | Non-Robust | MNIST | E_1_CNN_NAT/mnist_ultra_verifiable_cnn_best_og.pth | Pending |
| E_1_CNN_AT | Expert | Robust | MNIST | E_1_CNN_AT/mnist_ultra_verifiable_cnn_best_robust.pth | Pending |
| MoE_CNN_NAT | Router | Non-Robust | Mixed | MoE_CNN_NAT/meta_moe_ultra_verifiable_cnn_best_og.pth | Pending |
| MoE_CNN_AT | Router | Robust | Mixed | MoE_CNN_AT/meta_moe_ultra_verifiable_cnn_best_og.pth | Pending |

---

## Expert Verification Commands

### CIFAR-10 Expert (E_0)

#### E_0_CNN_NAT (Non-Robust Training)

```bash
# Epsilon = 2/255 (0.00784)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_0_CNN_NAT\cifar10_ultra_verifiable_cnn_best_og.pth" ^
    --dataset CIFAR10 ^
    --epsilon 0.00784 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 4/255 (0.01569)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_0_CNN_NAT\cifar10_ultra_verifiable_cnn_best_og.pth" ^
    --dataset CIFAR10 ^
    --epsilon 0.01569 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 8/255 (0.03137)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_0_CNN_NAT\cifar10_ultra_verifiable_cnn_best_og.pth" ^
    --dataset CIFAR10 ^
    --epsilon 0.03137 ^
    --num_images 20 ^
    --timeout 300
```

#### E_0_CNN_AT (Robust Training)

```bash
# Epsilon = 2/255 (0.00784)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_0_CNN_AT\cifar10_ultra_verifiable_cnn_best_robust.pth" ^
    --dataset CIFAR10 ^
    --epsilon 0.00784 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 4/255 (0.01569)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_0_CNN_AT\cifar10_ultra_verifiable_cnn_best_robust.pth" ^
    --dataset CIFAR10 ^
    --epsilon 0.01569 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 8/255 (0.03137)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_0_CNN_AT\cifar10_ultra_verifiable_cnn_best_robust.pth" ^
    --dataset CIFAR10 ^
    --epsilon 0.03137 ^
    --num_images 20 ^
    --timeout 300
```

---

### MNIST Expert (E_1)

#### E_1_CNN_NAT (Non-Robust Training)

```bash
# Epsilon = 2/255 (0.00784)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_1_CNN_NAT\mnist_ultra_verifiable_cnn_best_og.pth" ^
    --dataset MNIST ^
    --epsilon 0.00784 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 4/255 (0.01569)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_1_CNN_NAT\mnist_ultra_verifiable_cnn_best_og.pth" ^
    --dataset MNIST ^
    --epsilon 0.01569 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 8/255 (0.03137)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_1_CNN_NAT\mnist_ultra_verifiable_cnn_best_og.pth" ^
    --dataset MNIST ^
    --epsilon 0.03137 ^
    --num_images 20 ^
    --timeout 300
```

#### E_1_CNN_AT (Robust Training)

```bash
# Epsilon = 2/255 (0.00784)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_1_CNN_AT\mnist_ultra_verifiable_cnn_best_robust.pth" ^
    --dataset MNIST ^
    --epsilon 0.00784 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 4/255 (0.01569)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_1_CNN_AT\mnist_ultra_verifiable_cnn_best_robust.pth" ^
    --dataset MNIST ^
    --epsilon 0.01569 ^
    --num_images 20 ^
    --timeout 300

# Epsilon = 8/255 (0.03137)
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\E_1_CNN_AT\mnist_ultra_verifiable_cnn_best_robust.pth" ^
    --dataset MNIST ^
    --epsilon 0.03137 ^
    --num_images 20 ^
    --timeout 300
```

---

## Router (MoE) Verification Commands

### MoE_CNN_NAT (Non-Robust Training)

#### Epsilon = 2/255 (0.00784)

```bash
python verify_all_router_samples.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\MoE_CNN_NAT\meta_moe_ultra_verifiable_cnn_best_og.pth" ^
    --num_mnist 50 ^
    --num_cifar 50 ^
    --epsilon 0.00784 ^
    --timeout 300
```

#### Epsilon = 4/255 (0.01569)

```bash
python verify_all_router_samples.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\MoE_CNN_NAT\meta_moe_ultra_verifiable_cnn_best_og.pth" ^
    --num_mnist 50 ^
    --num_cifar 50 ^
    --epsilon 0.01569 ^
    --timeout 300
```

#### Epsilon = 8/255 (0.03137)

```bash
python verify_all_router_samples.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\MoE_CNN_NAT\meta_moe_ultra_verifiable_cnn_best_og.pth" ^
    --num_mnist 50 ^
    --num_cifar 50 ^
    --epsilon 0.03137 ^
    --timeout 300
```

### MoE_CNN_AT (Robust Training)

#### Epsilon = 2/255 (0.00784)

```bash
python verify_all_router_samples.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\MoE_CNN_AT\meta_moe_ultra_verifiable_cnn_best_og.pth" ^
    --num_mnist 50 ^
    --num_cifar 50 ^
    --epsilon 0.00784 ^
    --timeout 300
```

#### Epsilon = 4/255 (0.01569)

```bash
python verify_all_router_samples.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\MoE_CNN_AT\meta_moe_ultra_verifiable_cnn_best_og.pth" ^
    --num_mnist 50 ^
    --num_cifar 50 ^
    --epsilon 0.01569 ^
    --timeout 300
```

#### Epsilon = 8/255 (0.03137)

```bash
python verify_all_router_samples.py ^
    --model_path "D:\Mixture-of-Experts_Research\paper\artifacts\MoE_CNN_AT\meta_moe_ultra_verifiable_cnn_best_og.pth" ^
    --num_mnist 50 ^
    --num_cifar 50 ^
    --epsilon 0.03137 ^
    --timeout 300
```

---

## Results Summary

### Expert Verification Results

#### CIFAR-10 Expert (E_0)

**E_0_CNN_NAT (Non-Robust Training)**

| Epsilon | Verified | Falsified | Timeout | Unknown | Avg Time (s) | Notes |
|---------|----------|-----------|---------|---------|--------------|-------|
| 2/255 (0.00784) | | | | | | |
| 4/255 (0.01569) | | | | | | |
| 8/255 (0.03137) | | | | | | |

**E_0_CNN_AT (Robust Training)**

| Epsilon | Verified | Falsified | Timeout | Unknown | Avg Time (s) | Notes |
|---------|----------|-----------|---------|---------|--------------|-------|
| 2/255 (0.00784) | 18 | 0 | 2 | 32.3(s) | |
| 4/255 (0.01569) | 11 | 0 | 9 | 153.5(s) | | |
| 8/255 (0.03137) | | | | | | |

---

#### MNIST Expert (E_1)

**E_1_CNN_NAT (Non-Robust Training)**

| Epsilon | Verified | Falsified | Timeout | Unknown | Avg Time (s) | Notes |
|---------|----------|-----------|---------|---------|--------------|-------|
| 2/255 (0.00784) | | | | | | |
| 4/255 (0.01569) | | | | | | |
| 8/255 (0.03137) | | | | | | |

**E_1_CNN_AT (Robust Training)**

| Epsilon | Verified | Falsified | Timeout | Unknown | Avg Time (s) | Notes |
|---------|----------|-----------|---------|---------|--------------|-------|
| 2/255 (0.00784) | | | | | | |
| 4/255 (0.01569) | | | | | | |
| 8/255 (0.03137) | | | | | | |

---

### Router (MoE) Verification Results

**MoE_CNN_NAT (Non-Robust Training)**

| Epsilon | Verified | Falsified | Timeout | Unknown | Avg Time (s) | Notes |
|---------|----------|-----------|---------|---------|--------------|-------|
| 2/255 (0.00784) | 40 | 0 | 0 | 0 | 8.83(s) | 100% |
| 4/255 (0.01569) | 38 | 0 | 0 | 2 | 9.76(s) | 95% (GPU OOM) |
| 8/255 (0.03137) | 40 | 0 | 0 | 0 | 8.66(s) | 100% |

**MoE_CNN_AT (Robust Training)**

| Epsilon | Verified | Falsified | Timeout | Unknown | Avg Time (s) | Notes |
|---------|----------|-----------|---------|---------|--------------|-------|
| 2/255 (0.00784) | 39 | 0 | 0 | 1 | 9.14(s) | 97.5% (GPU OOM) |
| 4/255 (0.01569) | 40 | 0 | 0 | 0 | 9.25(s) | 100% |
| 8/255 (0.03137) | 40 | 0 | 0 | 0 | 8.59(s) | 100% |

---

## Hypothesis Verification Map

### Hypothesis 1: Adversarial Robustness-Accuracy Trade-off
- **Test**: Compare Certified Robustness Accuracy (CRA) between AT and NAT models
- **Status**: Pending verification
- **Expected**: AT models > NAT models in CRA

### Hypothesis 2: Sampling-Based Empirical Verification Translates to Formal Verification
- **Test**: Compare empirical AA (PGD attacks) with formal CRA from alpha-beta-CROWN
- **Status**: Pending verification
- **Expected**: Models with higher empirical robustness should have higher CRA

### Hypothesis 3: Compositional Robustness
- **Test**: Verify router and experts separately, then combine results
- **Status**: Pending verification
- **Expected**: Robust router + robust experts = verified MoE system

---

## Notes and Observations

- All models use VerifiableCNN architecture (96K parameters)
- Verification timeout: 300 seconds per sample
- Test set: 100 samples per expert (stratified by class if possible)
- Router test: 50 MNIST + 50 CIFAR10 samples
- Property: Correct classification maintained under L∞ perturbations

---

## Commands Reference

Run expert verification from the Mixture-of-Experts_Research directory:
```bash
cd D:\Mixture-of-Experts_Research
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py --help
```

Run router verification:
```bash
cd D:\Mixture-of-Experts_Research
python verify_all_router_samples.py --help
```

---

## Last Updated
- Initial creation: 2025-11-06
- Last run: (pending)
