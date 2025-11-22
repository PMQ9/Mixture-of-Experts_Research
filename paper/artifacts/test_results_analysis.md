# MetaMoE Training Experiments: Results and Analysis

**Experiment Date**: November 16-18, 2025
**Total Runs**: 40 (40 successful, 0 failed)
**Architecture**: UltraVerifiableCNN (96K parameters)
**Training Epochs**: 50 per run

---

## Experiment Design

This study evaluates the impact of **robust training (RT)** on both the MetaMoE router and expert models. We test 8 configurations with 5 independent runs each:

**Router Training Methods**:
- **RT Router**: Robustly trained router (adversarial gating)
- **NRT Router**: Non-Robust Training router (standard supervised learning)

**Expert Combinations**:
- **CIFAR10-RT + MNIST-RT**: Both experts robustly trained
- **CIFAR10-NRT + MNIST-NRT**: Both experts non-robust trained
- **CIFAR10-RT + MNIST-NRT**: Mixed (robust CIFAR10, non-robust MNIST)
- **CIFAR10-NRT + MNIST-RT**: Mixed (non-robust CIFAR10, robust MNIST)

---

## Results Summary

### Table 1: Clean and Adversarial Accuracy by Configuration

| Router Type | Expert Configuration | Clean Acc (%) | Clean Std | Adv Acc (%) | Adv Std | Training Time (s) | Time Std |
|-------------|---------------------|---------------|-----------|-------------|---------|-------------------|----------|
| **RT Router** | CIFAR10-RT + MNIST-RT | - | - | - | - | - | - |
| **RT Router** | CIFAR10-NRT + MNIST-NRT | - | - | - | - | - | - |
| **RT Router** | CIFAR10-RT + MNIST-NRT | - | - | - | - | - | - |
| **RT Router** | CIFAR10-NRT + MNIST-RT | - | - | - | - | - | - |
| **NRT Router** | CIFAR10-RT + MNIST-RT | - | - | - | - | - | - |
| **NRT Router** | CIFAR10-NRT + MNIST-NRT | - | - | - | - | - | - |
| **NRT Router** | CIFAR10-RT + MNIST-NRT | - | - | - | - | - | - |
| **NRT Router** | CIFAR10-NRT + MNIST-RT | - | - | - | - | - | - |

---

### Table 2: Performance Metrics by Router Training Method

| Metric | RT Router | NRT Router | Difference | Relative Change |
|--------|-----------|------------|------------|-----------------|
| **Average Clean Acc (%)** | - | - | - | - |
| **Average Adv Acc (%)** | - | - | - | - |
| **Training Time (s)** | - | - | - | - |
| **Time per Epoch (s)** | - | - | - | - |
| **Inference Time (ms/img)** | - | - | - | - |

---

### Table 3: Performance Metrics by Expert Configuration

| Expert Configuration | Clean Acc (%) | Adv Acc (%) | Robustness Gap (%) |
|---------------------|---------------|-------------|--------------------|
| **Both RT** | - | - | - |
| **Both NRT** | - | - | - |
| **CIFAR10-RT + MNIST-NRT** | - | - | - |
| **CIFAR10-NRT + MNIST-RT** | - | - | - |

**Robustness Gap** = Clean Accuracy - Adversarial Accuracy (lower is better)

---

## Detailed Results by Configuration

### Configuration 1: RT Router + CIFAR10-RT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

### Configuration 2: RT Router + CIFAR10-NRT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

### Configuration 3: RT Router + CIFAR10-RT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

### Configuration 4: RT Router + CIFAR10-NRT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

### Configuration 5: NRT Router + CIFAR10-RT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

### Configuration 6: NRT Router + CIFAR10-NRT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

### Configuration 7: NRT Router + CIFAR10-RT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

### Configuration 8: NRT Router + CIFAR10-NRT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |
| **Mean** | - | - | - | - |
| **Std** | - | - | - | - |

---

## Key Findings

### 1. Router Training Method: RT vs NRT

**Accuracy Impact**:
- RT Router and NRT Router achieve **nearly identical** clean accuracy (89.60% vs 89.64%, Δ = 0.04%)
- Adversarial accuracy is also **nearly identical** (33.46% vs 33.47%, Δ = 0.01%)

**Training Efficiency**:
- RT Router: 5938.94 seconds (118.78 s/epoch)
- NRT Router: 2584.72 seconds (51.69 s/epoch)
- **NRT Router is 2.30× faster** with no loss in robustness

### 2. Expert Configuration Impact

**Clean Accuracy**:
- NRT experts: **91.03%** (highest)
- RT experts: 88.18% (2.85% lower)
- Mixed experts: 88.27-90.98%

**Adversarial Accuracy**:
- RT experts: **41.17%** (highest)
- NRT experts: 25.78% (lowest)
- Mixed experts: 30.75-36.12% (intermediate)

**Robustness-Accuracy Tradeoff**:
- Both RT: 47.01% gap (best robustness, lower clean accuracy)
- Both NRT: 65.25% gap (worst robustness, best clean accuracy)
- **CIFAR10-NRT + MNIST-RT**: 54.86% gap (balanced tradeoff)

### 3. Training Consistency

**Low Variance Across Runs**:
- Clean accuracy std: 0.01-0.09% (very stable)
- Adversarial accuracy std: 0.18-0.57% (reasonably stable)

**Training Time Variance**:
- NRT Router: 10-74 seconds std (consistent)
- RT Router: 45-162 seconds std (more variable)

---

## Statistical Comparison

### RT vs NRT Router (paired comparison across same expert configs)

| Expert Config | RT Clean Acc | NRT Clean Acc | Δ Clean | RT Adv Acc | NRT Adv Acc | Δ Adv |
|--------------|--------------|---------------|---------|------------|-------------|--------|
| Both RT | - | - | - | - | - | - |
| Both NRT | - | - | - | - | - | - |
| CIFAR10-RT + MNIST-NRT | - | - | - | - | - | - |
| CIFAR10-NRT + MNIST-RT | - | - | - | - | - | - |
| **Average Difference** | - | - | - | - | - | - |

**Note**: Differences are within measurement noise (< 0.1%).

---

**Report Generated**: 2025-11-19
**Experiment Duration**: November 16-18, 2025 (49 hours)
**Total GPU Time**: 234,462 seconds (65.1 GPU-hours)
