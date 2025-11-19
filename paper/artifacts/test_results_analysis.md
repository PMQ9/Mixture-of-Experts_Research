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
| **RT Router** | CIFAR10-RT + MNIST-RT | **88.19** | 0.09 | **41.12** | 0.57 | 5954.77 | 45.20 |
| **RT Router** | CIFAR10-NRT + MNIST-NRT | **91.00** | 0.08 | **25.83** | 0.36 | 5949.87 | 73.14 |
| **RT Router** | CIFAR10-RT + MNIST-NRT | **88.25** | 0.05 | **30.70** | 0.21 | 5880.87 | 162.15 |
| **RT Router** | CIFAR10-NRT + MNIST-RT | **90.95** | 0.09 | **36.17** | 0.38 | 5970.26 | 107.91 |
| **NRT Router** | CIFAR10-RT + MNIST-RT | **88.17** | 0.05 | **41.22** | 0.38 | 2566.06 | 10.11 |
| **NRT Router** | CIFAR10-NRT + MNIST-NRT | **91.06** | 0.03 | **25.78** | 0.16 | 2618.02 | 107.86 |
| **NRT Router** | CIFAR10-RT + MNIST-NRT | **88.29** | 0.04 | **30.80** | 0.52 | 2588.01 | 73.20 |
| **NRT Router** | CIFAR10-NRT + MNIST-RT | **91.00** | 0.01 | **36.07** | 0.26 | 2566.80 | 16.71 |

---

### Table 2: Performance Metrics by Router Training Method

| Metric | RT Router | NRT Router | Difference | Relative Change |
|--------|-----------|------------|------------|-----------------|
| **Average Clean Acc (%)** | 89.60 | 89.64 | +0.04 | +0.04% |
| **Average Adv Acc (%)** | 33.46 | 33.47 | +0.01 | +0.03% |
| **Training Time (s)** | 5938.94 | 2584.72 | -3354.22 | **-56.5%** |
| **Time per Epoch (s)** | 118.78 | 51.69 | -67.09 | **-56.5%** |
| **Inference Time (ms/img)** | 0.062 | 0.063 | +0.001 | +1.6% |

---

### Table 3: Performance Metrics by Expert Configuration

| Expert Configuration | Clean Acc (%) | Adv Acc (%) | Robustness Gap (%) |
|---------------------|---------------|-------------|--------------------|
| **Both RT** | 88.18 | **41.17** | 47.01 |
| **Both NRT** | **91.03** | **25.78** | **65.25** |
| **CIFAR10-RT + MNIST-NRT** | 88.27 | 30.75 | 57.52 |
| **CIFAR10-NRT + MNIST-RT** | **90.98** | **36.12** | 54.86 |

**Robustness Gap** = Clean Accuracy - Adversarial Accuracy (lower is better)

---

## Detailed Results by Configuration

### Configuration 1: RT Router + CIFAR10-RT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.8811 | 0.4140 | 5946.85 | 118.94 |
| 2 | 0.8811 | 0.4121 | 6009.84 | 120.20 |
| 3 | 0.8832 | 0.4037 | 5913.95 | 118.28 |
| 4 | 0.8822 | 0.4173 | 5906.46 | 118.13 |
| 5 | 0.8817 | 0.4091 | 5996.75 | 119.93 |
| **Mean** | **0.8819** | **0.4112** | **5954.77** | **119.10** |
| **Std** | **0.0009** | **0.0057** | **45.20** | **0.90** |

---

### Configuration 2: RT Router + CIFAR10-NRT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.9106 | 0.2571 | 6075.07 | 121.50 |
| 2 | 0.9087 | 0.2596 | 5900.00 | 118.00 |
| 3 | 0.9102 | 0.2550 | 5974.14 | 119.48 |
| 4 | 0.9097 | 0.2555 | 5895.61 | 117.91 |
| 5 | 0.9110 | 0.2642 | 5904.52 | 118.09 |
| **Mean** | **0.9100** | **0.2583** | **5949.87** | **119.00** |
| **Std** | **0.0008** | **0.0036** | **73.14** | **1.46** |

---

### Configuration 3: RT Router + CIFAR10-RT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.8831 | 0.3059 | 5940.69 | 118.81 |
| 2 | 0.8826 | 0.3049 | 6087.23 | 121.74 |
| 3 | 0.8829 | 0.3070 | 5880.54 | 117.61 |
| 4 | 0.8817 | 0.3098 | 5630.86 | 112.62 |
| 5 | 0.8821 | 0.3075 | 5865.05 | 117.30 |
| **Mean** | **0.8825** | **0.3070** | **5880.87** | **117.62** |
| **Std** | **0.0005** | **0.0021** | **162.15** | **3.24** |

---

### Configuration 4: RT Router + CIFAR10-NRT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.9106 | 0.3595 | 6181.19 | 123.62 |
| 2 | 0.9095 | 0.3573 | 5927.64 | 118.55 |
| 3 | 0.9097 | 0.3632 | 5921.51 | 118.43 |
| 4 | 0.9097 | 0.3666 | 6006.19 | 120.12 |
| 5 | 0.9079 | 0.3620 | 5914.76 | 118.30 |
| **Mean** | **0.9095** | **0.3617** | **5970.26** | **119.81** |
| **Std** | **0.0009** | **0.0038** | **107.91** | **2.16** |

---

### Configuration 5: NRT Router + CIFAR10-RT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.8809 | 0.4076 | 2576.96 | 51.54 |
| 2 | 0.8821 | 0.4169 | 2569.42 | 51.39 |
| 3 | 0.8816 | 0.4112 | 2553.81 | 51.08 |
| 4 | 0.8818 | 0.4112 | 2555.50 | 51.11 |
| 5 | 0.8821 | 0.4142 | 2574.59 | 51.49 |
| **Mean** | **0.8817** | **0.4122** | **2566.06** | **51.32** |
| **Std** | **0.0005** | **0.0038** | **10.11** | **0.20** |

---

### Configuration 6: NRT Router + CIFAR10-NRT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.9105 | 0.2581 | 2588.66 | 51.77 |
| 2 | 0.9103 | 0.2584 | 2809.65 | 56.19 |
| 3 | 0.9106 | 0.2596 | 2554.65 | 51.09 |
| 4 | 0.9111 | 0.2577 | 2572.92 | 51.46 |
| 5 | 0.9106 | 0.2554 | 2564.23 | 51.28 |
| **Mean** | **0.9106** | **0.2578** | **2618.02** | **52.36** |
| **Std** | **0.0003** | **0.0016** | **107.86** | **2.16** |

---

### Configuration 7: NRT Router + CIFAR10-RT + MNIST-NRT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.8837 | 0.3127 | 2576.83 | 51.54 |
| 2 | 0.8828 | 0.3026 | 2541.09 | 50.82 |
| 3 | 0.8826 | 0.3092 | 2544.93 | 50.90 |
| 4 | 0.8826 | 0.3120 | 2548.90 | 50.98 |
| 5 | 0.8830 | 0.3034 | 2728.80 | 54.58 |
| **Mean** | **0.8829** | **0.3080** | **2588.01** | **51.76** |
| **Std** | **0.0004** | **0.0052** | **73.20** | **1.46** |

---

### Configuration 8: NRT Router + CIFAR10-NRT + MNIST-RT

| Run | Clean Acc | Adv Acc | Training Time (s) | Avg Time/Epoch (s) |
|-----|-----------|---------|-------------------|--------------------|
| 1 | 0.9099 | 0.3579 | 2552.21 | 51.04 |
| 2 | 0.9101 | 0.3579 | 2588.37 | 51.77 |
| 3 | 0.9101 | 0.3638 | 2581.42 | 51.63 |
| 4 | 0.9098 | 0.3602 | 2556.12 | 51.12 |
| 5 | 0.9100 | 0.3637 | 2555.88 | 51.12 |
| **Mean** | **0.9100** | **0.3607** | **2566.80** | **51.34** |
| **Std** | **0.0001** | **0.0026** | **16.71** | **0.33** |

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
| Both RT | 88.19% | 88.17% | -0.02% | 41.12% | 41.22% | +0.10% |
| Both NRT | 91.00% | 91.06% | +0.06% | 25.83% | 25.78% | -0.05% |
| CIFAR10-RT + MNIST-NRT | 88.25% | 88.29% | +0.04% | 30.70% | 30.80% | +0.10% |
| CIFAR10-NRT + MNIST-RT | 90.95% | 91.00% | +0.05% | 36.17% | 36.07% | -0.10% |
| **Average Difference** | - | - | **+0.04%** | - | - | **+0.01%** |

**Note**: Differences are within measurement noise (< 0.1%).

---

**Report Generated**: 2025-11-19
**Experiment Duration**: November 16-18, 2025 (49 hours)
**Total GPU Time**: 234,462 seconds (65.1 GPU-hours)
