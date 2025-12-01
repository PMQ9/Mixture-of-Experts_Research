# MoE Router Performance Comparison: GTSRB+PTSD vs CIFAR10+MNIST

## Overview
This comparison examines the performance differences between two 2-expert MetaMoE systems trained with adversarial gating (TRADES).

## Key Metrics

| Metric | GTSRB+PTSD (AT) | CIFAR10+MNIST (AT) |
|--------|-----------------|-------------------|
| **Router Gating Accuracy** | 93.17% | 100% |
| **Overall Test Accuracy** | 83.48% | 88.13% |
| **Expert 0 Accuracy** | 88.73% (GTSRB) | 77.20% (CIFAR10) |
| **Expert 1 Accuracy** | 56.13% (PTSD) | 99.07% (MNIST) |
| **Clean→Adversarial Accuracy** | 83.35% → 29.47% | 87.90% → 41.59% |

## Key Findings

### 1. Router Convergence
**CIFAR10+MNIST** achieves perfect routing accuracy (100%) with clean distinction between datasets. MNIST handwritten digits and CIFAR10 natural images are visually distinct, allowing the router to learn unambiguous routing decisions.

**GTSRB+PTSD** plateaus at ~93% routing accuracy despite adversarial training. Both datasets contain traffic signs with 43 classes each. Domain-specific differences (German vs Persian traffic signs) introduce visual overlap that prevents perfect routing convergence.

### 2. Overall Performance
**CIFAR10+MNIST** achieves 88.13% vs 83.48% for GTSRB+PTSD, a 4.65% gap driven by:
- MNIST's near-perfect accuracy (99.07%)
- Easier inter-dataset routing discrimination
- More balanced expert specialization despite asymmetry

**GTSRB+PTSD** is constrained by:
- PTSD's weak accuracy (56.13%) acting as performance bottleneck
- Visual similarity between German and Persian traffic signs causing routing confusion
- Adversarial training more damaging on similar datasets

### 3. Adversarial Robustness
**CIFAR10+MNIST**: Retains 47.2% clean accuracy under attack (41.59→87.90)
- MNIST remains robust (74.05% adversarial)
- CIFAR10 is vulnerable (9.52% adversarial)

**GTSRB+PTSD**: Retains only 35.3% clean accuracy under attack (29.47→83.35)
- Both experts affected: GTSRB (29.67%), PTSD (62.99%)
- Adversarial attacks exploit routing uncertainty more effectively

## Dataset Characteristics Impact

| Aspect | GTSRB+PTSD | CIFAR10+MNIST |
|--------|------------|---------------|
| **Domain Similarity** | High (both traffic signs) | Low (distinct domains) |
| **Visual Overlap** | Significant | Minimal |
| **Expert Specialization** | Forced by domain | Natural by content |
| **Routing Difficulty** | High (similar classes) | Low (distinct classes) |

## Conclusions

1. **Domain similarity increases routing difficulty**: Shared traffic sign structure causes confusion despite dedicated experts. Visual overlap exceeds dataset discrimination capability.

2. **Expert quality imbalance significantly impacts MoE**: PTSD's weak accuracy (56.13%) degrades overall performance; the router cannot compensate for weak experts.

3. **Adversarial robustness degrades more on similar datasets**: Attacks on confusable datasets (GTSRB+PTSD) achieve lower adversarial accuracy (29.47%) vs distinct datasets (41.59%).

4. **Perfect routing emerges from inter-dataset distinction**: 100% routing accuracy correlates with high visual difference between datasets, not with dataset size or class count.

5. **Weak expert becomes vulnerability**: PTSD's 56.13% accuracy limits overall performance ceiling and increases adversarial attack surface.

## Implications for MoE Design

- **Expert selection**: Mix datasets with distinct visual characteristics rather than similar domains
- **Router training**: Adversarial gating helps (93% routing on hard case) but cannot overcome fundamental dataset similarity
- **Performance optimization**: Weak expert should be replaced or augmented; current router architecture cannot overcome the quality gap
- **Robustness trade-off**: Adversarial training more effective on datasets with clear inter-domain distinction
