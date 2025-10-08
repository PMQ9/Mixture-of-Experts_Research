# Expert Architecture Guide

## Available Expert Architectures

### Small Expert Architectures (Verification-Friendly)

#### 1. SmallExpertCNN (~1.5M parameters)
- **Use case**: GTSRB (43 classes) or complex datasets
- **Parameters**: 1,509,163
- **Architecture**: 4 conv blocks + 2 FC layers
- **Command**: `--model_arch small_cnn`

**Layer Details:**
```
Conv1: 3 -> 64 (32x32 -> 16x16)
Conv2: 64 -> 128 (16x16 -> 8x8)
Conv3: 128 -> 256 (8x8 -> 4x4)
Conv4: 256 -> 256 (4x4 -> 2x2)
FC1: 1024 -> 512
FC2: 512 -> num_classes
```

#### 2. TinyExpertCNN (~620K parameters)
- **Use case**: CIFAR-10, MNIST (10 classes)
- **Parameters**: 620,810
- **Architecture**: 3 conv blocks + 2 FC layers
- **Command**: `--model_arch tiny_cnn`

**Layer Details:**
```
Conv1: 3 -> 32 (32x32 -> 16x16)
Conv2: 32 -> 64 (16x16 -> 8x8)
Conv3: 64 -> 128 (8x8 -> 4x4)
FC1: 2048 -> 256
FC2: 256 -> num_classes
```

#### 3. MicroExpertCNN (~67K parameters) **[RECOMMENDED FOR VERIFICATION]**
- **Use case**: Formal verification with NNV/GNNV
- **Parameters**: 66,890
- **Architecture**: 3 conv blocks + 1 FC layer
- **Command**: `--model_arch micro_cnn`
- **Optimized for**: Neural network verification tools (minimal layers, simpler structure)

**Layer Details:**
```
Conv1: 3 -> 32 (32x32 -> 16x16)
Conv2: 32 -> 64 (16x16 -> 8x8)
Conv3: 64 -> 64 (8x8 -> 4x4)
FC: 1024 -> num_classes
```

---

## Comparison with ConvNeXt-Tiny

| Architecture | Parameters | Advantage | Disadvantage |
|--------------|-----------|-----------|--------------|
| **convnext_tiny** | ~28M | Higher accuracy | Too large for verification |
| **small_cnn** | ~1.5M | Good accuracy, 18× smaller | Lower accuracy than ConvNeXt |
| **tiny_cnn** | ~620K | Lightweight, 45× smaller | Further accuracy tradeoff |
| **micro_cnn** | ~67K | Verification-friendly, 418× smaller | Lowest accuracy, best for verification |

---

## Training Examples

### Train individual expert with small_cnn:
```bash
python src/Vision_Transformer_Pytorch/train_moe.py --dataset GTSRB --model_arch small_cnn --epochs 200
```

### Train individual expert with tiny_cnn:
```bash
python src/Vision_Transformer_Pytorch/train_moe.py --dataset CIFAR10 --model_arch tiny_cnn --epochs 200
```

### Train individual expert with micro_cnn (verification-optimized):
```bash
python src/Vision_Transformer_Pytorch/train_moe.py --dataset MNIST --model_arch micro_cnn --epochs 200
```

### Train MetaMoE with small experts:
First train individual experts, then:
```bash
python src/Vision_Transformer_Pytorch/train_moe.py --meta_moe \
    --model_arch small_cnn \
    --gtsrb_model_path artifacts/results/gtsrb_small_cnn_best.pth \
    --cifar10_model_path artifacts/results/cifar10_tiny_cnn_best.pth \
    --epochs 100
```

---

## Export to ONNX for Verification

Models are automatically exported to ONNX when `--export_onnx True` (default):

```bash
python src/Vision_Transformer_Pytorch/train_moe.py --dataset GTSRB --model_arch micro_cnn --export_onnx True
```

Output: `artifacts/gtsrb_micro_cnn.onnx`

---

## Why These Architectures for Verification?

1. **Explicit layer definitions**: All layers are explicitly defined (no dynamic operations)
2. **Simple operations**: Conv2d, BatchNorm, ReLU, MaxPool, Linear only
3. **Fewer layers**: Easier for verification tools to analyze
4. **No complex modules**: No attention mechanisms, no residual connections (in micro version)
5. **Fixed input size**: 32x32x3 RGB images
6. **Deterministic**: No dropout during inference

---

## Neural Network Verification Tools Compatibility

These architectures are designed to work with:
- **NNV (Neural Network Verification)**: MATLAB-based verification
- **GNNV**: GPU-accelerated neural network verification
- **ONNX Runtime**: Standard inference engine
- **Other verification tools**: Compatible with most tools that support ONNX format

---

## Recommended Usage

- **For research/production**: Use `small_cnn` or `tiny_cnn`
- **For formal verification**: Use `micro_cnn`
- **For maximum accuracy**: Use `convnext_tiny` (but verification will be challenging)

---

## Architecture File

All architectures are defined in: `src/Vision_Transformer_Pytorch/small_expert.py`

You can modify the architectures as needed for your specific verification requirements.
