# NNV Verification Setup Guide

This guide explains how to set up and use NNV (Neural Network Verification) to formally verify your Mixture-of-Experts architecture.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [NNV Installation](#nnv-installation)
3. [Model Export Workflow](#model-export-workflow)
4. [Running Verification](#running-verification)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **MATLAB** (2023a or newer) with toolboxes:
  - Computer Vision
  - Deep Learning
  - Image Processing
  - Optimization
  - Statistics and Machine Learning
  - Symbolic Math
- **Python** 3.10+ with PyTorch

### MATLAB Add-ons (Install via MATLAB Add-On Manager)
1. Deep Learning Toolbox Converter for ONNX Model Format
   - Home tab → Add-Ons → Get Add-ons → Search "ONNX"

---

## NNV Installation

### 1. Clone NNV (Already Done)
Your forked NNV is located at: `modules/nnv_moe/`

### 2. Initialize NNV in MATLAB

```matlab
% Navigate to NNV directory
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')

% Run installation script
install

% Or just add to path (faster for repeated use)
startup_nnv

% Optional: Save path permanently (requires admin privileges)
savepath
```

### 3. Verify Installation

```matlab
% Test if NNV is working
help onnx2nnv
% Should display help text for onnx2nnv function
```

---

## Model Export Workflow

### Step 1: Train an Expert Model

For verification, **MicroExpertCNN** is recommended due to its small size (~67K params):

```bash
# Train GTSRB expert with micro_cnn
python src/Vision_Transformer_Pytorch/train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --batch_size 128
```

Output: `artifacts/results/gtsrb_micro_cnn_best.pth`

### Step 2: Export to ONNX

```bash
# Export single model
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best.pth \
    --output_dir artifacts/nnv_models

# Or export all micro_cnn models
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --models_dir artifacts/results \
    --output_dir artifacts/nnv_models \
    --filter micro_cnn
```

Output: `artifacts/nnv_models/gtsrb_micro_cnn_best.onnx`

### Step 3: Verify ONNX Model (Optional)

```python
import onnx

model = onnx.load('artifacts/nnv_models/gtsrb_micro_cnn_best.onnx')
onnx.checker.check_model(model)
print("ONNX model is valid!")
```

---

## Running Verification

### Method 1: Using the Provided Script

1. Open MATLAB
2. Navigate to verification directory:
   ```matlab
   cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')
   ```

3. Edit `verify_expert_nnv.m` configuration section:
   ```matlab
   % Set these parameters
   onnx_model_path = fullfile('..', '..', 'artifacts', 'nnv_models', 'gtsrb_micro_cnn_best.onnx');
   dataset_name = 'GTSRB';
   epsilon = 2/255;  % L-infinity perturbation bound
   reachMethod = 'approx-star';  % Fast approximate method
   test_image_idx = 1;
   ```

4. Run the script:
   ```matlab
   verify_expert_nnv
   ```

### Method 2: Custom MATLAB Script

```matlab
%% Load NNV
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

%% Load model
net = onnx2nnv('../../artifacts/nnv_models/gtsrb_micro_cnn_best.onnx');

%% Load test image
img = imread('path/to/test/image.ppm');
img = single(imresize(img, [32 32])) / 255.0;

% Normalize (GTSRB)
meanNorm = [0.3337, 0.3064, 0.3171];
stdNorm = [0.2672, 0.2564, 0.2629];
for c = 1:3
    img(:,:,c) = (img(:,:,c) - meanNorm(c)) / stdNorm(c);
end

%% Create input set
epsilon = 2/255;
lb = img - epsilon;
ub = img + epsilon;
IS = ImageStar(lb, ub);

%% Verify robustness
reachOptions = struct;
reachOptions.reachMethod = 'approx-star';
target_class = 1;  % Target class index

res = net.verify_robustness(IS, reachOptions, target_class);

if res == 1
    disp('✓ Network is verified robust!');
elseif res == 0
    disp('✗ Network is not robust');
else
    disp('? Unknown result');
end
```

---

## Understanding Results

### Verification Outcomes

| Result | Meaning |
|--------|---------|
| `res == 1` | **VERIFIED ROBUST** - All perturbed inputs are correctly classified |
| `res == 0` | **NOT ROBUST** - Found a counterexample (adversarial input) |
| `res == -1` | **UNKNOWN** - Could not determine (timeout, approximation limitations) |

### Reachability Methods

| Method | Type | Speed | Accuracy | Best For |
|--------|------|-------|----------|----------|
| `'exact-star'` | Sound & Complete | Slow | Exact | Small models, critical verification |
| `'approx-star'` | Sound but Incomplete | Fast | Over-approximate | Quick checks, larger models |
| `'abs-dom'` | Abstract Domains | Very Fast | Very approximate | Initial screening |

### Interpreting Output Ranges

The visualization shows:
- **Error bars**: Range of possible outputs for each class within the input set
- **Red X**: Prediction for the original (unperturbed) image
- **Green line**: True class

**Robust if**: The true class has the highest lower bound (minimum possible output)

---

## Verification Parameters

### Epsilon (ε) Selection

```matlab
% Common choices for L-infinity perturbations
epsilon = 1/255;    % Very small (0.004) - highly robust
epsilon = 2/255;    % Small (0.008) - typical for MNIST
epsilon = 8/255;    % Standard (0.031) - typical for CIFAR-10/GTSRB
epsilon = 16/255;   % Large (0.063) - challenging
```

### Dataset-Specific Normalization

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

## Troubleshooting

### Issue 1: NNV Not Found

**Error**: `Undefined function or variable 'onnx2nnv'`

**Solution**:
```matlab
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
```

### Issue 2: ONNX Import Fails

**Error**: `Could not load ONNX model`

**Possible causes**:
1. Unsupported layer types
2. ONNX version mismatch
3. Complex operations

**Solutions**:
- Use simpler architectures (micro_cnn, tiny_cnn)
- Try different opset versions: `--opset_version 11` (recommended), `9`, or `13`
- Check ONNX model with `netron` tool: `netron model.onnx`

### Issue 3: Out of Memory

**Error**: MATLAB runs out of memory during verification

**Solutions**:
1. Use approximate methods instead of exact: `reachOptions.reachMethod = 'approx-star'`
2. Reduce input set size (smaller epsilon)
3. Use smaller model (micro_cnn instead of small_cnn)
4. Enable parallel computing (if available):
   ```matlab
   reachOptions.numCores = 4;
   ```

### Issue 4: Verification Takes Too Long

**Solutions**:
1. Use `'approx-star'` instead of `'exact-star'`
2. Set timeout:
   ```matlab
   reachOptions.timeout = 300;  % 5 minutes
   ```
3. Use abstract domain methods:
   ```matlab
   reachOptions.reachMethod = 'abs-dom';
   ```

### Issue 5: Dataset Not Found

**Error**: `GTSRB test images not found`

**Solution**:
Ensure dataset structure matches:
```
data/
├── GTSRB/
│   ├── Training/
│   └── Test/
│       └── Images/
├── CIFAR10/
│   ├── Training/
│   └── Test/
└── MNIST/
    ├── Training/
    └── Test/
```

---

## Best Practices

### 1. Start Small
- Begin with **micro_cnn** (~67K params)
- Use **approx-star** method
- Test on **1-10 images** first

### 2. Choose Appropriate Epsilon
- Start with small values (1-2/255)
- Increase gradually to find robustness limits

### 3. Verify Adversarially Trained Models
Models trained with `--adv_training` are more likely to verify:
```bash
python train_moe.py --dataset GTSRB --model_arch micro_cnn --epochs 50 --adv_training
```

### 4. Batch Verification
Create a script to verify multiple images:
```matlab
for img_idx = 1:100
    res = verify_image(net, img_idx, epsilon);
    results(img_idx) = res;
end
certified_robust_accuracy = sum(results == 1) / length(results);
```

---

## Advanced Topics

### Verifying MetaMoE System

For MetaMoE, you need to verify:
1. **Individual experts** (covered above)
2. **Router network** (meta gating network)
3. **Combined system** (more complex)

### Export MetaGatingNet:
```python
# In train_moe.py or custom script
meta_moe = torch.load('artifacts/meta_moe_small_cnn_best.pth')
router = meta_moe.router

# Export router separately
torch.onnx.export(router, dummy_input, 'router.onnx')
```

### Compositional Verification
- Verify each expert independently
- Verify router classifies inputs correctly
- Combine results for end-to-end guarantee

---

## Example Verification Workflow

### Complete Example: GTSRB Micro Expert

```bash
# 1. Train model
python src/Vision_Transformer_Pytorch/train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --export_onnx True

# 2. Export to ONNX (if not done automatically)
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best.pth \
    --output_dir artifacts/nnv_models
```

```matlab
% 3. Verify in MATLAB
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')
verify_expert_nnv  % Run verification script
```

---

## Additional Resources

### NNV Documentation
- Official repo: https://github.com/verivital/nnv
- Tutorial examples: `modules/nnv_moe/code/nnv/examples/Tutorial/`
- GTSRB examples: `modules/nnv_moe/code/nnv/examples/Tutorial/NN/GTSRB/`

### Research Papers
- NNV 2.0: [CAV 2023 Tool Paper](https://link.springer.com/chapter/10.1007/978-3-031-37703-7_19)
- Original NNV: [CAV 2020 Tool Paper](https://link.springer.com/chapter/10.1007/978-3-030-53288-8_1)

### Related Work
- α,β-CROWN: Alternative verification tool
- ERAN: ETH Robustness Analyzer
- Marabou: SMT-based verifier

---

## Contact and Support

For issues specific to:
- **NNV tool**: https://github.com/verivital/nnv/issues
- **This project**: Check CLAUDE.md or contact repository maintainers
