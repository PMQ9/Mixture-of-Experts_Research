# NNV Verification Quick Start

Get up and running with formal verification in 5 steps!

## ⚡ 5-Minute Setup

### Step 1: Install NNV in MATLAB (One-time)

```matlab
% Open MATLAB and run:
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
```

**Expected output**: "NNV installed successfully" or similar messages

---

### Step 2: Train a Micro Expert (if you don't have one)

```bash
# In terminal/PowerShell:
cd d:\Mixture-of-Experts_Research

python src/Vision_Transformer_Pytorch/train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --batch_size 128
```

**Output**: `artifacts/results/gtsrb_micro_cnn_best.pth`

**Time**: ~10-20 minutes (depending on GPU)

---

### Step 3: Export to ONNX

```bash
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best.pth \
    --output_dir artifacts/nnv_models
```

**Output**: `artifacts/nnv_models/gtsrb_micro_cnn_best.onnx`

**Time**: ~5 seconds

---

### Step 4: Test ONNX Export (Optional but Recommended)

```bash
pip install onnxruntime  # if not already installed

python src/Formal_Neural_Network_Verification/test_onnx_export.py \
    --pth_model artifacts/results/gtsrb_micro_cnn_best.pth \
    --onnx_model artifacts/nnv_models/gtsrb_micro_cnn_best.onnx
```

**Expected**: "✓ ALL TESTS PASSED"

---

### Step 5: Verify in MATLAB

```matlab
% In MATLAB:
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')
quick_verify_example
```

**Expected output**:
```
=== Quick NNV Verification Example ===

✓ Model loaded: X layers
Predicted class: Y
✓ VERIFIED ROBUST: Network is robust within epsilon-ball
✓ Visualization complete
```

**Time**: ~1-5 minutes (depending on model and method)

---

## 🎯 What You Just Did

1. **Installed NNV**: Formal verification tool in MATLAB
2. **Trained a small CNN**: ~67K parameters (good for verification)
3. **Exported to ONNX**: Standard format for neural network exchange
4. **Tested correctness**: Ensured PyTorch ↔ ONNX consistency
5. **Verified robustness**: Proved all inputs in ε-ball are correctly classified

---

## 🚀 Next Steps

### Verify on Real Dataset

Edit `verify_expert_nnv.m` configuration section:

```matlab
% Update these lines:
onnx_model_path = fullfile('..', '..', 'artifacts', 'nnv_models', 'gtsrb_micro_cnn_best.onnx');
dataset_name = 'GTSRB';
epsilon = 2/255;
test_image_idx = 1;  % Try different images
```

Then run:
```matlab
verify_expert_nnv
```

### Try Different Epsilons

Test robustness at different perturbation levels:

```matlab
epsilons = [1/255, 2/255, 4/255, 8/255];
for i = 1:length(epsilons)
    epsilon = epsilons(i);
    % ... run verification
end
```

### Verify Multiple Images

Create a loop to compute **certified robust accuracy**:

```matlab
num_test = 100;
results = zeros(num_test, 1);

for img_idx = 1:num_test
    res = verify_image(net, img_idx, epsilon);
    results(img_idx) = res;
end

certified_accuracy = sum(results == 1) / num_test;
fprintf('Certified Robust Accuracy: %.2f%%\n', certified_accuracy * 100);
```

### Train Adversarially Robust Models

Models trained with adversarial training verify better:

```bash
python train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES \
    --trades_beta 6.0
```

---

## 📊 Understanding Your Results

### Verification Outcomes

| MATLAB Result | Meaning | What to Do |
|---------------|---------|------------|
| `res == 1` | ✅ **Verified Robust** | Great! Try larger epsilon |
| `res == 0` | ❌ **Not Robust** | Found adversarial example. Train with `--adv_training` |
| `res == -1` | ❓ **Unknown** | Approximation limit. Try exact method or smaller epsilon |

### Typical Results

**Standard Model (no adversarial training)**:
- ε = 1/255: Usually robust
- ε = 2/255: Sometimes robust
- ε = 8/255: Rarely robust

**Adversarially Trained Model**:
- ε = 1/255: Almost always robust
- ε = 2/255: Usually robust
- ε = 8/255: Sometimes robust

---

## 🛠️ Common Issues & Fixes

### Issue: "Undefined function 'onnx2nnv'"

**Fix**:
```matlab
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
```

### Issue: ONNX export fails

**Fix**: Make sure you're using a simple architecture:
```bash
# Use micro_cnn, not convnext or resnet
python export_to_onnx.py --model_path artifacts/results/gtsrb_micro_cnn_best.pth ...
```

### Issue: Verification takes forever

**Fix**: Use approximate method:
```matlab
reachOptions.reachMethod = 'approx-star';  % Fast
% Instead of:
% reachOptions.reachMethod = 'exact-star';  % Slow but precise
```

### Issue: Out of memory

**Fix 1**: Smaller epsilon
```matlab
epsilon = 1/255;  % Instead of 8/255
```

**Fix 2**: Use micro_cnn instead of small_cnn
```bash
python train_moe.py --model_arch micro_cnn  # 67K params
# Not: --model_arch small_cnn  # 1.5M params
```

---

## 📚 Learn More

### Documentation
- **[NNV_SETUP_GUIDE.md](NNV_SETUP_GUIDE.md)**: Comprehensive guide
- **[README.md](README.md)**: Tool reference and examples

### NNV Examples
Check out example scripts in:
```
modules/nnv_moe/code/nnv/examples/Tutorial/
├── NN/MNIST/verify.m
├── NN/GTSRB/verify_robust_1.m
└── ...
```

### Key Concepts

**Formal Verification**: Mathematical proof that a property holds
- Unlike testing (checks some inputs), verification checks **all** inputs in a set

**ε-ball (Epsilon Ball)**: All inputs within ε distance from original
- L∞ norm: Each pixel can change by at most ε

**Certified Robustness**: Mathematically proven that all inputs in ε-ball are correct
- Stronger than empirical robustness (testing on adversarial examples)

---

## 💡 Tips for Success

### 1. Start Small
- Use **micro_cnn** (~67K params)
- Test on **1 image** first
- Use **small epsilon** (1-2/255)
- Use **approx-star** method

### 2. Build Up Gradually
Once basics work:
- Increase epsilon
- Test more images
- Try exact methods
- Use larger models (tiny_cnn)

### 3. Compare Methods
Run both approximate and exact:
```matlab
% Approximate (fast)
reachOptions.reachMethod = 'approx-star';
res_approx = net.verify_robustness(IS, reachOptions, target);

% Exact (slow but precise)
reachOptions.reachMethod = 'exact-star';
res_exact = net.verify_robustness(IS, reachOptions, target);
```

### 4. Train for Verification
Models that verify better:
- Use adversarial training (`--adv_training`)
- Use TRADES loss (`--at_mode TRADES`)
- Train longer
- Use smaller learning rate

---

## 🎓 Example Complete Workflow

### End-to-End: Train → Export → Verify

```bash
# Terminal: Train adversarially robust model
python train_moe.py \
    --dataset GTSRB \
    --model_arch micro_cnn \
    --epochs 50 \
    --adv_training \
    --at_mode TRADES

# Terminal: Export to ONNX
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/results/gtsrb_micro_cnn_best_robust.pth \
    --output_dir artifacts/nnv_models

# Terminal: Test export
python src/Formal_Neural_Network_Verification/test_onnx_export.py \
    --pth_model artifacts/results/gtsrb_micro_cnn_best_robust.pth \
    --onnx_model artifacts/nnv_models/gtsrb_micro_cnn_best_robust.onnx
```

```matlab
% MATLAB: Verify
cd('d:\Mixture-of-Experts_Research\modules\nnv_moe\code\nnv')
startup_nnv
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')

% Edit verify_expert_nnv.m to point to your model
% Update: onnx_model_path = '...gtsrb_micro_cnn_best_robust.onnx'

% Run verification
verify_expert_nnv
```

---

## ✅ Checklist

Before asking for help, verify:

- [ ] NNV installed in MATLAB (ran `startup_nnv`)
- [ ] Model trained successfully (.pth file exists)
- [ ] ONNX export successful (.onnx file created)
- [ ] ONNX test passed (test_onnx_export.py)
- [ ] Dataset exists in correct location
- [ ] Paths in MATLAB scripts are correct

---

## 🆘 Get Help

**Still stuck?**

1. Check [NNV_SETUP_GUIDE.md](NNV_SETUP_GUIDE.md) for detailed troubleshooting
2. Review [README.md](README.md) for tool reference
3. Look at NNV examples: `modules/nnv_moe/code/nnv/examples/`
4. Check NNV issues: https://github.com/verivital/nnv/issues

---

**Ready to verify? Run this now:**

```matlab
cd('d:\Mixture-of-Experts_Research\src\Formal_Neural_Network_Verification')
quick_verify_example
```

Good luck! 🚀
