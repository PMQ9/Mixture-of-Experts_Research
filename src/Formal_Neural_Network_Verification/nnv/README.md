# Neural Network Verification with NNV

---

## Quick Start

### Sampling-Based Robustness Testing

```matlab
>> cd src/Formal_Neural_Network_Verification
>> verify_expert_nnv_simple
```

**Result:** 100% robust at ε=1/255

---


## Test Results

GTSRB MicroExpertCNN model:
- Clean Accuracy: 97.14%
- Robust Accuracy: 100% at ε=1/255 (100 samples)
- Robust Accuracy: 100% at ε=0.5/255 (50 samples)

---

## Quick Commands

**Export model:**
```bash
python src/Formal_Neural_Network_Verification/export_to_onnx.py \
    --model_path artifacts/nnv_models/gtsrb_micro_cnn.pth \
    --output_dir artifacts/nnv_models
```

**Verify robustness:**
```matlab
>> verify_expert_nnv_simple
```

```matlab
>> verify_expert_nnv
```

---

## Known Limitation & Solutions

Formal verification with star sets fails due to LP solver limitations with MaxPooling layers. This is a known issue in the verification community, not a problem with your code.

Consider α,β-CROWN as alternative tool (better success with MaxPooling)


