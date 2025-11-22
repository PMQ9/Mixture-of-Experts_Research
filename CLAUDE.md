# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT: DO NOT USE EMOJIS** in any code, documentation, or communication unless explicitly requested by the user.

**IMPORTANT: Terminology** - Use "NRT" (Non-Robust Training) instead of "NAT" (Non-Adversarially Trained) in all documentation. NRT is the correct term for standard supervised learning without adversarial defenses.

## Project Overview

This is a Mixture-of-Experts (MoE) research project for safety-critical systems at Vanderbilt University's Institute of Software Integrated Systems. The project trains separate expert models on different datasets (GTSRB, CIFAR-10, MNIST, etc.), freezes them, and trains a meta-router to dynamically select which expert to activate based on input. This approach is designed to be scalable and resource-efficient.

## Core Architecture

### Two-Level MoE System

1. **Token-level MoE** (`VisionTransformer` with `MoEBlock`):
   - Vision Transformer with mixture-of-experts at each transformer block
   - Each block routes tokens to top-k experts dynamically
   - Used for training individual dataset experts
   - Located in `src/Vision_Transformer_Pytorch/vision_transformer_moe.py`

2. **Dataset-level MoE** (`MetaMoE`):
   - High-level router (`MetaGatingNet`) that routes entire images to frozen expert models
   - Each expert is a complete pre-trained model for a specific dataset
   - Supports fine-tuning by adding new experts incrementally
   - Also in `src/Vision_Transformer_Pytorch/vision_transformer_moe.py`

### Key Components

**Expert Architectures** (`src/Vision_Transformer_Pytorch/small_expert.py`):
- `SmallExpertCNN`: ~1.5M params (for complex datasets like GTSRB)
- `TinyExpertCNN`: ~620K params (for CIFAR-10/MNIST)
- `MicroExpertCNN`: ~67K params (optimized for formal verification tools, uses MaxPool)
- `NNVCompatibleCNN`: ~67K params (uses AvgPool instead of MaxPool for NNV compatibility, 1-3% accuracy drop)
- Also supports larger models: ConvNeXt-Tiny (~28M params), ResNet50, EfficientNet, etc.

**Router Architecture**:
- `AttentionRouter`: Token-level routing using learnable expert tokens
- `MetaGatingNet`: Dataset-level routing using ConvNeXt/ResNet backbone with temperature-scaled softmax

**Training Pipeline** (`src/Vision_Transformer_Pytorch/train_moe.py`):
- Supports both individual expert training and MetaMoE training
- Includes adversarial training (PGD, TRADES)
- CutMix augmentation for individual experts
- Automatic ONNX export:
  - Standard ONNX via `log_functions.py` (general use)
  - NNV-compatible ONNX via `Formal_Neural_Network_Verification/export_to_onnx.py` (automatic for nnv_cnn, micro_cnn, tiny_cnn, small_cnn)

## Common Commands

### Training Individual Experts

Train a GTSRB expert with small CNN:
```bash
python train.py --dataset GTSRB --model_arch small_cnn --epochs 100
```

Train CIFAR-10 expert with tiny CNN:
```bash
python train.py --dataset CIFAR10 --model_arch tiny_cnn --epochs 100
```

Train with adversarial robustness:
```bash
python train.py --dataset GTSRB --model_arch small_cnn --epochs 100 --adv_training
```

### Training MetaMoE

**IMPORTANT:** For MetaMoE training, use `--gating_backbone` (NOT `--model_arch`) to specify the router architecture. The `--model_arch` argument specifies the expert architecture, while `--gating_backbone` specifies the router/gating network architecture.

**Recommended architecture:** Use `ultra_verifiable_cnn` for both experts and router for formal verification compatibility.

Train MetaMoE with 2 pre-trained experts (supports wildcards in paths):
```bash
python train.py --meta_moe \
    --model_arch ultra_verifiable_cnn \
    --gating_backbone ultra_verifiable_cnn \
    --gtsrb_model_path artifacts/results/gtsrb_*_best.pth \
    --cifar10_model_path artifacts/results/cifar10_*_best.pth \
    --epochs 100 \
    --meta_top_k 1
```

Fine-tune MetaMoE by adding a new expert:
```bash
python train.py --meta_moe --fine_tune_meta_moe \
    --model_arch ultra_verifiable_cnn \
    --gating_backbone ultra_verifiable_cnn \
    --mnist_model_path artifacts/results/mnist_*_best.pth
```

Train MetaMoE with adversarial gating:
```bash
python train.py --meta_moe \
    --model_arch ultra_verifiable_cnn \
    --gating_backbone ultra_verifiable_cnn \
    --gtsrb_model_path artifacts/results/gtsrb_*_best.pth \
    --cifar10_model_path artifacts/results/cifar10_*_best.pth \
    --adv_gating_train \
    --at_mode TRADES \
    --trades_beta 6.0
```

### Model Evaluation

Evaluate a single image:
```bash
python src/Vision_Transformer_Pytorch/meta_moe_eval.py \
    --model_path artifacts/meta_moe_small_cnn_best.pth \
    --image_path data/GTSRB/Test/Images/00000.ppm \
    --model_type meta \
    --device cuda
```

Test adversarial robustness:
```bash
python train.py --dataset GTSRB \
    --art_attack \
    --art_attack_mode PGD
```

### Configuration Overrides

Override Vision Transformer config parameters:
```bash
python train.py --dataset GTSRB \
    --config_overrides "img_size=48,patch_size=8,embed_dim=256,depth=12,num_experts=5,top_k=2"
```

Available config parameters (see `VisionTransformerConfig` in `vision_transformer_moe.py`):
- `img_size`, `patch_size`, `embed_dim`, `depth`, `num_heads`, `mlp_ratio`
- `num_experts`, `top_k`, `balance_loss_weight`, `drop_path_rate`

### Utilities

Calculate dataset normalization values:
```bash
python src/Normalization_Value/gtsrb_normalization_compute.py --dataset GTSRB
```

Test small expert architectures:
```bash
python src/Vision_Transformer_Pytorch/small_expert.py
```

Visualize robustness (expert switching under adversarial perturbations):
```bash
python train.py --meta_moe --visualize_robustness
```

## Dataset Structure

Datasets must be organized with CSV metadata files containing `meta_class` labels:

```
data/
├── GTSRB/
│   ├── Training/
│   │   ├── train_with_meta_class.csv  # Required: has 'meta_class' column
│   │   └── [images organized by class folders]
│   └── Test/
│       ├── testset_with_meta_class.csv  # Required
│       └── Images/
├── CIFAR10/
│   ├── Training/
│   │   └── train_with_meta_class.csv
│   └── Test/
│       └── testset_with_meta_class.csv
└── MNIST/
    ├── Training/
    │   └── train_with_meta_class.csv
    └── Test/
        └── testset_with_meta_class.csv
```

**CSV format requirements**:
- Training CSV: `Filename`, `ClassId`, `meta_class` columns
- Test CSV: `Filename`, `ClassId`, `meta_class` columns (meta_class can be auto-filled via `default_meta_class` parameter)

**Meta-class mapping** (for MetaMoE):
- 0: GTSRB
- 1: CIFAR-10
- 2: MNIST

## Model Output Structure

Trained models are saved to `artifacts/` or `artifacts/results/`:

- Individual experts: `{dataset}_{architecture}_best.pth` (e.g., `gtsrb_small_cnn_best.pth`)
- MetaMoE: `meta_moe_{architecture}_best.pth`
- With adversarial training: `*_best_robust.pth`
- Fine-tuned: `*_best_finetuned.pth`
- ONNX exports: `{dataset}_{architecture}.onnx`

Training artifacts:
- `training_log.txt`: Full training logs
- `training_metrics.png`: Training/validation curves
- `pipeline_log.txt`: CI/CD pipeline logs (GitLab)

## Pre-trained Experts (Paper Experiments)

Pre-trained expert models are stored in `paper/artifacts/`:

```
paper/artifacts/
├── E_0_CNN_AT/          # CIFAR-10 Adversarially Trained expert
│   └── cifar10_ultra_verifiable_cnn_best_robust.pth
├── E_0_CNN_NAT/         # CIFAR-10 Non-Adversarially Trained expert
│   └── cifar10_ultra_verifiable_cnn_best_og.pth
├── E_1_CNN_AT/          # MNIST Adversarially Trained expert
│   └── mnist_ultra_verifiable_cnn_best_robust.pth
├── E_1_CNN_NAT/         # MNIST Non-Adversarially Trained expert
│   └── mnist_ultra_verifiable_cnn_best_og.pth
├── MoE_CNN_AT/          # MoE Router trained with Adversarial Gating (outputs go here)
└── MoE_CNN_NAT/         # MoE Router trained with Non-Adversarial Gating (outputs go here)
```

**Expert naming convention:**
- `E_0_*`: CIFAR-10 expert
- `E_1_*`: MNIST expert
- `*_AT`: Adversarially Trained (robust) → `*_best_robust.pth`
- `*_NAT`: Non-Adversarially Trained (standard) → `*_best_og.pth`

**Default architecture:** `ultra_verifiable_cnn` (96K parameters)

**Training MoE routers with pre-trained experts:**

Train router with AT gating and AT experts:
```bash
python train.py --meta_moe \
    --model_arch ultra_verifiable_cnn \
    --gating_backbone ultra_verifiable_cnn \
    --cifar10_model_path paper/artifacts/E_0_CNN_AT/cifar10_ultra_verifiable_cnn_best_robust.pth \
    --mnist_model_path paper/artifacts/E_1_CNN_AT/mnist_ultra_verifiable_cnn_best_robust.pth \
    --adv_gating_train \
    --art_attack \
    --epochs 200
```

Train router with NAT gating and mixed experts (CIFAR10-AT + MNIST-NAT):
```bash
python train.py --meta_moe \
    --model_arch ultra_verifiable_cnn \
    --gating_backbone ultra_verifiable_cnn \
    --cifar10_model_path paper/artifacts/E_0_CNN_AT/cifar10_ultra_verifiable_cnn_best_robust.pth \
    --mnist_model_path paper/artifacts/E_1_CNN_NAT/mnist_ultra_verifiable_cnn_best_og.pth \
    --art_attack \
    --epochs 200
```

The `--adv_gating_train` flag enables adversarial training for the router (gating network). Omit it for non-adversarial router training.

## Neural Network Verification

This project integrates with formal verification tools (NNV/GNNV in MATLAB):

**File conversions** (`src/Formal_Neural_Network_Verification/File_Conversion/`):
- `pth_to_mat.py`: Convert PyTorch models to MATLAB .mat format
- `onnx_to_mat.py`: Convert ONNX models to MATLAB format

**Verification workflow**:
1. Train small expert (preferably `micro_cnn` for faster verification)
2. Export to ONNX automatically with `--export_onnx True`
3. Convert to .mat format for NNV/GNNV
4. Run verification in MATLAB using `modules/nnv_moe/`

**Submodules**:
- NNV and GNNV are included as git submodules in `modules/`
- Initialize: `git submodule update --init --recursive`

## Important Implementation Details

### MetaMoE Routing Behavior

- The `MetaMoE` routes **entire images** to experts (dataset-level routing), not individual tokens
- Each expert produces logits for its own class space (e.g., GTSRB: 43 classes, CIFAR-10: 10 classes)
- Outputs are concatenated into a unified output space with `class_offsets`
- Example: If GTSRB has 43 classes and CIFAR-10 has 10, total output is 53 classes
  - GTSRB classes: [0, 42]
  - CIFAR-10 classes: [43, 52]

### Loss Functions

**For individual experts**:
- Classification loss: `LabelSmoothingCrossEntropy` (smoothing=0.1)
- Balance loss: Encourages uniform expert usage in token-level MoE
- Total loss: `cls_loss + balance_loss_weight * balance_loss`

**For MetaMoE**:
- Classification loss: `LabelSmoothingCrossEntropy` on combined output
- Gating loss: `CrossEntropyLoss` on router predictions vs. true meta_class
- Total loss: `cls_loss + gating_loss_weight * gating_loss`

### Adversarial Training Modes

**PGD (Projected Gradient Descent)**:
- Generates adversarial examples via iterative FGSM
- Loss: `(clean_loss + adv_loss) / 2`

**TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)**:
- Minimizes KL divergence between clean and adversarial predictions
- Loss: `adv_ce_loss + trades_beta * kl_div`

### Model Architecture Files

**Main files**:
- `vision_transformer_moe.py`: Core MoE classes (VisionTransformer, MetaMoE, routers, datasets)
- `small_expert.py`: Lightweight expert architectures for verification
- `model_wrapper.py`: Factory functions for creating models, wrapper classes
- `train_moe.py`: Main training script with all training logic
- `config.py`: Normalization values and default hyperparameters

**Model loading**:
- Models are saved as complete PyTorch objects (not just state_dicts) for easier loading
- To load: `model = torch.load(path, map_location=device, weights_only=False)`
- MetaMoE models contain frozen expert models as `nn.ModuleList`

## CI/CD Pipeline

GitLab CI/CD is configured (`.gitlab-ci.yml`) with stages:
1. `test_gpu`: Check GPU availability
2. `prepare`: Clone/update repository
3. `train`: Run training pipeline
4. `clean_up`: Cleanup (not implemented)

**Environment variable**: `CICD_EPOCH` controls epoch count in CI (default: uses `DEFAULT_PARAMS['epoch']`)

**Runner tag**: `Gem12Server` (configured for Windows with PowerShell)

## Development Notes

### Adding New Expert Architectures

1. Define architecture in `small_expert.py` or create new file
2. Add import to `model_wrapper.py`
3. Add case to `create_model()` function in `model_wrapper.py`
4. Add choice to `--model_arch` argument in `train_moe.py`

### Adding New Datasets

1. Create dataset directory structure in `data/`
2. Generate CSV files with `meta_class` column
3. Add normalization values to `config.py`
4. Add dataset parameters to `dataset_params` dict in `train_moe.py` main()
5. Update `--dataset` choices in argument parser

### Modifying Router Architecture

For token-level routing:
- Edit `AttentionRouter` or `MoEBlock` in `vision_transformer_moe.py`

For dataset-level routing:
- Edit `MetaGatingNet` in `vision_transformer_moe.py`
- Modify `--gating_backbone` choices to add new backbone architectures

## Python Environment

**Python version**: 3.10

**Key dependencies**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm matplotlib netron onnx adversarial-robustness-toolbox timm scipy
```

## Neural Network Verification (NNV)

### Status: TOOL LIMITATIONS - USE SAMPLING-BASED VERIFICATION

NNV (Neural Network Verification toolkit) integration has been implemented but **formal verification cannot complete due to fundamental NNV limitations**, not model design issues.

**Main documentation:**
- `src/Formal_Neural_Network_Verification/NNV_VERIFICATION_GUIDE.md` - Complete setup and usage guide
- `src/Formal_Neural_Network_Verification/WHY_NNV_WORKS_FOR_OTHERS.md` - Explains why NNV works for research papers but not our models
- `NNV_README.md` - Quick reference at repository root

### What Was Implemented

**1. Verification-Optimized Architecture (UltraVerifiableCNN):**
- 96K parameters (4% smaller than baseline NNVCompatibleCNN)
- 4 convolutional layers with gradual channel growth (20→28→40→56)
- Average pooling instead of max pooling (linear operations for LP solvers)
- Achieved 87% accuracy on GTSRB (acceptable trade-off for verifiability)

**2. BatchNorm Folding:**
- Automatically folds BatchNorm parameters into convolutional layer weights during ONNX export
- Eliminates BatchNorm layers from ONNX graph (reduces from 11 to 15 operators, 0 BN layers)
- Mathematically equivalent (verified: max error 3.29e-05)
- Implemented in `export_to_onnx.py` via `fold_batch_norm_into_conv()` and `NNVSimplifiedWrapper`

**3. Clean ONNX Export:**
- Static `Flatten` operator instead of dynamic `Shape+Gather+Reshape` (eliminates ONNXParams error)
- Minimal operator count (15 operators for UltraVerifiableCNN)
- Supports all expert architectures: Micro, Tiny, Small, NNVCompatible, UltraVerifiable

### Why NNV Formal Verification Fails

**Observed behavior:**
- ONNX has 15 operators, NNV reports 17 layers (internal expansion)
- Verification API fails: "Output size is set to 0, but it must be >= 1"
- LP solver runs 14+ minutes, returns "UNKNOWN" instead of TRUE/FALSE
- Even with Gurobi optimizer, verification times out

**Root causes:**
- NNV is designed for tiny fully-connected networks (ACAS Xu: 300 neurons, 13-20K params)
- Our 96K-parameter CNN is 5-7x larger and uses convolutions (much harder than FC)
- Input dimensionality: 3,072D (32x32x3) vs 5D for ACAS Xu (exponential complexity)
- Even on ACAS Xu, NNV only verifies 64% of networks (best method)

**Conclusion:** Our network is beyond state-of-the-art for formal verification. This is a tool limitation, not a model design failure.

### What Works: Sampling-Based Verification

**Recommended approach:**
```matlab
>> cd src/Formal_Neural_Network_Verification
>> verify_expert_nnv_simple
```

**Results:**
- Tests 100 random perturbations within epsilon ball
- Completes in ~10 seconds
- UltraVerifiableCNN: 95% robust (95/100 samples at ε=1/255)
- Scientifically valid and widely accepted in top-tier papers

### Training Verification-Optimized Models

```bash
# UltraVerifiableCNN (recommended for verification)
python train.py --dataset GTSRB --model_arch ultra_verifiable_cnn --epochs 100 --adv_training

# NNVCompatibleCNN (baseline comparison)
python train.py --dataset GTSRB --model_arch nnv_cnn --epochs 100 --adv_training
```

**Automatic ONNX export:** Models are automatically exported with BatchNorm folding to `artifacts/nnv_models/`

### Expert Architectures for Verification

**UltraVerifiableCNN (NEW - Balanced):**
- Parameters: 96K (4% smaller than NNVCompatibleCNN)
- Structure: 4 conv (20→28→40→56 channels) + 3 AvgPool + 2 FC
- Accuracy: 87% on GTSRB (acceptable trade-off)
- ONNX: 15 operators, 0 BatchNorm
- Use case: Verification research, when 87% accuracy is sufficient

**NNVCompatibleCNN (Baseline):**
- Parameters: 100K
- Structure: 3 conv (32→64→64 channels) + 3 AvgPool + 1 FC
- Accuracy: 95-97% on GTSRB
- ONNX: 11 operators, 0 BatchNorm (after folding)
- Use case: High accuracy baseline, sampling-based verification

**MicroExpertCNN:**
- Parameters: 67K
- Uses MaxPool (not ideal for NNV but faster training)
- Accuracy: 95-97%
- Use case: General purpose, not optimized for NNV

### Key Files

**Python:**
- `export_to_onnx.py` - ONNX export with BatchNorm folding, supports all architectures
- `inspect_onnx.py` - Inspect ONNX structure (verify operator count, BN layers)
- `test_bn_folding.py` - Validate BatchNorm folding correctness

**MATLAB:**
- `verify_expert_nnv_simple.m` - Sampling-based verification (USE THIS)
- `verify_expert_nnv.m` - Formal verification attempt (will timeout, has automatic fallback)

**Documentation:**
- `NNV_VERIFICATION_GUIDE.md` - Comprehensive guide (setup, usage, why it fails, what to report)
- `WHY_NNV_WORKS_FOR_OTHERS.md` - Explains ACAS Xu vs our CNNs, state-of-the-art limitations
- `NNV_README.md` - Quick reference

### Gurobi Setup (Optional)

Gurobi is a commercial LP solver that NNV uses. While it doesn't solve the fundamental issues, it's faster than open-source alternatives.

**Installation:** See detailed steps in `NNV_VERIFICATION_GUIDE.md`

**Key point:** Even with Gurobi, NNV still times out on our models. The issue is network scale, not solver speed.

### For Research Papers

**What to report:**
- "We designed a verification-optimized CNN (96K params, 87% accuracy)"
- "Implemented BatchNorm folding to reduce ONNX complexity (0 BN layers)"
- "Used average pooling for LP solver compatibility"
- "Empirically verified robustness: 95% of 100 random perturbations within ε=1/255"

**Honest acknowledgment:**
- "Formal verification with NNV could not complete due to tool limitations (LP solver timeouts). This is consistent with state-of-the-art limitations for CNNs of this scale. NNV is designed for tiny fully-connected networks (e.g., ACAS Xu: 300 neurons, 13K params) while our CNN has 96K parameters with convolutional layers."

**What NOT to claim:**
- "Formally verified the model" (unless using alpha-beta-CROWN or similar)
- "Proved robustness guarantees" (sampling provides empirical evidence, not formal proofs)

### Alternative Verification Tools

If formal verification is required, consider:
1. **alpha-beta-CROWN** - **RECOMMENDED** - State-of-the-art, handles larger CNNs, won VNN-COMP (see below)
2. **ERAN** - ETH Zurich's verifier, better CNN support than NNV
3. **Marabou** - SMT-based, different approach

## Formal Verification with alpha-beta-CROWN (RECOMMENDED)

**alpha-beta-CROWN** is the state-of-the-art neural network verifier, winning VNN-COMP 2021-2024. Unlike NNV, it can handle large CNNs (millions of parameters) with formal guarantees.

### Status: FULLY FUNCTIONAL

alpha-beta-CROWN is integrated and ready to use for formal verification of both expert models and the MetaMoE router.

**Main documentation:**
- [src/Formal_Neural_Network_Verification/alpha-beta-crown/README.md](src/Formal_Neural_Network_Verification/alpha-beta-crown/README.md) - Complete consolidated guide (includes router and expert verification)
- [modules/alpha-beta-CROWN/README.md](modules/alpha-beta-CROWN/README.md) - Official alpha-beta-CROWN documentation

**Key achievements:**
- Router verification: 100% success rate on 20 test samples (10 MNIST + 10 CIFAR10)
- Average verification time: 10.82 seconds per sample at epsilon = 2/255
- Expert verification: Scalable to models with millions of parameters
- Supports configurable sample counts: 1 to 10,000 images per dataset

### How alpha-beta-CROWN Was Adapted to MoE Architecture

Standard alpha-beta-CROWN is designed for single-model verification. We made several critical adaptations to verify the **MetaMoE router** (dataset-level routing):

#### 1. Router-Only Export (Key Innovation)

**Challenge:** MetaMoE contains 3 components:
- Router (96K params): Routes images to experts
- Expert 0 (96K-1.5M params): Frozen CIFAR10 model
- Expert 1 (96K-1.5M params): Frozen MNIST model
- **Total:** 2M+ parameters (intractable for verification)

**Solution:** Extract and verify ONLY the router
- Created `export_router_to_abcrown.py` to export router as standalone ONNX
- Router input: 3x32x32 normalized image
- Router output: 2-class logits [logit_expert0, logit_expert1]
- Verification property: "argmax(logits) doesn't change within epsilon-ball"

**Implementation:**
```python
# Extract router from MetaMoE (export_router_to_abcrown.py)
class RouterOnlyWrapper(nn.Module):
    def __init__(self, meta_gating_net):
        super().__init__()
        self.router = meta_gating_net  # Just the routing network

    def forward(self, x):
        return self.router(x)  # Returns [batch, 2] logits

# Export to ONNX
router_wrapper = RouterOnlyWrapper(meta_moe.meta_gating_net)
torch.onnx.export(router_wrapper, dummy_input, onnx_path)
```

**Impact:** Reduced verification from 2M+ params to 96K params (20x smaller, verification time: ~11s per sample)

#### 2. Removed BatchNorm from Router

**Challenge:** BatchNorm has different behavior in train vs eval mode, causing verification inconsistencies

**Solution:** Designed `UltraVerifiableCNN_Features` without BatchNorm
- No BatchNorm layers in router architecture
- Increased channel capacity to compensate (20→28→40→56)
- Used Average Pooling (linear operation, easier to verify)
- Achieved 99.97% routing accuracy without BatchNorm

**Impact:** Deterministic behavior, no train/eval discrepancy

#### 3. Raw Logits Output (No Softmax)

**Challenge:** Router originally applied temperature-scaled softmax, making bound propagation harder

**Solution:** Return raw logits directly
```python
# Old router forward (harder to verify)
def forward(self, x):
    logits = self.router(x)
    probs = F.softmax(logits / self.temperature, dim=-1)
    return probs * self.num_experts  # Output sums to num_experts

# New router forward (verification-friendly)
def forward(self, x):
    logits = self.router(x)
    return logits  # Raw logits, no softmax, no scaling
```

**Impact:** Simplified verification property (just compare logits), easier for CROWN bounds

#### 4. VNNLIB Generation with Configurable Sample Counts

**Challenge:** Need to test robustness on multiple samples from both datasets (MNIST and CIFAR10)

**Solution:** Created `src/Formal_Neural_Network_Verification/alpha-beta-crown/prepare_router_verification.py` with dynamic stride sampling
- Supports 1 to 10,000 samples per dataset
- Dynamic stride: `stride = dataset_size / num_samples`
- Automatic cleanup of old VNNLIB files before generating new ones
- Unified normalization: mean=[0.295, 0.291, 0.274], std=[0.325, 0.321, 0.319]

**Usage:**
```bash
# Generate 100 MNIST + 100 CIFAR10 specifications
python src/Formal_Neural_Network_Verification/alpha-beta-crown/prepare_router_verification.py --num_mnist 100 --num_cifar 100
```

**Impact:** Flexible testing from quick tests (10 samples) to thorough verification (10,000 samples)

#### 5. Flat VNNLIB Indexing

**Challenge:** VNNLIB format expects flat variable indexing

**Solution:** Generate VNNLIB with flat indexing (X_0 to X_3071)
```python
# Flatten pixel indexing (generate_router_vnnlib.py)
pixel_idx = 0
for c in range(3):      # channels
    for h in range(32):  # height
        for w in range(32):  # width
            vnnlib += f"(declare-const X_{pixel_idx} Real)\n"
            vnnlib += f"(assert (<= X_{pixel_idx} {upper_bound}))\n"
            vnnlib += f"(assert (>= X_{pixel_idx} {lower_bound}))\n"
            pixel_idx += 1
```

**Impact:** Compatible with alpha-beta-CROWN's VNNLIB parser

#### 6. Property Negation for Counterexample Search

**Challenge:** Verification finds counterexamples, not proofs of correctness

**Solution:** Negate the desired property in VNNLIB
```python
# For MNIST sample (should route to expert 1)
# Desired property: Y_1 > Y_0  (expert 1 logit is larger)
# VNNLIB property: Y_0 >= Y_1  (negation, to find counterexamples)

if true_expert == 1:  # MNIST
    vnnlib += "(assert (>= Y_0 Y_1))\n"  # Try to find Y_0 >= Y_1
else:  # CIFAR10
    vnnlib += "(assert (>= Y_1 Y_0))\n"  # Try to find Y_1 >= Y_0
```

**Logic:**
- If verifier finds NO counterexample → Property holds → Verified
- If verifier finds counterexample → Property falsified

**Impact:** Correct verification semantics for alpha-beta-CROWN

#### 7. End-to-End Verification Workflow

**Challenge:** Manual workflow (export ONNX, generate VNNLIB, run verification) is error-prone

**Solution:** Created `verify_all_router_samples.py` for one-command verification
```bash
# Complete workflow: .pth → ONNX → VNNLIB → Verification → Report
python verify_all_router_samples.py \
    --model_path artifacts/meta_moe_ultra_verifiable_cnn_best_og.pth \
    --num_mnist 100 --num_cifar 100
```

**What it does:**
1. Auto-exports router from .pth to ONNX (if not already exported)
2. Cleans up old VNNLIB specifications
3. Generates fresh VNNLIB specs for both datasets
4. Runs alpha-beta-CROWN verification on all samples
5. Generates comprehensive report: `artifacts/router_verification_results.txt`

**Impact:** 1-command workflow, no manual steps, automatic cleanup

#### Summary of MoE Adaptations

| Adaptation | Purpose | Impact |
|------------|---------|--------|
| Router-only export | Reduce model size | 20x smaller (2M → 96K params) |
| No BatchNorm | Deterministic behavior | Eliminated train/eval discrepancy |
| Raw logits output | Simpler bounds | Easier CROWN propagation |
| Configurable samples | Flexible testing | 1-10,000 samples per dataset |
| Flat VNNLIB indexing | Parser compatibility | Works with alpha-beta-CROWN |
| Property negation | Counterexample search | Correct verification semantics |
| Unified normalization | Consistent input | No distribution mismatch |
| End-to-end workflow | Automation | 1-command verification |

**Result:** 100% verification success rate on 20 samples, average 10.82s per sample, scalable to 10,000 samples

### Quick Start - Router Verification

**One-command verification (recommended):**

```bash
# Default: 10 MNIST + 10 CIFAR10 samples (~2 minutes)
python verify_all_router_samples.py \
    --model_path artifacts/meta_moe_ultra_verifiable_cnn_best_og.pth

# Medium verification: 100+100 samples (~30 minutes, good for papers)
python verify_all_router_samples.py \
    --model_path artifacts/meta_moe_ultra_verifiable_cnn_best_og.pth \
    --num_mnist 100 --num_cifar 100

# Thorough verification: 200+200 samples (~1 hour)
python verify_all_router_samples.py \
    --model_path artifacts/meta_moe_ultra_verifiable_cnn_best_og.pth \
    --num_mnist 200 --num_cifar 200 \
    --timeout 120

# Or use existing ONNX file
python verify_all_router_samples.py \
    --onnx_path artifacts/abcrown_models/my_router_only.onnx \
    --num_mnist 50 --num_cifar 50
```

**Expected output:**
```
================================================================================
VERIFICATION SUMMARY
================================================================================
Total samples: 20
  Verified:   20 (100.0%)
  Falsified:  0 (0.0%)
  Timeout:    0 (0.0%)
  Unknown:    0 (0.0%)

Total time: 216.45 seconds
Average time per sample: 10.82 seconds
================================================================================
```

### Quick Start - Expert Verification

**Verify a trained expert model:**

```bash
# Automated verification (recommended)
python src/Formal_Neural_Network_Verification/verify_expert_abcrown.py \
    --model_path artifacts/gtsrb_ultra_verifiable_cnn_best.pth \
    --dataset GTSRB \
    --epsilon 0.00784 \
    --num_images 10 \
    --timeout 300
```

**Manual workflow:**

```bash
# Step 1: Export model to ONNX
python src/Formal_Neural_Network_Verification/export_to_abcrown.py \
    --model_path artifacts/gtsrb_ultra_verifiable_cnn_best.pth \
    --output_dir artifacts/abcrown_models

# Step 2: Run verification
cd modules/alpha-beta-CROWN/complete_verifier
python abcrown.py --config exp_configs/moe_experts/gtsrb_expert_linf.yaml
```

### Key Features

1. **Scales to large models:**
   - Handles CNNs with millions of parameters
   - GPU-accelerated (much faster than NNV)
   - Supports ResNet, ConvNeXt, Transformers

2. **Formal guarantees:**
   - Provides provable robustness certificates
   - State-of-the-art bound propagation (α-CROWN, β-CROWN)
   - Complete and incomplete verification modes

3. **Easy to use:**
   - ONNX input format
   - YAML configuration files
   - Automatic BatchNorm folding

### Verification Results Interpretation

alpha-beta-CROWN outputs:
- **Verified**: Property holds, model is provably robust
- **Falsified**: Counterexample found (adversarial example exists)
- **Timeout**: Verification incomplete within time limit
- **Unknown**: Could not determine status

Example output:
```
Total images: 10
Verified: 7 (70%)    # Provably robust
Falsified: 2 (20%)   # Adversarial examples found
Timeout: 1 (10%)     # Unknown status
```

### Configuration Files

Pre-configured verification settings are available in:
- [modules/alpha-beta-CROWN/complete_verifier/exp_configs/moe_experts/gtsrb_expert_linf.yaml](modules/alpha-beta-CROWN/complete_verifier/exp_configs/moe_experts/gtsrb_expert_linf.yaml)
- [modules/alpha-beta-CROWN/complete_verifier/exp_configs/moe_experts/cifar10_expert_linf.yaml](modules/alpha-beta-CROWN/complete_verifier/exp_configs/moe_experts/cifar10_expert_linf.yaml)

### Key Settings

**Epsilon (perturbation bound):**
- 2/255 = 0.00784 (standard robustness test)
- 4/255 = 0.01569 (larger perturbation)
- 8/255 = 0.03137 (very large perturbation)

**Timeout:**
- 60s: Quick test
- 300s: Standard (recommended)
- 600s: Thorough verification

**Branching method** (`bab.branching.method`):
- `kfsb`: Default, balanced speed/accuracy
- `fsb`: Most accurate, slower
- `babsr`: Fastest, less accurate

### Advantages Over NNV

| Feature | alpha-beta-CROWN | NNV |
|---------|------------------|-----|
| **Model Size** | Millions of params | ~100K params max |
| **Speed** | GPU-accelerated | CPU-only, slow |
| **Success Rate** | High (VNN-COMP winner) | Low for CNNs |
| **Architecture Support** | CNN, ResNet, Transformers | Fully-connected, simple CNN |
| **Ease of Use** | ONNX + YAML | MATLAB + .mat conversion |

### For Research Papers

**What to report:**
- "Formally verified X% of test images using alpha-beta-CROWN (VNN-COMP 2021-2024 winner)"
- "Provable robustness within ε=2/255 L∞ perturbation for Y% of images"
- "Verification completed with Z seconds average timeout per image"

**Citation:**
```bibtex
@article{wang2021beta,
  title={{Beta-CROWN}: Efficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

### Troubleshooting

**Issue: "ConvertModel.__init__() got an unexpected keyword argument 'quirks'" on Ubuntu/Linux**

Cause: Wrong `onnx2pytorch` version. alpha-beta-CROWN requires a custom fork from Verified-Intelligence.

Solution:
```bash
pip uninstall onnx2pytorch -y
pip install git+https://github.com/Verified-Intelligence/onnx2pytorch.git
```

**Issue: "No module named 'auto_LiRPA'"**

Solution:
```bash
cd modules/alpha-beta-CROWN/complete_verifier
export PYTHONPATH="../auto_LiRPA:$PYTHONPATH"  # Linux/Mac
set PYTHONPATH=..\auto_LiRPA;%PYTHONPATH%      # Windows
```

**Issue: GPU out of memory**

Solution: Reduce batch size in config:
```yaml
solver:
  batch_size: 512  # Reduce from 1024
```

**Issue: Verification timeout**

Solutions:
- Increase timeout: `--timeout 600`
- Use faster branching: `bab.branching.method: babsr`
- Verify fewer images: `--num_images 5`

### Key Files

**Router verification (main scripts):**
- `verify_all_router_samples.py` - **[RECOMMENDED]** Complete end-to-end router verification (.pth → ONNX → VNNLIB → verification → report)
- `src/Formal_Neural_Network_Verification/alpha-beta-crown/prepare_router_verification.py` - Generate VNNLIB specifications with configurable sample counts (1-10,000 per dataset)
- `src/Formal_Neural_Network_Verification/alpha-beta-crown/export_router_to_abcrown.py` - Export router-only ONNX from MetaMoE
- `src/Formal_Neural_Network_Verification/alpha-beta-crown/generate_router_vnnlib.py` - Generate VNNLIB specs for router verification

**Expert verification:**
- `src/Formal_Neural_Network_Verification/verify_expert_abcrown.py` - High-level expert verification interface
- `src/Formal_Neural_Network_Verification/export_to_abcrown.py` - Export PyTorch to ONNX with BatchNorm folding

**Configuration templates:**
- `modules/alpha-beta-CROWN/complete_verifier/exp_configs/moe_experts/` - Verification configs for expert and router models

**Documentation:**
- `src/Formal_Neural_Network_Verification/alpha-beta-crown/README.md` - Complete consolidated guide (router + expert verification)
- `modules/alpha-beta-CROWN/README.md` - Official alpha-beta-CROWN documentation

### Architecture Comparison

| Model | Params | Accuracy | ONNX Ops | NNV Layers | Verification |
|-------|--------|----------|----------|------------|--------------|
| UltraVerifiableCNN | 96K | 87% | 15 | 17 | Sampling only |
| NNVCompatibleCNN | 100K | 95% | 11 | 13 | Sampling only |
| MicroExpertCNN | 67K | 95% | 11 | 13 | Sampling only |

**Note:** All models use sampling-based verification. Formal verification fails for all due to NNV tool limitations.

### Important Notes

1. **NNV is for tiny networks:** Designed for ACAS Xu (300 neurons, fully-connected)
2. **Our models are beyond state-of-the-art:** 96K params with convolutions is too large for current tools
3. **Sampling is scientifically valid:** Widely used in ICML/NeurIPS/ICLR papers
4. **Our optimizations still matter:** BatchNorm folding, architecture design useful for future tools
5. **Success rate on ACAS Xu:** Even tiny networks only verify 64% of the time

**Bottom line:** Use sampling-based verification (`verify_expert_nnv_simple.m`) and report empirical robustness in papers.
