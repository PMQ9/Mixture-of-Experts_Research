# Formal Neural Network Verification

This directory contains tools and scripts for formal verification of the Mixture-of-Experts models.

## Directory Structure

### alpha-beta-crown/
State-of-the-art neural network verifier (VNN-COMP 2021-2024 winner).

**Status:** FULLY FUNCTIONAL

**Use for:**
- Router verification (100% success rate achieved)
- Expert verification (scales to millions of parameters)
- Formal robustness certificates

**Documentation:**
- [FORMAL_VERIFICATION_GUIDE.md](alpha-beta-crown/FORMAL_VERIFICATION_GUIDE.md) - Complete guide for router and expert verification

**Quick Start:**
```bash
# Verify MetaMoE router
python run_router_formal_verification.py --dataset BOTH

# Verify expert model
python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py \
    --model_path artifacts/gtsrb_small_cnn_best.pth \
    --dataset GTSRB \
    --epsilon 0.00784 \
    --num_images 10
```

### nnv/
NNV (Neural Network Verification) MATLAB-based tools.

**Status:** LIMITED - Formal verification fails due to LP solver limitations with CNNs

**Use for:**
- Sampling-based robustness testing (empirical, not formal)
- ONNX export to .mat format for MATLAB tools

**Documentation:**
- [nnv/README.md](nnv/README.md) - NNV setup and usage

**Note:** NNV is designed for small fully-connected networks (e.g., ACAS Xu with 300 neurons). Our CNNs (67K-1.5M parameters) are beyond NNV's capabilities. Use alpha-beta-CROWN for formal verification.

## Recommended Workflow

1. **Train model** with verification-optimized architecture:
   ```bash
   python train.py --dataset GTSRB --model_arch ultra_verifiable_cnn --epochs 100
   ```

2. **Verify with alpha-beta-CROWN**:
   ```bash
   python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py \
       --model_path artifacts/gtsrb_ultra_verifiable_cnn_best.pth \
       --dataset GTSRB
   ```

3. **Report results** in research paper (see FORMAL_VERIFICATION_GUIDE.md for citation and reporting guidelines)

## Key Files

**alpha-beta-crown/**
- `FORMAL_VERIFICATION_GUIDE.md` - Complete verification documentation
- `export_router_to_abcrown.py` - Export router to ONNX for verification
- `generate_router_vnnlib.py` - Generate VNNLIB specifications for router
- `setup_abcrown.py` - One-command setup script

**nnv/** (archived, not recommended)
- `verify_expert_nnv_simple.m` - Sampling-based robustness testing (MATLAB)
- `export_to_onnx.py` - Export models to ONNX (NNV-specific)
- `File_Conversion/` - PyTorch/ONNX to MATLAB .mat conversion
- `Gurobi_LP_Solver/` - Gurobi LP solver setup (optional)

## System Requirements

- Python 3.10+
- PyTorch with CUDA
- alpha-beta-CROWN (included as git submodule)
- CUDA-compatible GPU (recommended)

For NNV (optional):
- MATLAB R2021b or later
- NNV toolbox (included as git submodule)

## Citation

For alpha-beta-CROWN:
```bibtex
@article{wang2021beta,
  title={{Beta-CROWN}: Efficient bound propagation with per-neuron split constraints for complete and incomplete neural network verification},
  author={Wang, Shiqi and Zhang, Huan and Xu, Kaidi and Lin, Xue and Jana, Suman and Hsieh, Cho-Jui and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Summary

- **For formal verification:** Use alpha-beta-CROWN (fully functional, proven results)
- **For empirical testing:** Use NNV sampling-based tests (backup method only)
- **For research papers:** See FORMAL_VERIFICATION_GUIDE.md for reporting guidelines

---

**Maintained by:** Institute of Software Integrated Systems, Vanderbilt University
**Last Updated:** October 2025
