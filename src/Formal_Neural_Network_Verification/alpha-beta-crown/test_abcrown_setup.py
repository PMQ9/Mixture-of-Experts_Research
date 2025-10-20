"""
Test alpha-beta-CROWN Setup

This script verifies that alpha-beta-CROWN is properly installed and configured.
Run this before attempting verification to ensure everything works.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages are installed"""
    print("\n" + "="*80)
    print("Testing Package Imports")
    print("="*80)

    packages = {
        'torch': 'PyTorch',
        'onnx': 'ONNX',
        'onnxruntime': 'ONNX Runtime',
        'onnxsim': 'ONNX Simplifier',
        'yaml': 'PyYAML',
        'numpy': 'NumPy',
    }

    all_passed = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  [OK] {name} ({package})")
        except ImportError:
            print(f"  [FAIL] {name} ({package}) - NOT INSTALLED")
            all_passed = False

    return all_passed


def test_auto_lirpa():
    """Test that auto_LiRPA can be imported"""
    print("\n" + "="*80)
    print("Testing auto_LiRPA")
    print("="*80)

    try:
        sys.path.insert(0, str(Path(__file__).parent / 'modules' / 'alpha-beta-CROWN' / 'auto_LiRPA'))
        import auto_LiRPA
        print(f"  [OK] auto_LiRPA loaded successfully")
        print(f"  Version: {auto_LiRPA.__version__}")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to load auto_LiRPA: {e}")
        return False


def test_abcrown_structure():
    """Test that alpha-beta-CROWN directory structure is correct"""
    print("\n" + "="*80)
    print("Testing alpha-beta-CROWN Directory Structure")
    print("="*80)

    base_dir = Path(__file__).parent / 'modules' / 'alpha-beta-CROWN'

    required_paths = {
        'complete_verifier': 'Complete verifier directory',
        'complete_verifier/abcrown.py': 'Main verification script',
        'auto_LiRPA': 'auto_LiRPA submodule',
        'auto_LiRPA/auto_LiRPA/__init__.py': 'auto_LiRPA package',
    }

    all_passed = True
    for path, description in required_paths.items():
        full_path = base_dir / path
        if full_path.exists():
            print(f"  [OK] {description}")
        else:
            print(f"  [FAIL] {description} - NOT FOUND: {full_path}")
            all_passed = False

    return all_passed


def test_verification_scripts():
    """Test that verification scripts exist"""
    print("\n" + "="*80)
    print("Testing Verification Scripts")
    print("="*80)

    base_dir = Path(__file__).parent / 'src' / 'Formal_Neural_Network_Verification'

    required_scripts = {
        'export_to_abcrown.py': 'ONNX export script',
        'verify_expert_abcrown.py': 'Verification script',
        'ABCROWN_QUICKSTART.md': 'Quick start guide',
    }

    all_passed = True
    for script, description in required_scripts.items():
        full_path = base_dir / script
        if full_path.exists():
            print(f"  [OK] {description}")
        else:
            print(f"  [FAIL] {description} - NOT FOUND: {full_path}")
            all_passed = False

    return all_passed


def test_config_files():
    """Test that config files exist"""
    print("\n" + "="*80)
    print("Testing Configuration Files")
    print("="*80)

    base_dir = Path(__file__).parent / 'modules' / 'alpha-beta-CROWN' / 'complete_verifier' / 'exp_configs' / 'moe_experts'

    required_configs = {
        'gtsrb_expert_linf.yaml': 'GTSRB verification config',
        'cifar10_expert_linf.yaml': 'CIFAR-10 verification config',
    }

    all_passed = True
    for config, description in required_configs.items():
        full_path = base_dir / config
        if full_path.exists():
            print(f"  [OK] {description}")
        else:
            print(f"  [FAIL] {description} - NOT FOUND: {full_path}")
            all_passed = False

    return all_passed


def test_expert_models():
    """Check if any expert models are available"""
    print("\n" + "="*80)
    print("Checking for Expert Models")
    print("="*80)

    artifacts_dir = Path(__file__).parent / 'artifacts'

    if not artifacts_dir.exists():
        print(f"  [WARN] No artifacts directory found")
        print(f"  You'll need to train a model first:")
        print(f"    python train.py --dataset GTSRB --model_arch small_cnn --epochs 100")
        return False

    # Look for .pth files
    pth_files = list(artifacts_dir.rglob('*.pth'))

    if pth_files:
        print(f"  [OK] Found {len(pth_files)} model file(s):")
        for pth_file in pth_files[:5]:  # Show first 5
            print(f"    - {pth_file.relative_to(Path.cwd())}")
        if len(pth_files) > 5:
            print(f"    ... and {len(pth_files) - 5} more")
        return True
    else:
        print(f"  [WARN] No .pth model files found in {artifacts_dir}")
        print(f"  You'll need to train a model first:")
        print(f"    python train.py --dataset GTSRB --model_arch small_cnn --epochs 100")
        return False


def main():
    print("\n" + "="*80)
    print("alpha-beta-CROWN Setup Verification")
    print("="*80)

    results = {}
    results['imports'] = test_imports()
    results['auto_lirpa'] = test_auto_lirpa()
    results['structure'] = test_abcrown_structure()
    results['scripts'] = test_verification_scripts()
    results['configs'] = test_config_files()
    results['models'] = test_expert_models()

    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    for test_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {test_name.upper()}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*80)
    if all_passed:
        print("[SUCCESS] All tests PASSED - alpha-beta-CROWN is ready to use!")
        print("\nNext steps:")
        print("  1. Train a model (if you don't have one):")
        print("     python train.py --dataset GTSRB --model_arch small_cnn --epochs 100")
        print("\n  2. Verify the model:")
        print("     python src/Formal_Neural_Network_Verification/verify_expert_abcrown.py \\")
        print("         --model_path artifacts/gtsrb_small_cnn_best.pth \\")
        print("         --dataset GTSRB")
    else:
        print("[FAILED] Some tests FAILED - please fix the issues above")
        print("\nTo fix:")
        print("  1. Install missing packages: pip install <package>")
        print("  2. Initialize submodules: git submodule update --init --recursive")
        print("  3. Check file paths and permissions")

    print("="*80 + "\n")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
