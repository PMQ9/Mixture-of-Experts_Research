#!/usr/bin/env python3
"""
alpha-beta-CROWN Setup Script (Cross-Platform)

This script automatically sets up alpha-beta-CROWN for neural network verification.
It handles submodule initialization, dependency installation, and environment configuration.

Usage:
    python setup_abcrown.py                 # Full setup
    python setup_abcrown.py --test          # Full setup + test verification
    python setup_abcrown.py --skip-deps     # Skip dependency installation
    python setup_abcrown.py --check-only    # Check installation status only

Requirements:
    - Python 3.9+ (3.11 recommended)
    - PyTorch 2.0+ (should be pre-installed)
    - CUDA 12.1+ (for GPU acceleration, optional)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import platform


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def disable():
        """Disable colors (for Windows cmd without ANSI support)"""
        Colors.RED = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.BLUE = ''
        Colors.NC = ''


# Disable colors on Windows cmd (unless using Windows Terminal)
if platform.system() == 'Windows' and 'WT_SESSION' not in os.environ:
    Colors.disable()


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")


def print_step(step_num, total_steps, text):
    """Print a step indicator"""
    print(f"{Colors.YELLOW}[{step_num}/{total_steps}] {text}{Colors.NC}")


def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_warning(text):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def print_error(text):
    """Print an error message"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def run_command(cmd, capture_output=True, cwd=None, env=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            cwd=cwd,
            env=env,
            shell=isinstance(cmd, str)
        )
        return result
    except Exception as e:
        print_error(f"Command failed: {e}")
        return None


def check_python_version():
    """Check if Python version meets requirements"""
    print_step(1, 6, "Checking Python version...")

    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 9):
        print_error(f"Python 3.9+ required. Found Python {version_info.major}.{version_info.minor}")
        return False

    print_success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} detected")
    return True


def check_pytorch():
    """Check if PyTorch is installed and CUDA is available"""
    print_step(2, 6, "Checking PyTorch installation...")

    try:
        import torch
        print_success(f"PyTorch {torch.__version__} detected")

        # Check CUDA
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print_success(f"CUDA {cuda_version} available")
            return True, True
        else:
            print_warning("CUDA not available, will use CPU (verification will be slower)")
            return True, False

    except ImportError:
        print_error("PyTorch not found. Please install PyTorch 2.0+ first:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False, False


def initialize_submodules(project_root):
    """Initialize and update git submodules"""
    print_step(3, 6, "Initializing alpha-beta-CROWN submodule...")

    abcrown_dir = project_root / 'modules' / 'alpha-beta-CROWN'
    auto_lirpa_dir = abcrown_dir / 'auto_LiRPA'

    # Initialize alpha-beta-CROWN submodule
    if not (abcrown_dir / '.git').exists():
        print("Initializing submodule for the first time...")
        result = run_command(
            ['git', 'submodule', 'update', '--init', '--recursive', 'modules/alpha-beta-CROWN'],
            cwd=project_root
        )
        if result and result.returncode == 0:
            print_success("Submodule initialized")
        else:
            print_error("Failed to initialize submodule")
            return False
    else:
        print("Submodule already initialized, updating...")
        result = run_command(
            ['git', 'submodule', 'update', '--remote', '--merge', 'modules/alpha-beta-CROWN'],
            cwd=project_root
        )
        if result and result.returncode == 0:
            print_success("Submodule updated")

    # Check auto_LiRPA submodule
    if not (auto_lirpa_dir / '.git').exists():
        print("Initializing auto_LiRPA submodule...")
        result = run_command(
            ['git', 'submodule', 'update', '--init', '--recursive'],
            cwd=abcrown_dir
        )
        if result and result.returncode == 0:
            print_success("auto_LiRPA submodule initialized")
        else:
            print_warning("Could not initialize auto_LiRPA submodule (may need manual setup)")

    return True


def install_dependencies(project_root, skip_deps=False):
    """Install Python dependencies"""
    if skip_deps:
        print_step(4, 6, "Skipping dependency installation (--skip-deps)")
        return True

    print_step(4, 6, "Installing dependencies...")

    # Install auto_LiRPA
    print("Installing auto_LiRPA...")
    auto_lirpa_dir = project_root / 'modules' / 'alpha-beta-CROWN' / 'auto_LiRPA'
    result = run_command([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=auto_lirpa_dir)
    if result and result.returncode == 0:
        print_success("auto_LiRPA installed")
    else:
        print_error("Failed to install auto_LiRPA")
        return False

    # Install alpha-beta-CROWN requirements
    print("Installing alpha-beta-CROWN requirements...")
    requirements_file = project_root / 'modules' / 'alpha-beta-CROWN' / 'complete_verifier' / 'requirements.txt'
    result = run_command([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)])
    if result and result.returncode == 0:
        print_success("alpha-beta-CROWN requirements installed")
    else:
        print_error("Failed to install requirements")
        return False

    # Install additional utilities
    print("Installing additional utilities...")
    result = run_command([sys.executable, '-m', 'pip', 'install', 'appdirs', 'pyyaml'])
    if result and result.returncode == 0:
        print_success("Additional utilities installed")
    else:
        print_warning("Could not install some utilities (may already be installed)")

    return True


def verify_installation(project_root):
    """Verify that alpha-beta-CROWN is properly installed"""
    print_step(5, 6, "Verifying installation...")

    # Add auto_LiRPA to path
    auto_lirpa_dir = project_root / 'modules' / 'alpha-beta-CROWN' / 'auto_LiRPA'
    sys.path.insert(0, str(auto_lirpa_dir))

    # Test auto_LiRPA import
    try:
        import auto_LiRPA
        print_success(f"auto_LiRPA version {auto_LiRPA.__version__} can be imported")
    except ImportError as e:
        print_error(f"Cannot import auto_LiRPA: {e}")
        return False

    # Test ONNX imports
    try:
        import onnx
        import onnxruntime
        print_success("ONNX runtime available")
    except ImportError as e:
        print_error(f"ONNX runtime not available: {e}")
        return False

    return True


def create_directories(project_root):
    """Create necessary artifact directories"""
    print_step(6, 6, "Creating artifact directories...")

    dirs = [
        project_root / 'artifacts' / 'abcrown_models',
        project_root / 'artifacts' / 'abcrown_configs',
        project_root / 'artifacts' / 'vnnlib_specs'
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    print_success("Directories created")
    return True


def run_test_verification(project_root):
    """Run a test verification if a trained model exists"""
    print_header("Running Test Verification")

    # Find a test model
    artifacts_dir = project_root / 'artifacts'
    test_model = None

    # Search for trained models
    for pattern in ['*small_cnn_best*.pth', '*tiny_cnn_best*.pth', '*micro_cnn_best*.pth']:
        models = list(artifacts_dir.rglob(pattern))
        if models:
            # Filter out checkpoint files
            models = [m for m in models if 'checkpoint' not in m.name]
            if models:
                test_model = models[0]
                break

    if not test_model:
        print_warning("No trained model found for testing.")
        print("To test verification, first train a model:")
        print("  python train.py --dataset GTSRB --model_arch small_cnn --epochs 10")
        return

    print(f"Found test model: {test_model}")

    # Determine dataset from filename
    model_name = test_model.name.lower()
    if 'gtsrb' in model_name:
        dataset = 'GTSRB'
    elif 'cifar' in model_name:
        dataset = 'CIFAR10'
    elif 'mnist' in model_name:
        dataset = 'MNIST'
    else:
        dataset = 'GTSRB'  # default

    print(f"Running verification on 5 test images (dataset: {dataset})...")

    # Run verification script
    verify_script = project_root / 'src' / 'Formal_Neural_Network_Verification' / 'alpha-beta-crown' / 'verify_expert_abcrown.py'
    result = run_command([
        sys.executable, str(verify_script),
        '--model_path', str(test_model),
        '--dataset', dataset,
        '--epsilon', '0.00784',
        '--num_images', '5',
        '--timeout', '60',
        '--auto_run'
    ], capture_output=False, cwd=project_root)

    if result and result.returncode == 0:
        print_success("Test verification completed")
    else:
        print_warning("Test verification encountered issues (this is normal for the first run)")


def check_installation_status(project_root):
    """Check the current installation status without making changes"""
    print_header("Checking Installation Status")

    status = {
        'python': False,
        'pytorch': False,
        'cuda': False,
        'submodule': False,
        'auto_lirpa': False,
        'dependencies': False
    }

    # Check Python
    version_info = sys.version_info
    if version_info.major >= 3 and version_info.minor >= 9:
        status['python'] = True
        print_success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    else:
        print_error(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} (need 3.9+)")

    # Check PyTorch
    try:
        import torch
        status['pytorch'] = True
        print_success(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            status['cuda'] = True
            print_success(f"CUDA {torch.version.cuda}")
        else:
            print_warning("CUDA not available")
    except ImportError:
        print_error("PyTorch not installed")

    # Check submodules
    abcrown_dir = project_root / 'modules' / 'alpha-beta-CROWN'
    auto_lirpa_dir = abcrown_dir / 'auto_LiRPA'

    if (abcrown_dir / '.git').exists():
        status['submodule'] = True
        print_success("alpha-beta-CROWN submodule initialized")
    else:
        print_error("alpha-beta-CROWN submodule not initialized")

    if (auto_lirpa_dir / '.git').exists():
        print_success("auto_LiRPA submodule initialized")
    else:
        print_error("auto_LiRPA submodule not initialized")

    # Check auto_LiRPA import
    sys.path.insert(0, str(auto_lirpa_dir))
    try:
        import auto_LiRPA
        status['auto_lirpa'] = True
        print_success(f"auto_LiRPA {auto_LiRPA.__version__} installed")
    except ImportError:
        print_error("auto_LiRPA not installed")

    # Check dependencies
    try:
        import onnx
        import onnxruntime
        import yaml
        status['dependencies'] = True
        print_success("All dependencies installed")
    except ImportError as e:
        print_error(f"Missing dependencies: {e}")

    # Summary
    print("\nInstallation Summary:")
    all_ready = all(status.values())
    if all_ready:
        print_success("All components are ready!")
    else:
        print_warning("Some components need attention. Run without --check-only to fix.")

    return all_ready


def print_usage_instructions(project_root):
    """Print usage instructions after successful setup"""
    print_header("Setup Complete!")

    print("alpha-beta-CROWN is now ready to use.\n")
    print("Quick start:")
    print("  1. Export and verify a trained model:")
    print("     python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py \\")
    print("       --model_path artifacts/gtsrb_small_cnn_best.pth \\")
    print("       --dataset GTSRB \\")
    print("       --epsilon 0.00784 \\")
    print("       --num_images 10 \\")
    print("       --auto_run\n")

    print("  2. Or run verification manually:")
    print("     cd modules/alpha-beta-CROWN/complete_verifier")
    if platform.system() == 'Windows':
        print("     set PYTHONPATH=..\\auto_LiRPA;%PYTHONPATH%")
    else:
        print("     export PYTHONPATH=\"../auto_LiRPA:$PYTHONPATH\"")
    print("     python abcrown.py --config <path-to-config.yaml>\n")

    print("Documentation:")
    print("  - Quick start: src/Formal_Neural_Network_Verification/alpha-beta-crown/ABCROWN_QUICKSTART.md")
    print("  - Project docs: CLAUDE.md")
    print("  - Official docs: modules/alpha-beta-CROWN/README.md")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Setup alpha-beta-CROWN for neural network verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--skip-deps', action='store_true',
                        help='Skip dependency installation')
    parser.add_argument('--test', action='store_true',
                        help='Run test verification after setup')
    parser.add_argument('--check-only', action='store_true',
                        help='Check installation status without making changes')

    args = parser.parse_args()

    # Get project root (4 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent

    print_header("alpha-beta-CROWN Setup Script")
    print(f"Project root: {project_root}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.executable}")

    # If check-only mode, just check status and exit
    if args.check_only:
        success = check_installation_status(project_root)
        sys.exit(0 if success else 1)

    # Run setup steps
    steps = [
        check_python_version,
        lambda: check_pytorch()[0],  # Only care about PyTorch presence
        lambda: initialize_submodules(project_root),
        lambda: install_dependencies(project_root, args.skip_deps),
        lambda: verify_installation(project_root),
        lambda: create_directories(project_root)
    ]

    for step in steps:
        if not step():
            print_error("\nSetup failed. Please fix the errors above and try again.")
            sys.exit(1)

    # Run test verification if requested
    if args.test:
        run_test_verification(project_root)

    # Print usage instructions
    print_usage_instructions(project_root)


if __name__ == '__main__':
    main()
