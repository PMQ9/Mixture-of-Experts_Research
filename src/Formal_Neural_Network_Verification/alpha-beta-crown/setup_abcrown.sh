#!/bin/bash
# alpha-beta-CROWN Setup Script (Linux/Mac)
#
# This script automatically sets up alpha-beta-CROWN for neural network verification.
# It handles submodule initialization, dependency installation, and environment configuration.
#
# Usage:
#   bash setup_abcrown.sh                 # Full setup
#   bash setup_abcrown.sh --test          # Full setup + test verification
#   bash setup_abcrown.sh --skip-deps     # Skip dependency installation
#
# Requirements:
#   - Python 3.9+ (3.11 recommended)
#   - PyTorch 2.0+ (should be pre-installed)
#   - CUDA 12.1+ (for GPU acceleration, optional)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_DEPS=false
RUN_TEST=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-deps] [--test]"
            exit 1
            ;;
    esac
done

# Get project root (4 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}alpha-beta-CROWN Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    echo -e "${RED}Error: Python 3.9+ required. Found Python $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Check PyTorch installation
echo -e "${YELLOW}[2/6] Checking PyTorch installation...${NC}"
if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓ PyTorch $TORCH_VERSION detected${NC}"

    # Check CUDA availability
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        echo -e "${GREEN}✓ CUDA $CUDA_VERSION available${NC}"
    else
        echo -e "${YELLOW}⚠ CUDA not available, will use CPU (verification will be slower)${NC}"
    fi
else
    echo -e "${RED}Error: PyTorch not found. Please install PyTorch 2.0+ first:${NC}"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

# Initialize/update git submodules
echo -e "${YELLOW}[3/6] Initializing alpha-beta-CROWN submodule...${NC}"
cd "$PROJECT_ROOT"
if [ ! -d "modules/alpha-beta-CROWN/.git" ]; then
    echo "Initializing submodule for the first time..."
    git submodule update --init --recursive modules/alpha-beta-CROWN
    echo -e "${GREEN}✓ Submodule initialized${NC}"
else
    echo "Submodule already initialized, updating..."
    git submodule update --remote --merge modules/alpha-beta-CROWN
    echo -e "${GREEN}✓ Submodule updated${NC}"
fi

# Check that auto_LiRPA submodule is also initialized
if [ ! -d "modules/alpha-beta-CROWN/auto_LiRPA/.git" ]; then
    echo "Initializing auto_LiRPA submodule..."
    cd "$PROJECT_ROOT/modules/alpha-beta-CROWN"
    git submodule update --init --recursive
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✓ auto_LiRPA submodule initialized${NC}"
fi

# Install dependencies
if [ "$SKIP_DEPS" = false ]; then
    echo -e "${YELLOW}[4/6] Installing dependencies...${NC}"

    # Install auto_LiRPA in development mode
    echo "Installing auto_LiRPA..."
    cd "$PROJECT_ROOT/modules/alpha-beta-CROWN/auto_LiRPA"
    pip install -e . --quiet
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✓ auto_LiRPA installed${NC}"

    # Install alpha-beta-CROWN requirements
    echo "Installing alpha-beta-CROWN requirements..."
    pip install -r "$PROJECT_ROOT/modules/alpha-beta-CROWN/complete_verifier/requirements.txt" --quiet
    echo -e "${GREEN}✓ alpha-beta-CROWN requirements installed${NC}"

    # Install additional utilities
    echo "Installing additional utilities..."
    pip install appdirs pyyaml --quiet
    echo -e "${GREEN}✓ Additional utilities installed${NC}"
else
    echo -e "${YELLOW}[4/6] Skipping dependency installation (--skip-deps)${NC}"
fi

# Verify installation
echo -e "${YELLOW}[5/6] Verifying installation...${NC}"
cd "$PROJECT_ROOT/modules/alpha-beta-CROWN/complete_verifier"
export PYTHONPATH="../auto_LiRPA:$PYTHONPATH"

if python3 -c "import sys; sys.path.insert(0, '../auto_LiRPA'); import auto_LiRPA; print(f'✓ auto_LiRPA version {auto_LiRPA.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓ auto_LiRPA can be imported${NC}"
else
    echo -e "${RED}Error: Cannot import auto_LiRPA${NC}"
    exit 1
fi

if python3 -c "import onnx, onnxruntime; print('✓ ONNX runtime available')" 2>/dev/null; then
    echo -e "${GREEN}✓ ONNX runtime available${NC}"
else
    echo -e "${RED}Error: ONNX runtime not available${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}[6/6] Creating artifact directories...${NC}"
cd "$PROJECT_ROOT"
mkdir -p artifacts/abcrown_models
mkdir -p artifacts/abcrown_configs
mkdir -p artifacts/vnnlib_specs
echo -e "${GREEN}✓ Directories created${NC}"

# Run test verification if requested
if [ "$RUN_TEST" = true ]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running Test Verification${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Check if we have a test model
    cd "$PROJECT_ROOT"
    TEST_MODEL=$(find artifacts -name "*small_cnn_best*.pth" -o -name "*tiny_cnn_best*.pth" | head -n 1)

    if [ -z "$TEST_MODEL" ]; then
        echo -e "${YELLOW}No trained model found for testing.${NC}"
        echo "To test verification, first train a model:"
        echo "  python train.py --dataset GTSRB --model_arch small_cnn --epochs 10"
    else
        echo "Found test model: $TEST_MODEL"

        # Determine dataset from filename
        if [[ "$TEST_MODEL" == *"gtsrb"* ]]; then
            DATASET="GTSRB"
        elif [[ "$TEST_MODEL" == *"cifar"* ]]; then
            DATASET="CIFAR10"
        elif [[ "$TEST_MODEL" == *"mnist"* ]]; then
            DATASET="MNIST"
        else
            DATASET="GTSRB"  # default
        fi

        echo "Running verification on 5 test images..."
        python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py \
            --model_path "$TEST_MODEL" \
            --dataset "$DATASET" \
            --epsilon 0.00784 \
            --num_images 5 \
            --timeout 60 \
            --auto_run
    fi
fi

# Print success message and instructions
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "alpha-beta-CROWN is now ready to use."
echo ""
echo "Quick start:"
echo "  1. Export and verify a trained model:"
echo "     python src/Formal_Neural_Network_Verification/alpha-beta-crown/verify_expert_abcrown.py \\"
echo "       --model_path artifacts/gtsrb_small_cnn_best.pth \\"
echo "       --dataset GTSRB \\"
echo "       --epsilon 0.00784 \\"
echo "       --num_images 10 \\"
echo "       --auto_run"
echo ""
echo "  2. Or run verification manually:"
echo "     cd modules/alpha-beta-CROWN/complete_verifier"
echo "     export PYTHONPATH=\"../auto_LiRPA:\$PYTHONPATH\""
echo "     python abcrown.py --config <path-to-config.yaml>"
echo ""
echo "Documentation:"
echo "  - Quick start: src/Formal_Neural_Network_Verification/alpha-beta-crown/ABCROWN_QUICKSTART.md"
echo "  - Project docs: CLAUDE.md"
echo "  - Official docs: modules/alpha-beta-CROWN/README.md"
echo ""
