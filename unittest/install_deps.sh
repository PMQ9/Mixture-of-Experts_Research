#!/bin/bash
# Installation script for MoE Test Suite dependencies (Linux/macOS)

set -e

echo "========================================"
echo "MoE Test Suite - Dependency Installer"
echo "========================================"
echo ""

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "ERROR: Go is not installed!"
    echo ""
    echo "Please install Go from: https://golang.org/dl/"
    echo "Recommended version: Go 1.21 or higher"
    echo ""
    echo "Quick install (Linux):"
    echo "  wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz"
    echo "  sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz"
    echo "  export PATH=\$PATH:/usr/local/go/bin"
    echo ""
    echo "Quick install (macOS with Homebrew):"
    echo "  brew install go"
    echo ""
    exit 1
fi

echo "[1/5] Checking Go installation..."
go version
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "WARNING: Python is not found in PATH"
    echo "You may need Python for running model inference"
    echo ""
else
    echo "[2/5] Checking Python installation..."
    if command -v python3 &> /dev/null; then
        python3 --version
    else
        python --version
    fi
    echo ""
fi

echo "[3/5] Initializing Go module..."
go mod tidy
echo ""

echo "[4/5] Installing Go dependencies..."
echo "  - Installing gofpdf (PDF generation)..."
go get github.com/jung-kurt/gofpdf

echo "  - Installing go-chart (visualization)..."
go get github.com/wcharczuk/go-chart/v2

echo "  - Installing testify (testing utilities)..."
go get github.com/stretchr/testify/assert
echo ""

echo "[5/5] Verifying installation..."
go mod tidy
go build -o /tmp/report_generator_test ./cmd/report_generator > /dev/null 2>&1
go build -o /tmp/test_runner_test ./cmd/test_runner > /dev/null 2>&1
rm -f /tmp/report_generator_test /tmp/test_runner_test
echo ""

echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo ""
echo "Installed packages:"
echo "  - github.com/jung-kurt/gofpdf"
echo "  - github.com/wcharczuk/go-chart/v2"
echo "  - github.com/stretchr/testify"
echo ""
echo "Next steps:"
echo "  1. Create test images: python scripts/create_test_image.py"
echo "  2. Run tests: ./run_tests.sh"
echo "  3. Or use: go run cmd/test_runner/main.go"
echo ""
echo "Documentation:"
echo "  - Quick start: QUICKSTART.md"
echo "  - Full guide: README.md"
echo ""
