# Quick Start Guide

Get started with the MoE Research Test Suite in under 5 minutes!

## Prerequisites

- Go 1.21 or higher (check with `go version`)
- Python 3.10+ with PyTorch installed
- Trained MoE models in `artifacts/` directory

## Installation

### Step 1: Install Go Dependencies

**Easy Way - Use Installation Script:**

Windows:
```bash
cd unittest
install_deps.bat
```

Linux/macOS:
```bash
cd unittest
./install_deps.sh
```

**Manual Way:**
```bash
cd unittest
go mod tidy
go get github.com/jung-kurt/gofpdf
go get github.com/wcharczuk/go-chart/v2
go get github.com/stretchr/testify
```

**Or use Makefile:**
```bash
cd unittest
make install
```

### Step 2: Set Up Test Data

Create a test image for testing:

**Option A: Copy from your dataset**
```bash
# Create a simple test image using Python
python -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img.save('testdata/images/test_sample.png')
"
```

**Option B: Use existing dataset images**
```bash
# Copy a sample from GTSRB
cp ../data/GTSRB/Test/Images/00000.ppm testdata/images/gtsrb_sample.ppm
```

## Running Tests

### Quick Test Run

```bash
# Run all tests
go test -v ./...
```

### Generate PDF Report

```bash
# Run tests and save output
go test -v ./... | tee test_output.txt

# Generate PDF report
go run cmd/report_generator/main.go

# Report will be in: reports/test_report_YYYY-MM-DD_HH-MM-SS.pdf
```

### One-Command Full Suite

```bash
# Run everything automatically
go run cmd/test_runner/main.go
```

Or with Make:
```bash
make full
```

## Expected Output

After running the full suite, you'll see:

```
╔════════════════════════════════════════════╗
║  MoE Research Test Runner                 ║
║  Mixture-of-Experts Testing Suite         ║
╚════════════════════════════════════════════╝

Project Root: D:\Mixture-of-Experts_Research
Working Directory: D:\Mixture-of-Experts_Research\unittest

Running tests...
──────────────────────────────────────────
Running: ..........
✓ All tests passed (duration: 2.34s)

Generating PDF report...
──────────────────────────────────────────

Test Summary:
  Total:   8
  Passed:  8
  Failed:  0
  Skipped: 0

✓ Report generated successfully!
  Location: reports\test_report_2025-10-08_14-23-45.pdf
```

## What Gets Tested?

1. **Unit Tests** (`internal/models/inference_test.go`)
   - Model loading and validation
   - Single expert inference
   - MetaMoE routing
   - Performance benchmarks

2. **Regression Tests** (`internal/regression/regression_test.go`)
   - Output consistency with baselines
   - Multi-image batch testing
   - Router weight stability

## Troubleshooting

### "Model file not found"

Tests will skip if models aren't available. Train models first:

```bash
# Train a GTSRB expert
cd ..
python train.py --dataset GTSRB --model_arch small_cnn --epochs 10
```

### "Test image not found"

Create test images as shown in Step 2 above.

### "go: cannot find module"

Run:
```bash
go mod tidy
```

### Python inference errors

Ensure you're using the same Python environment as your training:

```bash
# Check Python path
which python

# Update config.json if needed
{
  "python_executable": "/path/to/your/python"
}
```

## Next Steps

- **Add more test images**: Copy more samples to `testdata/images/`
- **Create baselines**: Run regression tests to establish baselines
- **Customize tests**: Edit test files in `internal/*/`
- **Configure CI/CD**: See [README.md](README.md#cicd-integration)
- **View reports**: Open generated PDFs in `reports/`

## Common Commands

```bash
# Run specific test
go test -v -run TestModelInferenceBasic ./internal/models

# Run with coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Benchmark tests
go test -bench=. ./...

# Clean up
make clean
```

## Get Help

- Full documentation: [README.md](README.md)
- Test data setup: [testdata/README.md](testdata/README.md)
- Issues: Check test output and logs

Happy testing! 🚀
