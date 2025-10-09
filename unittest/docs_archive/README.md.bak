# MoE Research Testing Suite

Professional testing suite for the Mixture-of-Experts research project using Go.

## Features

- **Unit Tests**: Test individual components (model loading, inference, data processing)
- **Regression Tests**: Compare model outputs against baseline to detect unintended changes
- **PDF Reports**: Automatically generated test reports with visualizations and metrics
- **Python Integration**: Tests Python models via subprocess calls

## Quick Start

```bash
# Run all tests and generate PDF report
cd unittest
go test -v ./... > test_output.txt
go run cmd/report_generator/main.go

# View the report
# Opens: unittest/reports/test_report_YYYY-MM-DD_HH-MM-SS.pdf
```

## Structure

```
unittest/
├── cmd/
│   ├── report_generator/     # PDF report generator
│   └── test_runner/           # Main test runner
├── internal/
│   ├── models/                # Model testing utilities
│   ├── regression/            # Regression test suite
│   └── utils/                 # Helper functions
├── testdata/
│   ├── images/                # Test images for inference
│   ├── baselines/             # Baseline outputs for regression
│   └── models/                # Test model checkpoints
├── reports/                   # Generated PDF reports
└── go.mod
```

## Installation

Before running tests, install the required Go dependencies:

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

The script will:
- Check Go and Python installation
- Install all required Go packages
- Verify the installation

**Or use Makefile:**
```bash
make install
```

**Required packages:**
- `github.com/jung-kurt/gofpdf` - PDF generation
- `github.com/wcharczuk/go-chart/v2` - Charts and visualizations
- `github.com/stretchr/testify` - Testing utilities

## Running Tests

### Run All Tests
```bash
go test -v ./...
```

### Run Specific Test Suites
```bash
# Unit tests only
go test -v ./internal/models

# Regression tests only
go test -v ./internal/regression
```

### Generate PDF Report
```bash
go run cmd/report_generator/main.go
```

### Custom Test Runner
```bash
# Run tests with custom configuration
go run cmd/test_runner/main.go --config config.json
```

## Test Configuration

Create `config.json` in the unittest directory:

```json
{
  "python_executable": "python",
  "project_root": "..",
  "models_to_test": [
    "artifacts/results/gtsrb_small_cnn_best.pth",
    "artifacts/results/cifar10_tiny_cnn_best.pth",
    "artifacts/meta_moe_small_cnn_best.pth"
  ],
  "test_images": [
    "data/GTSRB/Test/Images/00000.ppm",
    "data/CIFAR10/Test/airplane_0001.png"
  ],
  "regression_tolerance": 1e-5
}
```

## Report Contents

The PDF report includes:

1. **Executive Summary**: Pass/fail counts, duration, overall status
2. **Unit Test Results**: Detailed results for each unit test
3. **Regression Test Results**: Output comparisons with baseline
4. **Performance Metrics**: Inference times, memory usage
5. **Visualizations**: Charts showing test trends over time
6. **Error Details**: Stack traces and debugging information

## CI/CD Integration

Add to your GitLab CI pipeline:

```yaml
test:
  stage: test
  script:
    - cd unittest
    - go test -v ./... | tee test_output.txt
    - go run cmd/report_generator/main.go
  artifacts:
    paths:
      - unittest/reports/*.pdf
    expire_in: 30 days
```

## Writing New Tests

### Unit Test Example

```go
package models_test

import (
    "testing"
    "moe-research/unittest/internal/models"
)

func TestModelInference(t *testing.T) {
    model := models.LoadModel("path/to/model.pth")
    output := model.Predict("path/to/image.png")

    if output.Class < 0 {
        t.Errorf("Invalid class prediction: %d", output.Class)
    }
}
```

### Regression Test Example

```go
package regression_test

import (
    "testing"
    "moe-research/unittest/internal/regression"
)

func TestOutputRegression(t *testing.T) {
    baseline := regression.LoadBaseline("gtsrb_model_baseline.json")
    current := regression.RunInference("gtsrb_model.pth", "test_image.png")

    if !regression.Compare(baseline, current, 1e-5) {
        t.Error("Model output differs from baseline")
    }
}
```

## Requirements

- Go 1.21 or higher
- Python 3.10+ (for model inference)
- PyTorch (installed in parent project)

## Dependencies

The following Go packages are used:

- `github.com/jung-kurt/gofpdf`: PDF generation
- `github.com/wcharczuk/go-chart/v2`: Chart generation for visualizations
- `github.com/stretchr/testify`: Testing utilities

Install dependencies:
```bash
go get github.com/jung-kurt/gofpdf
go get github.com/wcharczuk/go-chart/v2
go get github.com/stretchr/testify
```
