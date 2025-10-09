# MoE Research Testing Suite

**Go-based Testing Framework for Machine Learning Models**

A production-grade testing suite for validating Mixture-of-Experts (MoE) PyTorch models. Features cross-language integration, automated regression testing, and professional PDF report generation.

[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)](https://golang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://python.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Test Types](#test-types)
- [PDF Reports](#pdf-reports)
- [Architecture](#architecture)
- [Command Reference](#command-reference)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)

---

## Overview

This testing suite provides comprehensive validation for Mixture-of-Experts machine learning models:

- 🔗 **Cross-Language Testing** - Go tests for Python/PyTorch models
- 🔄 **Regression Testing** - Baseline management to detect output changes
- 📊 **PDF Reports** - Professional reports with visualizations
- ⚡ **Performance Benchmarks** - Inference time and resource monitoring
- 🚀 **CI/CD Ready** - GitLab CI integration included

### Why Go for ML Testing?

- **Performance**: Fast execution, efficient concurrency
- **Type Safety**: Compile-time error checking
- **Single Binary**: Easy deployment, no Python dependencies for tests
- **Cross-Platform**: Works on Windows, Linux, macOS

---

## Features

### ✅ Unit Testing
- Model loading and validation
- Single expert inference testing
- MetaMoE routing validation
- Performance benchmarking

### ✅ Regression Testing
- Automatic baseline creation
- Tolerance-based comparison
- Multi-image batch testing
- Detailed diff reporting

### ✅ PDF Report Generation
- Cover page with statistics
- Executive summary
- Detailed test results
- Performance metrics
- Color-coded status

### ✅ Multiple Interfaces
- Direct Go testing (`go test`)
- Makefile automation
- Cross-platform scripts (`.bat`, `.sh`)
- Full orchestrator

---

## Quick Start (5 Minutes)

Get started in under 5 minutes!

### 1. Install Dependencies

**Windows:**
```bash
cd unittest
install_deps.bat
```

**Linux/macOS:**
```bash
cd unittest
./install_deps.sh
```

### 2. Create Test Images

```bash
python scripts/create_test_image.py
```

### 3. Run Tests

**Option A - Quick Script:**
```bash
# Windows
run_tests.bat

# Linux/macOS
./run_tests.sh
```

**Option B - Full Orchestrator:**
```bash
go run cmd/test_runner/main.go
```

**Option C - Manual:**
```bash
go test -v ./... | tee test_output.txt
go run cmd/report_generator/main.go
```

### 4. View Report

PDF report generated at: `reports/test_report_YYYY-MM-DD_HH-MM-SS.pdf`

---

## Installation

### Prerequisites

- Go 1.21 or higher ([Download](https://golang.org/dl/))
- Python 3.10+ with PyTorch
- Trained MoE models in `artifacts/` directory

### Automated Installation

**Windows:**
```bash
cd unittest
install_deps.bat
```

**Linux/macOS:**
```bash
cd unittest
./install_deps.sh
```

The installation script will:
- ✅ Check Go and Python installation
- ✅ Install all required Go packages
- ✅ Verify the installation
- ✅ Provide next steps

### Manual Installation

```bash
cd unittest
go mod tidy
go get github.com/jung-kurt/gofpdf
go get github.com/wcharczuk/go-chart/v2
go get github.com/stretchr/testify/assert
```

**Or use Makefile:**
```bash
make install
```

### Required Packages

- `github.com/jung-kurt/gofpdf` - PDF generation
- `github.com/wcharczuk/go-chart/v2` - Charts and visualizations
- `github.com/stretchr/testify` - Testing utilities

---

## Project Structure

```
unittest/
├── cmd/                          # Executable commands
│   ├── report_generator/         # PDF report generator
│   │   └── main.go
│   └── test_runner/              # Main test orchestrator
│       └── main.go
│
├── internal/                     # Internal packages
│   ├── models/                   # Model testing
│   │   └── inference_test.go
│   ├── regression/               # Regression testing
│   │   ├── baseline.go
│   │   └── regression_test.go
│   ├── report/                   # PDF generation
│   │   └── generator.go
│   └── utils/                    # Utilities
│       ├── config.go
│       └── python_bridge.go
│
├── scripts/                      # Helper scripts
│   ├── inference.py              # Model inference wrapper
│   └── create_test_image.py      # Test data generation
│
├── testdata/                     # Test data
│   ├── images/                   # Test images
│   ├── baselines/                # Regression baselines (JSON)
│   └── models/                   # Optional test models
│
├── reports/                      # Generated PDF reports
│
├── config.json                   # Configuration
├── go.mod / go.sum              # Go dependencies
├── install_deps.bat/.sh          # Installation scripts
├── run_tests.bat/.sh             # Quick run scripts
├── Makefile                      # Build automation
└── README.md                     # This file
```

---

## Usage Guide

### Running Tests

**All tests:**
```bash
go test -v ./...
```

**Specific package:**
```bash
# Unit tests only
go test -v ./internal/models

# Regression tests only
go test -v ./internal/regression
```

**Specific test:**
```bash
go test -v -run TestModelInferenceBasic ./internal/models
```

**With coverage:**
```bash
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Generate PDF Report

**From test output:**
```bash
go test -v ./... > test_output.txt
go run cmd/report_generator/main.go
```

**Full suite:**
```bash
go run cmd/test_runner/main.go
```

### Using Makefile

```bash
make install          # Install dependencies
make test            # Run all tests
make test-unit       # Unit tests only
make test-regression # Regression tests only
make report          # Generate PDF report
make full            # Run tests + generate report
make clean           # Clean generated files
make coverage        # Generate coverage report
make help            # Show all commands
```

---

## Configuration

Edit `config.json`:

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
    "testdata/images/test_sample.png"
  ],
  "regression_tolerance": 1e-5,
  "report_output_dir": "reports"
}
```

### Parameters

- `python_executable` - Python command (default: "python")
- `project_root` - Path to project root (default: "..")
- `models_to_test` - List of model paths to test
- `test_images` - List of test image paths
- `regression_tolerance` - Tolerance for regression tests (default: 1e-5)
- `report_output_dir` - Output directory for reports (default: "reports")

---

## Test Types

### 1. Unit Tests (`internal/models/inference_test.go`)

**TestModelInferenceBasic:**
- Model loading validation
- Single expert inference
- Output format validation
- Confidence score checks

**TestMetaMoEInference:**
- MetaMoE routing behavior
- Router weight validation
- Expert selection verification

**TestInferencePerformance:**
- Inference time benchmarks
- Performance threshold checks

### 2. Regression Tests (`internal/regression/regression_test.go`)

**TestRegressionBaseline:**
- Baseline creation
- Output comparison with tolerance
- Diff reporting

**TestRegressionMetaMoE:**
- MetaMoE output consistency
- Router weight stability

**TestMultipleImages:**
- Batch testing across multiple images
- Aggregate regression results

### 3. Validation Tests

**TestModelValidation:**
- File existence checks
- Path validation
- Error handling verification

---

## PDF Reports

Generated reports include:

### 1. Cover Page
- Project title and date
- Test summary (total, passed, failed, skipped)
- Success rate percentage
- Total duration

### 2. Executive Summary
- Overall test status
- Test suite breakdown
- Per-suite statistics

### 3. Detailed Results
- Per-test results with status
- Test output and logs
- Error messages with details
- Color-coded status indicators

### 4. Performance Metrics
- Total execution time
- Average test duration
- Top 5 slowest tests
- Performance analysis

**Report Location:** `reports/test_report_YYYY-MM-DD_HH-MM-SS.pdf`

---

## Architecture

### Cross-Language Bridge

The Python Bridge enables Go to execute Python ML code:

```go
type PythonBridge struct {
    PythonExec  string
    ProjectRoot string
}

func (pb *PythonBridge) RunInference(modelPath, imagePath, modelType string) (*ModelOutput, error)
```

**Flow:**
1. Go test spawns Python subprocess
2. Python runs inference and outputs JSON
3. Go parses JSON into type-safe struct
4. Assertions run on ModelOutput

### Regression Testing System

**Baseline Management:**
- First run: Creates baseline (JSON file)
- Subsequent runs: Compares with baseline
- Tolerance-based comparison for numerical stability
- Detailed diff reporting

**Baseline Storage:**
- Format: `{model_name}_{image_name}_baseline.json`
- Contains: predictions, class, confidence, router info, timestamp
- Git-friendly JSON format

### PDF Generation

Uses `gofpdf` library for professional reports:
- Multi-section layout
- Color-coded status
- Custom formatting
- Automated generation from test results

---

## Command Reference

### Quick Commands

```bash
# Run all tests
go test -v ./...

# Run specific test
go test -v -run TestName ./internal/models

# Generate report
go run cmd/report_generator/main.go

# Full suite
go run cmd/test_runner/main.go

# Quick scripts
./run_tests.bat     # Windows
./run_tests.sh      # Linux/Mac
```

### Makefile Commands

```bash
make install          # Install dependencies
make test            # Run all tests
make test-unit       # Unit tests only
make test-regression # Regression tests only
make test-run TEST=X # Run specific test
make report          # Generate PDF report
make full            # Run full suite with report
make ci              # Run in CI mode
make clean           # Clean generated files
make bench           # Run benchmarks
make coverage        # Generate coverage report
make fmt             # Format code
make lint            # Run linter
make help            # Show all commands
```

### File Locations

| What | Where |
|------|-------|
| Test Output | `test_output.txt` |
| PDF Reports | `reports/test_report_*.pdf` |
| Test Images | `testdata/images/` |
| Baselines | `testdata/baselines/` |
| Configuration | `config.json` |

---

## CI/CD Integration

### GitLab CI Example

Add to `.gitlab-ci.yml`:

```yaml
test:
  stage: test
  script:
    - cd unittest
    - ./install_deps.sh
    - go test -v ./... | tee test_output.txt
    - go run cmd/report_generator/main.go
  artifacts:
    paths:
      - unittest/reports/*.pdf
    expire_in: 30 days
  tags:
    - go
```

### GitHub Actions Example

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      - name: Install dependencies
        run: cd unittest && ./install_deps.sh
      - name: Run tests
        run: cd unittest && go test -v ./...
      - name: Generate report
        run: cd unittest && go run cmd/report_generator/main.go
      - uses: actions/upload-artifact@v3
        with:
          name: test-report
          path: unittest/reports/*.pdf
```

---


## Changelog

### Version 1.1 - 2025-10-08

**Added:**
- ✅ Installation scripts (`install_deps.bat`, `install_deps.sh`)
- ✅ Automated dependency installation for Windows and Linux
- ✅ Consolidated documentation (this README)

**Fixed:**
- ✅ PDF report spacing - reduced excessive white space
- ✅ PDF report title - now displays correctly
- ✅ Improved visual hierarchy and readability

**Improved:**
- ✅ All documentation consolidated into single README
- ✅ Enhanced Makefile with better feedback
- ✅ Better error messages and guidance

### Version 1.0 - 2025-10-08 (Initial Release)

**Features:**
- Cross-language testing framework (Go ↔ Python)
- Unit tests for model inference
- Regression tests with baseline management
- PDF report generation
- CI/CD integration support
- Comprehensive documentation

**Components:**
- Python bridge for model inference
- Test suites (unit, regression, performance)
- PDF report generator
- Test orchestrator
- Configuration management

---

## Support & Contributing

### Get Help

1. Check this documentation
2. Review test output and logs
3. Try debug commands above
4. Check `test_output.txt` for details

### File Structure for Test Data

```
testdata/
├── images/           # Add your test images here
├── baselines/        # Auto-generated baselines
└── models/           # Optional: small test models
```

**Setup Test Data:**
```bash
# Create synthetic images
python scripts/create_test_image.py

# Or copy from datasets
cp ../data/GTSRB/Test/Images/*.ppm testdata/images/
```

### Dependencies

**Go Packages:**
- `github.com/jung-kurt/gofpdf` - PDF generation
- `github.com/wcharczuk/go-chart/v2` - Visualization
- `github.com/stretchr/testify` - Testing utilities

**Python Requirements:**
- PyTorch (for model inference)
- PIL/Pillow (for image processing)
- NumPy (for array operations)

---

## Quick Reference

### Installation
```bash
cd unittest && ./install_deps.sh    # Linux/Mac
cd unittest && install_deps.bat     # Windows
```

### Running Tests
```bash
go test -v ./...                    # All tests
make test                           # Using Makefile
./run_tests.sh                      # Quick script
```

### Generate Report
```bash
go run cmd/report_generator/main.go
```

### Clean Up
```bash
make clean
```
