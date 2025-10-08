# Testing Suite Architecture

This document describes the architecture of the MoE Research testing suite implemented in Go.

## Overview

The testing suite is designed to:
1. Test Python-based ML models from Go (cross-language testing)
2. Perform unit tests on individual components
3. Run regression tests to detect output changes
4. Generate comprehensive PDF reports
5. Be easily integrated into CI/CD pipelines

## Architecture Diagram

```
unittest/
│
├── cmd/                          # Executable commands
│   ├── report_generator/         # PDF report generator
│   │   └── main.go              # Parses test output → PDF
│   └── test_runner/              # Main test orchestrator
│       └── main.go              # Runs tests → generates report
│
├── internal/                     # Internal packages (not exported)
│   ├── models/                   # Model testing
│   │   └── inference_test.go    # Unit tests for inference
│   ├── regression/               # Regression testing
│   │   ├── baseline.go          # Baseline management
│   │   └── regression_test.go   # Regression test suite
│   ├── report/                   # Report generation
│   │   └── generator.go         # PDF generation logic
│   └── utils/                    # Utilities
│       ├── config.go            # Configuration management
│       └── python_bridge.go     # Go ↔ Python bridge
│
├── scripts/                      # Python helper scripts
│   ├── inference.py             # Model inference wrapper
│   └── create_test_image.py     # Test data generation
│
├── testdata/                     # Test data
│   ├── images/                  # Test images
│   ├── baselines/               # Regression baselines (JSON)
│   └── models/                  # Optional test models
│
├── reports/                      # Generated PDF reports
│
├── config.json                   # Test configuration
├── go.mod / go.sum              # Go dependencies
├── run_tests.bat / .sh          # Quick run scripts
└── Makefile                      # Build automation

```

## Component Details

### 1. Python Bridge (`internal/utils/python_bridge.go`)

**Purpose**: Enable Go tests to execute Python ML code

**Key Features**:
- Spawns Python subprocess for inference
- Passes JSON data between Go and Python
- Handles errors and timeouts
- Validates model/image paths

**Flow**:
```
Go Test → PythonBridge.RunInference() → exec.Command("python", "inference.py")
                                       ↓
                                  Python runs inference
                                       ↓
                                  JSON output to stdout
                                       ↓
Go Test ← ModelOutput struct ← json.Unmarshal()
```

### 2. Inference Tests (`internal/models/inference_test.go`)

**Test Cases**:
- `TestModelInferenceBasic`: Basic single expert inference
- `TestMetaMoEInference`: MetaMoE routing behavior
- `TestModelValidation`: Model file validation
- `TestInferencePerformance`: Performance benchmarks

**Validation**:
- Class predictions are valid
- Confidence scores in [0, 1]
- Inference times are reasonable
- Router weights sum to 1.0 (MetaMoE)

### 3. Regression Testing (`internal/regression/`)

**Purpose**: Detect unintended changes in model outputs

**Baseline Management**:
- Baselines stored as JSON files
- Format: `{model_name}_{image_name}_baseline.json`
- Contains: predictions, class, confidence, router info, timestamp

**Comparison Logic**:
```go
Compare(baseline, current, tolerance) → RegressionResult {
    - Check class prediction match
    - Compare confidence within tolerance
    - Compare logits element-wise
    - Compare router weights (if MetaMoE)
}
```

**Workflow**:
1. First run: Create baselines
2. Subsequent runs: Compare with baselines
3. Failures: Report differences with details

### 4. PDF Report Generator (`internal/report/generator.go`)

**Report Sections**:
1. **Cover Page**: Test summary and statistics
2. **Executive Summary**: Overall status and suite breakdown
3. **Detailed Results**: Per-test results with outputs
4. **Performance Metrics**: Timing analysis and slowest tests

**Technologies**:
- `gofpdf`: PDF generation
- `go-chart`: Visualization (charts/graphs)
- Custom formatting for test output

**Features**:
- Color-coded status (green=pass, red=fail, yellow=skip)
- Timestamped reports
- Professional formatting
- Error details with stack traces

### 5. Test Runner (`cmd/test_runner/main.go`)

**Orchestration**:
1. Load configuration
2. Run `go test -v ./...`
3. Capture output to `test_output.txt`
4. Parse test results
5. Generate PDF report
6. Display summary

**Configuration** (`config.json`):
```json
{
  "python_executable": "python",        // Python path
  "project_root": "..",                 // Project root
  "models_to_test": [...],             // Model paths
  "test_images": [...],                // Test image paths
  "regression_tolerance": 1e-5,        // Regression tolerance
  "report_output_dir": "reports"       // Report output
}
```

## Data Flow

### Unit Test Flow
```
1. Test starts
   ↓
2. Load configuration
   ↓
3. Validate model/image exist
   ↓
4. PythonBridge.RunInference()
   ↓
5. Python subprocess runs
   ↓
6. JSON output parsed
   ↓
7. Assertions on ModelOutput
   ↓
8. Test passes/fails
```

### Regression Test Flow
```
1. Test starts
   ↓
2. Try to load baseline
   ↓
3. If baseline missing:
   - Run inference
   - Save as baseline
   - Skip comparison
   ↓
4. If baseline exists:
   - Run inference
   - Compare with baseline
   - Report differences
   ↓
5. Test passes/fails
```

### Report Generation Flow
```
1. Tests complete → test_output.txt
   ↓
2. report_generator reads output
   ↓
3. Parse with regex:
   - === RUN TestName
   - --- PASS: TestName (0.23s)
   - --- FAIL: TestName (0.45s)
   ↓
4. Build TestSuite structs
   ↓
5. ReportGenerator.GenerateReport()
   ↓
6. PDF created with gofpdf
   ↓
7. Save to reports/ with timestamp
```

## Key Design Decisions

### Why Go for Testing?

1. **Performance**: Fast execution, efficient concurrency
2. **Single Binary**: Easy deployment (no Python deps for tests)
3. **Type Safety**: Compile-time error checking
4. **Resume Value**: Adds Go to skill set 😎
5. **PDF Generation**: Excellent libraries (`gofpdf`)

### Why Cross-Language?

- Models are in PyTorch (Python)
- Tests in Go provide:
  - Independent validation
  - Different perspective
  - CI/CD flexibility
  - Language-agnostic testing

### Why JSON for Communication?

- Language-neutral format
- Easy to parse in both Go and Python
- Human-readable for debugging
- Structured data validation

## Extensibility

### Adding New Tests

1. Create `*_test.go` file in appropriate package
2. Use `testing` package conventions
3. Leverage existing utilities (PythonBridge, config)
4. Follow naming: `TestFeatureName`

### Adding New Report Sections

1. Edit `internal/report/generator.go`
2. Add new `add*` method
3. Call from `GenerateReport()`
4. Use `gofpdf` API for formatting

### Adding New Metrics

1. Extend `ModelOutput` struct in `python_bridge.go`
2. Update `inference.py` to output new metrics
3. Add validation in tests
4. Include in reports

## Performance Considerations

### Test Execution
- Tests run in parallel by default (`go test`)
- Each test spawns Python subprocess (overhead ~100-200ms)
- Use `-p N` to control parallelism

### Report Generation
- Parsing is O(n) in test output size
- PDF generation is fast (<1s for 100 tests)
- Memory efficient (streaming)

### Baselines
- JSON files are small (~1-10 KB each)
- Comparison is O(n) in prediction vector size
- Git-friendly format

## Security Considerations

- No user input directly to shell commands
- Python paths validated before execution
- File paths sanitized
- Test isolation (no shared state)

## Future Enhancements

Potential additions:
1. **Database backend**: Store test history
2. **Web dashboard**: Interactive reports
3. **Email notifications**: Test failure alerts
4. **Parallel Python execution**: Batch inference
5. **Docker integration**: Containerized tests
6. **Slack/Discord webhooks**: Team notifications
7. **Grafana integration**: Metrics visualization
8. **A/B testing**: Compare model versions

## Debugging

### Test Failures
```bash
# Run single test with verbose output
go test -v -run TestName ./internal/models

# Check Python output
python scripts/inference.py --model_path ... --image_path ...

# Verify test data
ls -la testdata/images/
```

### Report Issues
```bash
# Check test output
cat test_output.txt

# Run report generator manually
go run cmd/report_generator/main.go -input test_output.txt
```

### Configuration Problems
```bash
# Validate JSON
python -m json.tool config.json

# Check paths
go run -c 'package main; import "fmt"; import u "internal/utils"; func main() { fmt.Println(u.GetProjectRoot()) }'
```

## Maintenance

### Regular Tasks
- Update baselines after intentional model changes
- Clean old reports: `make clean`
- Review and update test images
- Keep dependencies updated: `go get -u ./...`

### CI/CD Integration
See [README.md](README.md#cicd-integration) for GitLab CI configuration.

---

**Author**: Claude (with Go enthusiasm!)
**Date**: 2025-10-08
**Version**: 1.0
