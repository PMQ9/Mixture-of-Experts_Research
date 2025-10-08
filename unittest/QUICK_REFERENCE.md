# Quick Reference Card

**MoE Research Testing Suite - Essential Commands**

---

## 🚀 Quick Start (Copy & Paste)

```bash
# 1. Navigate to unittest directory
cd unittest

# 2. Install dependencies (first time only)
go mod tidy

# 3. Create test images (first time only)
python scripts/create_test_image.py

# 4. Run everything
go run cmd/test_runner/main.go

# 5. View report (Windows)
start reports\test_report_*.pdf
```

---

## 📋 Common Commands

### Run Tests

```bash
# All tests
go test -v ./...

# Specific package
go test -v ./internal/models
go test -v ./internal/regression

# Specific test
go test -v -run TestModelInferenceBasic ./internal/models

# With output to file
go test -v ./... | tee test_output.txt
```

### Generate Reports

```bash
# Generate PDF from test_output.txt
go run cmd/report_generator/main.go

# Full suite (tests + report)
go run cmd/test_runner/main.go

# Quick scripts
./run_tests.bat    # Windows
./run_tests.sh     # Linux/Mac
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

## 📁 File Locations

| What | Where |
|------|-------|
| **Test Output** | `test_output.txt` |
| **PDF Reports** | `reports/test_report_*.pdf` |
| **Test Images** | `testdata/images/` |
| **Baselines** | `testdata/baselines/` |
| **Configuration** | `config.json` |

---

## ⚙️ Configuration

Edit `config.json`:

```json
{
  "python_executable": "python",
  "project_root": "..",
  "models_to_test": [
    "artifacts/results/gtsrb_small_cnn_best.pth"
  ],
  "test_images": [
    "testdata/images/test_sample.png"
  ],
  "regression_tolerance": 1e-5,
  "report_output_dir": "reports"
}
```

---

## 🔧 Test Data Setup

### Option 1: Create Synthetic Images
```bash
python scripts/create_test_image.py
```

### Option 2: Copy from Dataset
```bash
# GTSRB
cp ../data/GTSRB/Test/Images/00000.ppm testdata/images/

# CIFAR-10
cp ../data/CIFAR10/Test/airplane_0001.png testdata/images/
```

---

## 📊 Understanding Test Results

### Test Status
- ✅ **PASS** - Test succeeded
- ❌ **FAIL** - Test failed (check error message)
- ⏭️  **SKIP** - Test skipped (missing dependencies)

### Common Skip Reasons
- "Model not found" - Train models first
- "Test image not found" - Add images to testdata/images/
- "Baseline not found" - First run creates baselines

---

## 🐛 Troubleshooting

### "go: cannot find module"
```bash
go mod tidy
```

### "Model file not found"
Train models or update paths in `config.json`

### "Python inference failed"
```bash
# Test Python directly
python scripts/inference.py \
  --model_path ../artifacts/results/gtsrb_small_cnn_best.pth \
  --image_path testdata/images/test_sample.png \
  --model_type single
```

### "Report generation failed"
Check that test_output.txt exists or run with sample data

### Tests hanging
Check Python process isn't waiting for input

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** | Complete user guide |
| **QUICKSTART.md** | 5-minute getting started |
| **ARCHITECTURE.md** | Technical architecture |
| **SHOWCASE.md** | Resume/portfolio showcase |
| **SUMMARY.txt** | Project summary |
| **This file** | Quick reference |

---

## 💡 Pro Tips

1. **Run tests in parallel** - Go does this automatically
2. **Use baselines** - First run creates them, subsequent runs compare
3. **Check PDF reports** - Much easier to read than terminal output
4. **Clean up regularly** - `make clean` removes old reports
5. **Version control baselines** - Commit to track model changes

---

## 🎯 Common Workflows

### First-Time Setup
```bash
cd unittest
go mod tidy
python scripts/create_test_image.py
go run cmd/test_runner/main.go
```

### Daily Testing
```bash
cd unittest
make full
# or
./run_tests.bat
```

### Before Committing Code
```bash
cd unittest
make test
make coverage
```

### After Model Training
```bash
cd unittest
# Update config.json with new model paths
go test -v ./internal/regression  # Create new baselines
make full
```

### CI/CD Pipeline
```bash
cd unittest
make ci
# Artifacts: reports/*.pdf
```

---

## 📞 Need Help?

1. Check documentation (especially QUICKSTART.md)
2. Read test output carefully
3. Try running Python scripts directly
4. Check configuration in config.json
5. Verify paths exist

---

## 🎓 Learn More

- **Go Testing**: https://golang.org/pkg/testing/
- **gofpdf**: https://github.com/jung-kurt/gofpdf
- **Project Architecture**: See ARCHITECTURE.md
- **Resume Tips**: See SHOWCASE.md

---

**Version**: 1.0
**Last Updated**: 2025-10-08
**Questions?** Check README.md for detailed documentation
