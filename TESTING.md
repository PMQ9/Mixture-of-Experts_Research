# Testing Guide

Complete testing infrastructure for the Mixture-of-Experts Research project with 90+ unit, integration, and regression tests.

## Quick Start

Install dependencies:
```bash
pip install -r .gitlab/test_requirements.txt
```

Run tests:
```bash
pytest tests/ -v                    # All tests
pytest tests/ -m unit -v            # Unit tests only
pytest tests/ -m regression -v      # Regression tests
pytest tests/ -m integration -v     # Integration tests
pytest tests/ -m "not slow" -v      # Skip slow tests
```

## What's Included

### Test Files (1,400+ lines)
- **tests/test_models.py** - 40+ unit tests for model architectures
- **tests/test_training_pipeline.py** - 30+ unit tests for training components
- **tests/test_integration.py** - 20+ integration tests for workflows
- **tests/test_regression.py** - 25+ regression tests for consistency
- **tests/conftest.py** - Pytest configuration and 15+ fixtures

### Configuration
- **pytest.ini** - Test discovery and markers
- **.gitlab/test_requirements.txt** - Dependencies

### CI/CD
- **.gitlab-ci.yml** - Updated with unit_tests stage (runs before training)

## Test Categories

### Unit Tests (50+ tests)
Fast tests without external dependencies:
- Model creation and forward pass
- Output shape validation
- Parameter counting
- Loss function behavior
- Learning rate scheduling
- Argument parsing
- Configuration loading
- Training utilities (accuracy, gradients, etc.)

### Integration Tests (20+ tests)
End-to-end workflow tests:
- Data loading (CIFAR-10, MNIST)
- Data augmentation and normalization
- Training and validation steps
- Checkpoint save/load
- ONNX export and inference

### Regression Tests (25+ tests)
Consistency and stability checks:
- Output determinism
- Loss stability
- Parameter initialization consistency
- Architecture stability
- Training state isolation
- No unintended state leakage

## Running Tests

### Common Commands
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_models.py::TestSmallExpertCNN -v

# Run specific test
pytest tests/test_models.py::TestSmallExpertCNN::test_forward_pass -v

# Run tests matching a pattern
pytest tests/ -k "forward_pass" -v

# Run with short output (one line per test)
pytest tests/ --tb=short

# Run with full output including print statements
pytest tests/ -v -s

# Generate coverage report
pip install pytest-cov
pytest tests/ --cov=src/Vision_Transformer_Pytorch --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Markers
Tests use markers for filtering:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Slower integration tests
- `@pytest.mark.regression` - Consistency checks
- `@pytest.mark.slow` - Time-consuming tests
- `@pytest.mark.gpu` - GPU-required tests

Examples:
```bash
pytest tests/ -m unit              # Only unit tests
pytest tests/ -m regression        # Only regression tests
pytest tests/ -m "not slow"        # Skip slow tests
pytest tests/ -m gpu               # Only GPU tests
```

## Available Fixtures

Reusable test components in conftest.py:

**Device Management:**
- `device` - Returns CUDA if available, otherwise CPU

**Directories:**
- `temp_dir` - Temporary directory for test artifacts
- `project_root` - Project root path
- `artifacts_dir` - Artifacts directory
- `data_dir` - Data directory path

**Test Data:**
- `small_batch` - Dummy image batch (2, 3, 32, 32)
- `small_labels` - Dummy labels (batch size 2)
- `config_small_vit` - Minimal Vision Transformer config

**Models:**
- `model_small_cnn` - SmallExpertCNN instance (1.5M params)
- `model_tiny_cnn` - TinyExpertCNN instance (620K params)
- `model_micro_cnn` - MicroExpertCNN instance (67K params)

Example:
```python
def test_model(self, model_small_cnn, small_batch, device):
    """Test model using fixtures."""
    small_batch = small_batch.to(device)
    output = model_small_cnn(small_batch)
    assert output.shape == (2, 10)
```

## Test Coverage

### Models Tested
- SmallExpertCNN (1.5M parameters)
- TinyExpertCNN (620K parameters)
- MicroExpertCNN (67K parameters)
- Model factory wrapper

### Training Components
- Loss functions: CrossEntropy, KL divergence, Balance loss
- Learning rate scheduling: StepLR, CosineAnnealing, Warmup
- Argument parsing and validation
- Configuration loading

### Workflows
- Data loading and augmentation
- Training steps and epochs
- Validation procedures
- Checkpointing and restoration
- ONNX export and inference

### Consistency
- Deterministic outputs (same seed)
- Parameter initialization reproducibility
- Gradient flow validation
- Output range validation
- No batch size dependency
- State isolation between batches

## CI/CD Integration

### GitHub Actions (.github/workflows/tests.yml)
Automated testing on all branches:
- Unit and regression tests on Ubuntu (Python 3.10)
- Integration tests on Ubuntu (Python 3.10)
- Runs on push to any branch and pull requests
- Test results uploaded as artifacts

### GitLab CI (.gitlab-ci.yml)
Pipeline stages:
1. test_gpu - Check GPU availability
2. prepare - Clone/update repository
3. unit_tests - Run pytest
4. train - Run training script
5. clean_up - Cleanup

Test stage details:
- Runs unit tests before training
- Tests must pass to proceed
- Logs saved to artifacts/unit_tests_log.txt
- Failures block training

## Troubleshooting

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### GPU Tests Fail
GPU tests auto-skip if CUDA unavailable. Force CPU:
```bash
pytest tests/ -m "not gpu" -v
```

### Out of Memory
```bash
pytest tests/ -m "not slow" -v
```

### Dataset Not Found
Some integration tests skip if datasets missing:
```bash
pytest tests/ -m "not cifar10 and not mnist and not gtsrb"
```

### Slow Tests
```bash
pytest tests/ -m "not slow" -v
```

## Writing New Tests

### File Structure
```python
import pytest
import torch

class TestNewFeature:
    """Test suite for new feature."""

    @pytest.mark.unit
    def test_basic_functionality(self, device):
        """Test basic functionality."""
        # Arrange
        test_input = torch.randn(2, 3, 32, 32).to(device)

        # Act
        result = some_function(test_input)

        # Assert
        assert result is not None
        assert result.shape == (2, 10)
```

### Naming Conventions
- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use descriptive names: `test_small_cnn_forward_pass`

### Best Practices
1. Use fixtures instead of creating duplicates
2. Test one behavior per test
3. Use markers properly
4. Write clear assertions
5. Use deterministic seeds

## Test Details

### test_models.py
Tests for model architectures:
- Creation and initialization
- Forward pass with valid input
- Output dtype and shape
- Parameter count (1.5M, 620K, 67K)
- Requires grad flag
- Backward pass and gradients
- Save and load functionality
- Model factory functions

### test_training_pipeline.py
Tests for training components:
- Loss functions: Cross entropy, label smoothing, KL divergence
- Gradient flow and backpropagation
- Learning rate schedulers
- Argument parsing
- Config module imports
- Normalization values structure
- Accuracy calculation
- Confusion matrix
- Early stopping logic
- Gradient clipping

### test_integration.py
Tests for complete workflows:
- CIFAR-10 dataset loading
- MNIST dataset loading
- Data normalization
- Data augmentation
- Single training step
- Validation step
- Mini epoch training
- Train-eval cycle
- Checkpoint save/load
- ONNX export
- ONNX inference

### test_regression.py
Tests for consistency:
- Deterministic output with same seed
- Output validity (no NaN/Inf)
- Output shape consistency
- Batch-size independence
- Loss bounds validation
- Loss determinism
- Gradient flow verification
- Weight initialization range
- Bias initialization
- Initialization reproducibility
- Layer count consistency
- Parameter count stability
- Train/eval mode differences
- Batch norm isolation
- Gradient accumulation

## Project Structure

```
tests/
├── __init__.py
├── conftest.py              # Fixtures and configuration
├── test_models.py           # Model architecture tests
├── test_training_pipeline.py # Training component tests
├── test_integration.py      # Workflow tests
└── test_regression.py       # Consistency tests

pytest.ini                    # Pytest configuration
.gitlab/test_requirements.txt # Dependencies
.gitlab-ci.yml               # CI/CD with unit_tests stage
```

## References

- See individual test file docstrings for specific test details
- Check conftest.py for available fixtures
- See pytest.ini for test markers and configuration
- See .gitlab-ci.yml for CI/CD pipeline integration
