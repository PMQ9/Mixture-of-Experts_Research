"""
Pytest configuration and fixtures for Mixture-of-Experts tests.

This module provides shared test fixtures and configuration for all tests.
"""

import os
import sys
import pytest
import torch
import tempfile
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "Vision_Transformer_Pytorch"))


@pytest.fixture(scope="session")
def device():
    """Return appropriate device for testing (CPU for CI, GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp(prefix="moe_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def artifacts_dir(temp_dir):
    """Create artifacts directory for test outputs."""
    artifacts = Path(temp_dir) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory path."""
    return project_root / "data"


@pytest.fixture
def small_batch():
    """Create a small batch of dummy data for testing."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def small_labels():
    """Create dummy labels for a small batch."""
    return torch.randint(0, 10, (2,))


@pytest.fixture
def config_small_vit():
    """Create a minimal VisionTransformer config for fast testing."""
    from vision_transformer_moe import VisionTransformerConfig

    return VisionTransformerConfig(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        num_experts=2,
        top_k=1,
        num_classes=10,
        balance_loss_weight=0.01,
    )


@pytest.fixture
def model_small_cnn(device):
    """Create a small CNN model for testing."""
    from small_expert import SmallExpertCNN

    model = SmallExpertCNN(num_classes=10)
    return model.to(device)


@pytest.fixture
def model_tiny_cnn(device):
    """Create a tiny CNN model for testing."""
    from small_expert import TinyExpertCNN

    model = TinyExpertCNN(num_classes=10)
    return model.to(device)


@pytest.fixture
def model_micro_cnn(device):
    """Create a micro CNN model for testing."""
    from small_expert import MicroExpertCNN

    model = MicroExpertCNN(num_classes=10)
    return model.to(device)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as regression tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before running tests."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Use deterministic algorithms if possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    yield

    # Cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
