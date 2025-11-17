"""
Unit tests for model architectures and creation.

Tests the following:
- Model creation and initialization
- Forward pass execution
- Output shape validation
- Model architecture integrity
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path


class TestSmallExpertCNN:
    """Test suite for SmallExpertCNN architecture."""

    @pytest.mark.unit
    def test_creation(self, model_small_cnn):
        """Test that SmallExpertCNN can be created."""
        assert model_small_cnn is not None
        assert isinstance(model_small_cnn, nn.Module)

    @pytest.mark.unit
    def test_forward_pass(self, model_small_cnn, small_batch, device):
        """Test forward pass with valid input."""
        small_batch = small_batch.to(device)
        output = model_small_cnn(small_batch)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 10)  # batch_size=2, num_classes=10

    @pytest.mark.unit
    def test_output_dtype(self, model_small_cnn, small_batch, device):
        """Test that output has correct dtype."""
        small_batch = small_batch.to(device)
        output = model_small_cnn(small_batch)

        assert output.dtype == torch.float32

    @pytest.mark.unit
    def test_parameter_count(self, model_small_cnn):
        """Test parameter count is reasonable."""
        total_params = sum(p.numel() for p in model_small_cnn.parameters())
        # SmallExpertCNN should have ~1.5M parameters
        assert 1e6 <= total_params <= 2e6, f"Got {total_params} parameters"

    @pytest.mark.unit
    def test_requires_grad(self, model_small_cnn):
        """Test that parameters require gradients."""
        for param in model_small_cnn.parameters():
            assert param.requires_grad


class TestTinyExpertCNN:
    """Test suite for TinyExpertCNN architecture."""

    @pytest.mark.unit
    def test_creation(self, model_tiny_cnn):
        """Test that TinyExpertCNN can be created."""
        assert model_tiny_cnn is not None
        assert isinstance(model_tiny_cnn, nn.Module)

    @pytest.mark.unit
    def test_forward_pass(self, model_tiny_cnn, small_batch, device):
        """Test forward pass with valid input."""
        small_batch = small_batch.to(device)
        output = model_tiny_cnn(small_batch)

        assert output.shape == (2, 10)

    @pytest.mark.unit
    def test_parameter_count(self, model_tiny_cnn):
        """Test parameter count is in expected range."""
        total_params = sum(p.numel() for p in model_tiny_cnn.parameters())
        # TinyExpertCNN should have ~620K parameters
        assert 500e3 <= total_params <= 750e3, f"Got {total_params} parameters"


class TestMicroExpertCNN:
    """Test suite for MicroExpertCNN architecture."""

    @pytest.mark.unit
    def test_creation(self, model_micro_cnn):
        """Test that MicroExpertCNN can be created."""
        assert model_micro_cnn is not None
        assert isinstance(model_micro_cnn, nn.Module)

    @pytest.mark.unit
    def test_forward_pass(self, model_micro_cnn, small_batch, device):
        """Test forward pass with valid input."""
        small_batch = small_batch.to(device)
        output = model_micro_cnn(small_batch)

        assert output.shape == (2, 10)

    @pytest.mark.unit
    def test_parameter_count(self, model_micro_cnn):
        """Test parameter count is minimal."""
        total_params = sum(p.numel() for p in model_micro_cnn.parameters())
        # MicroExpertCNN should have ~67K parameters
        assert 50e3 <= total_params <= 100e3, f"Got {total_params} parameters"


class TestModelBackwardPass:
    """Test suite for backward pass and gradient flow."""

    @pytest.mark.unit
    def test_small_cnn_backward(self, model_small_cnn, small_batch, small_labels, device):
        """Test backward pass for SmallExpertCNN."""
        small_batch = small_batch.to(device)
        small_labels = small_labels.to(device)

        output = model_small_cnn(small_batch)
        loss = nn.CrossEntropyLoss()(output, small_labels)
        loss.backward()

        # Check that gradients are computed
        for name, param in model_small_cnn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    @pytest.mark.unit
    def test_tiny_cnn_backward(self, model_tiny_cnn, small_batch, small_labels, device):
        """Test backward pass for TinyExpertCNN."""
        small_batch = small_batch.to(device)
        small_labels = small_labels.to(device)

        output = model_tiny_cnn(small_batch)
        loss = nn.CrossEntropyLoss()(output, small_labels)
        loss.backward()

        for name, param in model_tiny_cnn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestModelSaveLoad:
    """Test suite for model serialization."""

    @pytest.mark.unit
    def test_save_and_load_small_cnn(self, model_small_cnn, temp_dir, device):
        """Test saving and loading SmallExpertCNN."""
        model_path = Path(temp_dir) / "small_cnn_test.pth"

        torch.save(model_small_cnn.state_dict(), model_path)
        assert model_path.exists()

        loaded_model = type(model_small_cnn)(num_classes=10)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model = loaded_model.to(device)

        # Test that loaded model produces same output
        test_input = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            output_original = model_small_cnn(test_input)
            output_loaded = loaded_model(test_input)

        assert torch.allclose(output_original, output_loaded, atol=1e-5)

    @pytest.mark.unit
    def test_model_eval_mode(self, model_small_cnn, small_batch, device):
        """Test model in eval mode."""
        model_small_cnn.train()
        model_small_cnn.eval()

        small_batch = small_batch.to(device)
        with torch.no_grad():
            output = model_small_cnn(small_batch)

        assert output.shape == (2, 10)


class TestModelWrapper:
    """Test suite for model factory functions."""

    @pytest.mark.unit
    def test_create_model_small_cnn(self, device):
        """Test creating small_cnn via factory."""
        from model_wrapper import create_model

        model = create_model("small_cnn", num_classes=10)
        model = model.to(device)

        assert model is not None
        assert isinstance(model, nn.Module)

        test_input = torch.randn(2, 3, 32, 32).to(device)
        output = model(test_input)
        assert output.shape == (2, 10)

    @pytest.mark.unit
    def test_create_model_tiny_cnn(self, device):
        """Test creating tiny_cnn via factory."""
        from model_wrapper import create_model

        model = create_model("tiny_cnn", num_classes=10)
        model = model.to(device)

        assert model is not None
        test_input = torch.randn(2, 3, 32, 32).to(device)
        output = model(test_input)
        assert output.shape == (2, 10)

    @pytest.mark.unit
    def test_create_model_micro_cnn(self, device):
        """Test creating micro_cnn via factory."""
        from model_wrapper import create_model

        model = create_model("micro_cnn", num_classes=10)
        model = model.to(device)

        assert model is not None
        test_input = torch.randn(2, 3, 32, 32).to(device)
        output = model(test_input)
        assert output.shape == (2, 10)

    @pytest.mark.unit
    def test_create_model_invalid(self):
        """Test that invalid model name raises error."""
        from model_wrapper import create_model

        with pytest.raises((ValueError, KeyError, AttributeError)):
            create_model("invalid_model_name", num_classes=10)
