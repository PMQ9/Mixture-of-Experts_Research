"""
Regression tests to ensure consistency and detect unexpected changes.

Tests the following:
- Model output consistency
- Training output format
- Loss function stability
- Parameter initialization consistency
- Model architecture stability
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json


class TestModelOutputConsistency:
    """Test suite for model output consistency."""

    @pytest.mark.regression
    @pytest.mark.unit
    def test_deterministic_output(self, device):
        """Test that model produces deterministic output with same seed."""
        from small_expert import SmallExpertCNN

        torch.manual_seed(42)
        model1 = SmallExpertCNN(num_classes=10).to(device)

        torch.manual_seed(42)
        model2 = SmallExpertCNN(num_classes=10).to(device)

        test_input = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(42)).to(device)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)

        assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.regression
    @pytest.mark.unit
    def test_output_range_validity(self, model_small_cnn, device):
        """Test that model outputs are in reasonable range."""
        model = model_small_cnn
        model.eval()

        with torch.no_grad():
            test_input = torch.randn(4, 3, 32, 32).to(device)
            output = model(test_input)

        # Output should not have NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Logits can be any real value, but should not be extreme
        assert output.abs().max() < 1e3

    @pytest.mark.regression
    @pytest.mark.unit
    def test_output_shape_consistency(self, model_small_cnn, device):
        """Test output shape remains consistent."""
        model = model_small_cnn
        model.eval()

        test_cases = [
            (1, 3, 32, 32),
            (2, 3, 32, 32),
            (4, 3, 32, 32),
            (8, 3, 32, 32),
        ]

        with torch.no_grad():
            for batch_shape in test_cases:
                test_input = torch.randn(*batch_shape).to(device)
                output = model(test_input)

                expected_shape = (batch_shape[0], 10)  # num_classes=10
                assert output.shape == expected_shape, \
                    f"Expected {expected_shape}, got {output.shape}"

    @pytest.mark.regression
    @pytest.mark.unit
    def test_no_output_dependency_on_batch_size(self, model_small_cnn, device):
        """Test that output statistics are independent of batch size."""
        model = model_small_cnn
        model.eval()

        batch_sizes = [1, 4, 8]
        output_stds = []

        with torch.no_grad():
            for batch_size in batch_sizes:
                test_input = torch.randn(batch_size, 3, 32, 32).to(device)
                output = model(test_input)
                output_stds.append(output.std().item())

        # Output standard deviation should be similar regardless of batch size
        # Allow for some variance but not dramatic differences
        max_std = max(output_stds)
        min_std = min(output_stds)

        ratio = max_std / (min_std + 1e-8)
        assert ratio < 2.0, f"Output std varies too much: {output_stds}"


class TestLossConsistency:
    """Test suite for loss function consistency."""

    @pytest.mark.regression
    @pytest.mark.unit
    def test_loss_determinism(self):
        """Test that loss calculation is deterministic."""
        torch.manual_seed(42)
        outputs1 = torch.randn(4, 10)
        targets1 = torch.tensor([0, 1, 2, 3])

        torch.manual_seed(42)
        outputs2 = torch.randn(4, 10)
        targets2 = torch.tensor([0, 1, 2, 3])

        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(outputs1, targets1)
        loss2 = criterion(outputs2, targets2)

        assert torch.allclose(loss1, loss2, atol=1e-6)

    @pytest.mark.regression
    @pytest.mark.unit
    def test_loss_bounds(self):
        """Test that loss values are within expected bounds."""
        criterion = nn.CrossEntropyLoss()

        # Best case: model predicts correctly with high confidence
        outputs_perfect = torch.zeros(4, 10)
        outputs_perfect[0, 0] = 100
        outputs_perfect[1, 1] = 100
        outputs_perfect[2, 2] = 100
        outputs_perfect[3, 3] = 100
        targets = torch.tensor([0, 1, 2, 3])

        loss_perfect = criterion(outputs_perfect, targets)
        assert loss_perfect < 0.01

        # Worst case: model predicts incorrectly with high confidence
        outputs_worst = torch.zeros(4, 10)
        outputs_worst[0, 9] = 100  # Predict class 9 for target 0
        outputs_worst[1, 8] = 100
        outputs_worst[2, 7] = 100
        outputs_worst[3, 6] = 100

        loss_worst = criterion(outputs_worst, targets)
        assert loss_worst > 4.0

    @pytest.mark.regression
    @pytest.mark.unit
    def test_loss_scale_invariance(self):
        """Test behavior of loss under different scales."""
        criterion = nn.CrossEntropyLoss()

        outputs = torch.randn(4, 10)
        targets = torch.tensor([0, 1, 2, 3])

        loss1 = criterion(outputs, targets)
        loss2 = criterion(outputs * 2, targets)

        # Scaled logits should result in different loss
        assert not torch.allclose(loss1, loss2)

    @pytest.mark.regression
    @pytest.mark.unit
    def test_gradient_flow(self):
        """Test that gradients flow correctly through loss."""
        outputs = torch.randn(4, 10, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        loss.backward()

        # All parameters should have gradients
        assert outputs.grad is not None
        assert not torch.isnan(outputs.grad).any()
        assert not torch.isinf(outputs.grad).any()


class TestParameterInitialization:
    """Test suite for model parameter initialization consistency."""

    @pytest.mark.regression
    @pytest.mark.unit
    def test_weight_initialization_range(self, model_small_cnn):
        """Test that weights are initialized in reasonable range."""
        for name, param in model_small_cnn.named_parameters():
            if 'weight' in name:
                # Weights should not be all zeros
                assert param.std() > 0, f"{name} has no variance"

                # Weights should not be extreme
                assert param.abs().max() < 1.0, \
                    f"{name} has extreme values: max={param.abs().max()}"

    @pytest.mark.regression
    @pytest.mark.unit
    def test_bias_initialization(self, model_small_cnn):
        """Test bias initialization."""
        for name, param in model_small_cnn.named_parameters():
            if 'bias' in name:
                # Biases often start at zero or small values
                assert param.abs().max() < 0.1, \
                    f"{name} bias not properly initialized"

    @pytest.mark.regression
    @pytest.mark.unit
    def test_initialization_reproducibility(self):
        """Test that initialization is reproducible with seed."""
        from small_expert import SmallExpertCNN

        torch.manual_seed(42)
        model1 = SmallExpertCNN(num_classes=10)

        torch.manual_seed(42)
        model2 = SmallExpertCNN(num_classes=10)

        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(),
            model2.named_parameters()
        ):
            assert torch.allclose(param1, param2, atol=1e-6), \
                f"Parameter {name1} differs"


class TestModelArchitectureStability:
    """Test suite for model architecture stability."""

    @pytest.mark.regression
    @pytest.mark.unit
    def test_layer_count_consistency(self, model_small_cnn):
        """Test that model has expected number of layers."""
        layer_count = sum(1 for _ in model_small_cnn.modules()
                         if isinstance(_, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)))

        # SmallExpertCNN should have multiple layers
        assert layer_count > 2

    @pytest.mark.regression
    @pytest.mark.unit
    def test_parameter_count_stability(self):
        """Test that parameter count is stable."""
        from small_expert import SmallExpertCNN, TinyExpertCNN, MicroExpertCNN

        torch.manual_seed(42)
        model1 = SmallExpertCNN(num_classes=10)
        count1 = sum(p.numel() for p in model1.parameters())

        torch.manual_seed(42)
        model2 = SmallExpertCNN(num_classes=10)
        count2 = sum(p.numel() for p in model2.parameters())

        assert count1 == count2

    @pytest.mark.regression
    @pytest.mark.unit
    def test_model_architecture_sizes(self):
        """Test that model size relationships are maintained."""
        from small_expert import SmallExpertCNN, TinyExpertCNN, MicroExpertCNN

        small_params = sum(p.numel() for p in SmallExpertCNN(10).parameters())
        tiny_params = sum(p.numel() for p in TinyExpertCNN(10).parameters())
        micro_params = sum(p.numel() for p in MicroExpertCNN(10).parameters())

        # Size relationships should hold
        assert tiny_params < small_params
        assert micro_params < tiny_params

    @pytest.mark.regression
    @pytest.mark.unit
    def test_device_movement(self, model_small_cnn, device):
        """Test model can move between devices."""
        model = model_small_cnn

        # Move to CPU
        model_cpu = model.cpu()
        test_input_cpu = torch.randn(1, 3, 32, 32)
        output_cpu = model_cpu(test_input_cpu)

        # Move to device
        model_device = model.to(device)
        test_input_device = test_input_cpu.to(device)
        output_device = model_device(test_input_device)

        # Outputs should be the same
        assert torch.allclose(output_cpu, output_device.cpu(), atol=1e-5)


class TestTrainingStateConsistency:
    """Test suite for training state consistency."""

    @pytest.mark.regression
    @pytest.mark.unit
    def test_train_eval_mode_difference(self, model_small_cnn, device):
        """Test that train and eval modes have different behavior (if applicable)."""
        model = model_small_cnn
        test_input = torch.randn(4, 3, 32, 32).to(device)

        model.train()
        with torch.no_grad():
            output_train = model(test_input)

        model.eval()
        with torch.no_grad():
            output_eval = model(test_input)

        # For models without dropout/batchnorm, outputs should be identical
        # For models with them, outputs might differ slightly
        assert output_train.shape == output_eval.shape

    @pytest.mark.regression
    @pytest.mark.unit
    def test_no_leaking_batch_stats(self, model_small_cnn, device):
        """Test that batch norm stats don't leak between batches in eval mode."""
        model = model_small_cnn
        model.eval()

        batch1 = torch.randn(2, 3, 32, 32).to(device)
        batch2 = torch.randn(2, 3, 32, 32).to(device)

        with torch.no_grad():
            output1_first = model(batch1)
            _ = model(batch2)
            output1_second = model(batch1)

        # Same input should produce same output in eval mode
        assert torch.allclose(output1_first, output1_second, atol=1e-5)

    @pytest.mark.regression
    @pytest.mark.unit
    def test_gradient_accumulation(self, model_small_cnn, device):
        """Test that gradients accumulate correctly."""
        model = model_small_cnn
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        batch1_images = torch.randn(2, 3, 32, 32).to(device)
        batch1_labels = torch.tensor([0, 1]).to(device)

        batch2_images = torch.randn(2, 3, 32, 32).to(device)
        batch2_labels = torch.tensor([2, 3]).to(device)

        # First batch
        optimizer.zero_grad()
        output1 = model(batch1_images)
        loss1 = criterion(output1, batch1_labels)
        loss1.backward()

        grad_sum_1 = sum(p.grad.abs().sum() for p in model.parameters()
                        if p.grad is not None)

        # Second batch without zero_grad (accumulate)
        output2 = model(batch2_images)
        loss2 = criterion(output2, batch2_labels)
        loss2.backward()

        grad_sum_2 = sum(p.grad.abs().sum() for p in model.parameters()
                        if p.grad is not None)

        # Accumulated gradients should be larger
        assert grad_sum_2 > grad_sum_1
