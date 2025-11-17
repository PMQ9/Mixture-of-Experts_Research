"""
Unit tests for training pipeline components.

Tests the following:
- Loss functions
- Learning rate scheduling
- Argument parsing
- Configuration loading
- Training utilities
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class TestLossFunctions:
    """Test suite for loss functions used in training."""

    @pytest.mark.unit
    def test_crossentropy_loss(self):
        """Test standard cross entropy loss."""
        criterion = nn.CrossEntropyLoss()

        # Create dummy predictions and targets
        output = torch.randn(4, 10)  # batch_size=4, num_classes=10
        target = torch.tensor([0, 1, 2, 3])

        loss = criterion(output, target)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    @pytest.mark.unit
    def test_label_smoothing_loss(self):
        """Test label smoothing cross entropy loss."""
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        output = torch.randn(4, 10)
        target = torch.tensor([0, 1, 2, 3])

        loss = criterion(output, target)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    @pytest.mark.unit
    def test_kl_divergence(self):
        """Test KL divergence loss (for TRADES)."""
        output_clean = torch.randn(4, 10)
        output_adv = torch.randn(4, 10)

        p_clean = F.softmax(output_clean, dim=1)
        p_adv = F.softmax(output_adv, dim=1)

        kl_loss = F.kl_div(F.log_softmax(output_adv, dim=1), p_clean, reduction="batchmean")

        assert kl_loss.item() >= 0
        assert not torch.isnan(kl_loss)

    @pytest.mark.unit
    def test_balance_loss(self):
        """Test balance loss for mixture of experts."""
        batch_size = 4
        num_experts = 2

        # Simulate routing probabilities
        gating_output = torch.randn(batch_size, num_experts)
        gate = F.softmax(gating_output, dim=-1)

        # Balance loss: encourage uniform distribution
        importance = gate.sum(0)
        balance_loss = (importance.std() / importance.mean()).item()

        assert balance_loss >= 0

    @pytest.mark.unit
    def test_loss_nan_detection(self):
        """Test that loss functions detect NaN values."""
        criterion = nn.CrossEntropyLoss()

        # Create NaN outputs
        output = torch.tensor([[float('nan'), 1.0, 2.0], [1.0, 2.0, 3.0]])
        target = torch.tensor([0, 1])

        loss = criterion(output, target)
        assert torch.isnan(loss)


class TestLearningRateScheduling:
    """Test suite for learning rate schedulers."""

    @pytest.mark.unit
    def test_step_lr_scheduler(self):
        """Test StepLR scheduler."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr == 0.1

        # Step through scheduler
        for _ in range(5):
            scheduler.step()

        new_lr = optimizer.param_groups[0]['lr']
        assert abs(new_lr - 0.01) < 1e-5

    @pytest.mark.unit
    def test_cosine_annealing_scheduler(self):
        """Test CosineAnnealingLR scheduler."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # LR should decrease with cosine annealing
        assert lrs[0] > lrs[9]

    @pytest.mark.unit
    def test_warmup_scheduler(self):
        """Test learning rate warmup."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=0.1)

        # Simulate warmup
        warmup_epochs = 5
        total_epochs = 20

        lrs = []
        for epoch in range(total_epochs):
            if epoch < warmup_epochs:
                lr = 0.1 * (epoch + 1) / warmup_epochs
            else:
                lr = 0.1

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            lrs.append(lr)

        # Check warmup increases LR
        assert lrs[0] < lrs[warmup_epochs - 1]
        # Check plateau after warmup
        assert lrs[warmup_epochs] == lrs[warmup_epochs + 1]


class TestArgumentParsing:
    """Test suite for command-line argument parsing."""

    @pytest.mark.unit
    def test_parse_basic_args(self):
        """Test parsing basic training arguments."""
        from train_moe import parse_args

        args = parse_args([
            "--dataset", "CIFAR10",
            "--model_arch", "small_cnn",
            "--epochs", "10",
            "--batch_size", "32"
        ])

        assert args.dataset == "CIFAR10"
        assert args.model_arch == "small_cnn"
        assert args.epochs == 10
        assert args.batch_size == 32

    @pytest.mark.unit
    def test_parse_meta_moe_args(self):
        """Test parsing MetaMoE arguments."""
        from train_moe import parse_args

        args = parse_args([
            "--meta_moe",
            "--gating_backbone", "ultra_verifiable_cnn",
            "--meta_top_k", "1"
        ])

        assert args.meta_moe is True
        assert args.gating_backbone == "ultra_verifiable_cnn"
        assert args.meta_top_k == 1

    @pytest.mark.unit
    def test_parse_adversarial_args(self):
        """Test parsing adversarial training arguments."""
        from train_moe import parse_args

        args = parse_args([
            "--adv_training",
            "--art_attack",
            "--at_mode", "PGD"
        ])

        assert args.adv_training is True
        assert args.art_attack is True
        assert args.at_mode == "PGD"

    @pytest.mark.unit
    def test_parse_default_args(self):
        """Test default argument values."""
        from train_moe import parse_args

        args = parse_args([])

        # Check some sensible defaults exist
        assert hasattr(args, "epochs")
        assert hasattr(args, "batch_size")
        assert hasattr(args, "learning_rate")
        assert args.epochs > 0
        assert args.batch_size > 0


class TestConfigLoading:
    """Test suite for configuration loading."""

    @pytest.mark.unit
    def test_config_module_imports(self):
        """Test that config module can be imported."""
        from config import DEFAULT_PARAMS, NORMALIZATION_VALUES

        assert isinstance(DEFAULT_PARAMS, dict)
        assert isinstance(NORMALIZATION_VALUES, dict)

    @pytest.mark.unit
    def test_normalization_values_structure(self):
        """Test structure of normalization values."""
        from config import NORMALIZATION_VALUES

        # Check that common datasets are present
        common_datasets = ["CIFAR10", "MNIST", "GTSRB"]

        for dataset in common_datasets:
            if dataset in NORMALIZATION_VALUES:
                norm = NORMALIZATION_VALUES[dataset]
                assert "mean" in norm or "means" in norm
                assert "std" in norm or "stds" in norm

    @pytest.mark.unit
    def test_default_params_structure(self):
        """Test structure of default parameters."""
        from config import DEFAULT_PARAMS

        # Check essential keys
        essential_keys = ["epoch", "batch_size", "learning_rate"]

        for key in essential_keys:
            if key in DEFAULT_PARAMS:
                assert isinstance(DEFAULT_PARAMS[key], (int, float))


class TestTrainingHelpers:
    """Test suite for training helper functions."""

    @pytest.mark.unit
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        batch_size = 32
        num_classes = 10

        # Create predictions and targets
        outputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Calculate accuracy
        _, predictions = torch.max(outputs, 1)
        correct = (predictions == targets).sum().item()
        accuracy = correct / batch_size

        assert 0 <= accuracy <= 1

    @pytest.mark.unit
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        num_classes = 5
        batch_size = 20

        predictions = torch.randint(0, num_classes, (batch_size,))
        targets = torch.randint(0, num_classes, (batch_size,))

        # Simple confusion matrix
        conf_matrix = torch.zeros(num_classes, num_classes)
        for pred, target in zip(predictions, targets):
            conf_matrix[target, pred] += 1

        assert conf_matrix.shape == (num_classes, num_classes)
        assert conf_matrix.sum() == batch_size

    @pytest.mark.unit
    def test_early_stopping_logic(self):
        """Test early stopping logic."""
        patience = 3
        best_loss = float('inf')
        patience_counter = 0

        losses = [1.5, 1.4, 1.3, 1.35, 1.36, 1.37, 1.38, 1.39]

        for loss in losses:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Should have stopped after 3 epochs without improvement
        assert patience_counter >= patience

    @pytest.mark.unit
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10) * 100  # Large gradients

        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(param, max_norm)

        grad_norm = param.grad.norm().item()
        assert grad_norm <= max_norm + 1e-5
