"""
Integration tests for the complete training pipeline.

Tests the following:
- End-to-end training runs (mini)
- Data loading and preprocessing
- Model training and validation
- ONNX export
- Checkpoint saving and loading
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


class TestDataLoading:
    """Test suite for data loading and preprocessing."""

    @pytest.mark.integration
    def test_cifar10_loading(self):
        """Test CIFAR-10 dataset loading."""
        try:
            from torchvision import datasets, transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            # Load small subset
            train_dataset = datasets.CIFAR10(
                root='/tmp/cifar10',
                train=True,
                download=False,  # Don't download in tests
                transform=transform
            )

            assert len(train_dataset) > 0

            # Test data loading
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0
            )

            batch_images, batch_labels = next(iter(dataloader))

            assert batch_images.shape == (4, 3, 32, 32)
            assert batch_labels.shape == (4,)

        except (FileNotFoundError, RuntimeError):
            pytest.skip("CIFAR-10 dataset not available")

    @pytest.mark.integration
    def test_mnist_loading(self):
        """Test MNIST dataset loading."""
        try:
            from torchvision import datasets, transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            # Load small subset
            train_dataset = datasets.MNIST(
                root='/tmp/mnist',
                train=True,
                download=False,
                transform=transform
            )

            assert len(train_dataset) > 0

            # Test data loading
            dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0
            )

            batch_images, batch_labels = next(iter(dataloader))

            assert batch_images.shape == (4, 1, 28, 28)
            assert batch_labels.shape == (4,)

        except (FileNotFoundError, RuntimeError):
            pytest.skip("MNIST dataset not available")

    @pytest.mark.unit
    def test_data_normalization(self):
        """Test data normalization functionality."""
        from torchvision import transforms

        # Create normalized images
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Create dummy image
        image = torch.ones(3, 32, 32)
        normalized = normalize(image)

        # Check that normalization was applied
        assert not torch.allclose(image, normalized)

    @pytest.mark.unit
    def test_data_augmentation(self):
        """Test data augmentation transforms."""
        from torchvision import transforms

        augmentation = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        # Create dummy PIL image
        from PIL import Image
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        augmented = augmentation(img)
        assert augmented.shape == (3, 32, 32)


class TestMiniTrainingRun:
    """Test suite for mini training runs."""

    @pytest.mark.integration
    def test_training_step(self, model_small_cnn, device):
        """Test a single training step."""
        model = model_small_cnn
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Create dummy batch
        batch_images = torch.randn(4, 3, 32, 32).to(device)
        batch_labels = torch.randint(0, 10, (4,)).to(device)

        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    @pytest.mark.integration
    def test_validation_step(self, model_small_cnn, device):
        """Test a validation step."""
        model = model_small_cnn
        model.eval()

        criterion = nn.CrossEntropyLoss()

        # Create dummy batch
        batch_images = torch.randn(4, 3, 32, 32).to(device)
        batch_labels = torch.randint(0, 10, (4,)).to(device)

        with torch.no_grad():
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            accuracy = (predictions == batch_labels).sum().item() / batch_labels.size(0)

        assert loss.item() >= 0
        assert 0 <= accuracy <= 1

    @pytest.mark.integration
    @pytest.mark.slow
    def test_mini_epoch(self, model_small_cnn, device):
        """Test training for a mini epoch."""
        model = model_small_cnn
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        num_batches = 5
        batch_size = 4

        total_loss = 0.0

        for _ in range(num_batches):
            batch_images = torch.randn(batch_size, 3, 32, 32).to(device)
            batch_labels = torch.randint(0, 10, (batch_size,)).to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        assert avg_loss > 0
        assert not torch.isnan(torch.tensor(avg_loss))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_train_eval_cycle(self, model_small_cnn, device):
        """Test complete train-eval cycle."""
        model = model_small_cnn
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        num_train_batches = 5
        num_eval_batches = 3

        # Training phase
        model.train()
        train_losses = []

        for _ in range(num_train_batches):
            batch_images = torch.randn(4, 3, 32, 32).to(device)
            batch_labels = torch.randint(0, 10, (4,)).to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Evaluation phase
        model.eval()
        eval_losses = []
        eval_accuracies = []

        with torch.no_grad():
            for _ in range(num_eval_batches):
                batch_images = torch.randn(4, 3, 32, 32).to(device)
                batch_labels = torch.randint(0, 10, (4,)).to(device)

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

                _, predictions = torch.max(outputs, 1)
                accuracy = (predictions == batch_labels).sum().item() / batch_labels.size(0)

                eval_losses.append(loss.item())
                eval_accuracies.append(accuracy)

        # Assertions
        assert len(train_losses) == num_train_batches
        assert len(eval_losses) == num_eval_batches
        assert all(0 <= acc <= 1 for acc in eval_accuracies)


class TestCheckpointing:
    """Test suite for model checkpointing."""

    @pytest.mark.integration
    def test_save_checkpoint(self, model_small_cnn, temp_dir):
        """Test saving a checkpoint."""
        checkpoint_path = Path(temp_dir) / "checkpoint.pth"

        checkpoint = {
            'model_state_dict': model_small_cnn.state_dict(),
            'epoch': 1,
            'loss': 0.5,
        }

        torch.save(checkpoint, checkpoint_path)
        assert checkpoint_path.exists()

    @pytest.mark.integration
    def test_load_checkpoint(self, model_small_cnn, temp_dir, device):
        """Test loading a checkpoint."""
        checkpoint_path = Path(temp_dir) / "checkpoint_load.pth"

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model_small_cnn.state_dict(),
            'epoch': 5,
            'loss': 0.3,
        }
        torch.save(checkpoint, checkpoint_path)

        # Create new model and load checkpoint
        from small_expert import SmallExpertCNN
        new_model = SmallExpertCNN(num_classes=10).to(device)
        loaded_checkpoint = torch.load(checkpoint_path)

        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])

        assert loaded_checkpoint['epoch'] == 5
        assert loaded_checkpoint['loss'] == 0.3

        # Test that loaded model produces same output (both in eval mode)
        model_small_cnn.eval()
        new_model.eval()

        test_input = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            output1 = model_small_cnn(test_input)
            output2 = new_model(test_input)

        assert torch.allclose(output1, output2, atol=1e-5)


class TestONNXExport:
    """Test suite for ONNX export functionality."""

    @pytest.mark.integration
    def test_onnx_export(self, model_small_cnn, temp_dir, device):
        """Test ONNX model export."""
        try:
            import onnx
            import onnxscript
        except ImportError:
            pytest.skip("ONNX or onnxscript not installed")

        model = model_small_cnn
        model.eval()

        onnx_path = Path(temp_dir) / "model.onnx"

        # Create dummy input
        dummy_input = torch.randn(1, 3, 32, 32).to(device)

        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=14,
                verbose=False
            )

            # Verify ONNX model was created
            assert onnx_path.exists()

            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            assert onnx_model is not None

        except ModuleNotFoundError:
            pytest.skip("ONNX export dependencies not fully available")

    @pytest.mark.integration
    def test_onnx_inference(self, model_small_cnn, temp_dir, device):
        """Test inference with ONNX model."""
        try:
            import onnx
            import onnxruntime as ort
            import onnxscript
        except ImportError:
            pytest.skip("ONNX, onnxruntime, or onnxscript not installed")

        model = model_small_cnn
        model.eval()

        onnx_path = Path(temp_dir) / "model_inference.onnx"

        dummy_input = torch.randn(1, 3, 32, 32).to(device)

        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                opset_version=14,
            )

            # Load ONNX model
            sess = ort.InferenceSession(str(onnx_path))

            # Prepare input
            test_input = dummy_input.cpu().numpy()

            # Run inference
            outputs = sess.run(None, {'input': test_input})

            assert len(outputs) > 0
            assert outputs[0].shape == (1, 10)

        except ModuleNotFoundError:
            pytest.skip("ONNX export dependencies not fully available")
