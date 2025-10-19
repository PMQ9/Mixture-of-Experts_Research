"""
Small CNN Expert Architecture for MoE
Target: ~1M parameters, verification-friendly
Explicit architecture for neural network verification tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallExpertCNN(nn.Module):
    """
    Small CNN Expert for MoE system
    Input: 32x32x3 RGB images
    Architecture: 4 conv blocks + 2 FC layers
    Parameters: ~1.05M
    """

    def __init__(self, num_classes=10):
        super(SmallExpertCNN, self).__init__()
        self.num_classes = num_classes

        # Block 1: 32x32x3 -> 16x16x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16

        # Block 2: 16x16x64 -> 8x8x128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8

        # Block 3: 8x8x128 -> 4x4x256
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4

        # Block 4: 4x4x256 -> 2x2x256
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)  # [batch_size, 256*2*2]

        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyExpertCNN(nn.Module):
    """
    Even smaller CNN Expert for MoE system
    Input: 32x32x3 RGB images
    Architecture: 3 conv blocks + 2 FC layers
    Parameters: ~350K
    """

    def __init__(self, num_classes=10):
        super(TinyExpertCNN, self).__init__()
        self.num_classes = num_classes

        # Block 1: 32x32x3 -> 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x32 -> 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x64 -> 4x4x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)  # [batch_size, 128*4*4]

        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MicroExpertCNN(nn.Module):
    """
    Micro CNN Expert - verification-optimized
    Input: 32x32x3 RGB images
    Architecture: 3 conv blocks + 1 FC layer (minimal for verification)
    Parameters: ~180K
    Optimized for formal verification tools (fewer layers, simpler structure)
    """

    def __init__(self, num_classes=10, use_maxpool=True):
        super(MicroExpertCNN, self).__init__()
        self.num_classes = num_classes
        self.use_maxpool = use_maxpool

        if use_maxpool:
            # Original architecture with MaxPooling
            # Block 1: 32x32x3 -> 16x16x32
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Block 2: 16x16x32 -> 8x8x64
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Block 3: 8x8x64 -> 4x4x64
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            # NNV-compatible: Strided convolutions instead of MaxPool
            # Block 1: 32x32x3 -> 16x16x32
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.bn1 = nn.BatchNorm2d(32)

            # Block 2: 16x16x32 -> 8x8x64
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.bn2 = nn.BatchNorm2d(64)

            # Block 3: 8x8x64 -> 4x4x64
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layer
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        if self.use_maxpool:
            # Block 1
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)

            # Block 2
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)

            # Block 3
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
        else:
            # NNV-compatible forward pass (no pooling)
            # Block 1
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)

            # Block 2
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)

            # Block 3
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NNVCompatibleCNN(nn.Module):
    """
    NNV-Compatible CNN Expert - Uses AvgPool instead of MaxPool
    Input: 32x32x3 RGB images
    Architecture: 3 conv blocks + 1 FC layer (matches MicroExpertCNN)
    Parameters: ~67K (optimized for formal verification)

    Key features for NNV compatibility:
    - AvgPool instead of MaxPooling (linear operation, NNV-friendly)
    - BatchNorm (foldable into conv layers)
    - ReLU activation (piecewise linear)
    - Single FC layer (minimal complexity)
    """

    def __init__(self, num_classes=10):
        super(NNVCompatibleCNN, self).__init__()
        self.num_classes = num_classes

        # Block 1: 32x32x3 -> 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x32 -> 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x64 -> 4x4x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UltraVerifiableCNN(nn.Module):
    """
    Ultra-Verifiable CNN - Designed specifically for formal verification tools

    **V2 UPDATE:** Increased capacity for better accuracy while maintaining verifiability
    - Channels: 20→28→40→56 (balanced growth, ~85% of NNVCompatibleCNN width)
    - Added 4th conv layer for better feature extraction
    - Small FC hidden layer (64 features for lightweight classification)
    - Verification-friendly: 96K params (4% smaller than NNVCompatibleCNN)

    Design principles from verification literature:
    1. MODERATE WIDTH: 20-56 channels (smaller than 32-64-64, but sufficient)
    2. CONTROLLED DEPTH: 4 conv + 2 FC (one more conv, better features)
    3. AVGPOOL: Use average pooling (linear, LP-friendly)
    4. SMALL FC: 64-unit hidden layer (lightweight classifier)

    Input: 32x32x3 RGB images
    Architecture: 4 conv blocks + 2 FC layers
    Parameters: ~96K (4% smaller than NNVCompatibleCNN)

    Expected NNV layers: 9-10 (vs 13 before, still 23-31% reduction)
    Expected verification time: 10-15 minutes (vs 20+ minutes)
    Expected accuracy: 88-94% (vs 95%+ for larger models, 58% for V1)

    Trade-off: Balanced accuracy vs verifiability
    """

    def __init__(self, num_classes=43):
        super(UltraVerifiableCNN, self).__init__()
        self.num_classes = num_classes

        # Block 1: 32x32x3 -> 16x16x20
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x20 -> 8x8x28
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=28, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(28)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x28 -> 4x4x40
        self.conv3 = nn.Conv2d(in_channels=28, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(40)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 4: 4x4x40 -> 4x4x56
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=56, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(56)

        # Fully connected layers (reduced size for verification)
        self.fc1 = nn.Linear(56 * 4 * 4, 64)  # Small hidden layer (56 channels from conv4)
        self.dropout = nn.Dropout(p=0.3)  # Light dropout
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 4 (no pooling)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Feature extractor variants for use as gating network backbones
class SmallExpertCNN_Features(nn.Module):
    """
    Feature extractor variant of SmallExpertCNN
    Returns features instead of class predictions
    Feature dimension: 512
    """

    def __init__(self):
        super(SmallExpertCNN_Features, self).__init__()
        self.num_features = 512

        # Block 1: 32x32x3 -> 16x16x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x64 -> 8x8x128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x128 -> 4x4x256
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: 4x4x256 -> 2x2x256
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Feature layer (no dropout for feature extraction)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Return features
        x = self.fc1(x)
        x = F.relu(x)

        return x


class TinyExpertCNN_Features(nn.Module):
    """
    Feature extractor variant of TinyExpertCNN
    Returns features instead of class predictions
    Feature dimension: 256
    """

    def __init__(self):
        super(TinyExpertCNN_Features, self).__init__()
        self.num_features = 256

        # Block 1: 32x32x3 -> 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x32 -> 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x64 -> 4x4x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Feature layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Return features
        x = self.fc1(x)
        x = F.relu(x)

        return x


class MicroExpertCNN_Features(nn.Module):
    """
    Feature extractor variant of MicroExpertCNN
    Returns features instead of class predictions
    Feature dimension: 1024 (flattened conv features)
    """

    def __init__(self):
        super(MicroExpertCNN_Features, self).__init__()
        self.num_features = 1024  # 64 * 4 * 4

        # Block 1: 32x32x3 -> 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x32 -> 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x64 -> 4x4x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten and return features
        x = x.view(x.size(0), -1)

        return x


class NNVCompatibleCNN_Features(nn.Module):
    """
    Feature extractor variant of NNVCompatibleCNN
    Returns features instead of class predictions
    Feature dimension: 1024 (64 * 4 * 4 flattened conv features)
    """

    def __init__(self):
        super(NNVCompatibleCNN_Features, self).__init__()
        self.num_features = 1024  # 64 * 4 * 4

        # Block 1: 32x32x3 -> 16x16x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x32 -> 8x8x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x64 -> 4x4x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten and return features
        x = x.view(x.size(0), -1)

        return x


class UltraVerifiableCNN_Features(nn.Module):
    """
    Feature extractor variant of UltraVerifiableCNN
    Returns features instead of class predictions
    Feature dimension: 896 (56 * 4 * 4 flattened conv features)
    """

    def __init__(self):
        super(UltraVerifiableCNN_Features, self).__init__()
        self.num_features = 896  # 56 * 4 * 4

        # Block 1: 32x32x3 -> 16x16x20
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 2: 16x16x20 -> 8x8x28
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=28, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(28)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 3: 8x8x28 -> 4x4x40
        self.conv3 = nn.Conv2d(in_channels=28, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(40)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Block 4: 4x4x40 -> 4x4x56
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=56, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(56)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Block 4 (no pooling)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # Flatten and return features
        x = x.view(x.size(0), -1)

        return x


def print_model_info(model, model_name):
    """Print detailed model information"""
    print(f"\n{'='*60}")
    print(f"{model_name} Architecture")
    print(f"{'='*60}")
    print(model)
    print(f"\n{'='*60}")
    print(f"Total Parameters: {model.count_parameters():,}")
    print(f"{'='*60}\n")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of classes: {model.num_classes}\n")


if __name__ == "__main__":
    # Test all three architectures
    print("Testing Expert Architectures for 32x32 RGB images")

    # SmallExpertCNN (~1M params)
    model_small = SmallExpertCNN(num_classes=43)  # GTSRB has 43 classes
    print_model_info(model_small, "SmallExpertCNN")

    # TinyExpertCNN (~350K params)
    model_tiny = TinyExpertCNN(num_classes=10)  # CIFAR10/MNIST has 10 classes
    print_model_info(model_tiny, "TinyExpertCNN")

    # MicroExpertCNN (~180K params) - optimized for verification
    model_micro = MicroExpertCNN(num_classes=10)
    print_model_info(model_micro, "MicroExpertCNN (Verification-Optimized)")

    print("\nRecommendations:")
    print("- SmallExpertCNN: Good balance, ~1M params, suitable for complex datasets")
    print("- TinyExpertCNN: Lightweight, ~350K params, good for simpler datasets")
    print("- MicroExpertCNN: Minimal, ~180K params, optimized for formal verification tools")
