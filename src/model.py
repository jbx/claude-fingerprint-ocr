"""
CNN model architecture for digit classification.
Designed to work with 32x32 RGB images from the SVHN dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for digit classification.

    Input: 32x32 RGB images (3 channels)
    Output: 10 class probabilities (digits 0-9)
    """

    def __init__(self, in_channels=3):
        super(DigitCNN, self).__init__()

        self.in_channels = in_channels

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully connected layers
        # After conv layers on 32x32 input: 32->16->8->4->2
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # First block: conv -> bn -> relu -> pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4

        # Fourth block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 4x4 -> 2x2

        # Flatten and fully connected
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def predict(self, x):
        """Get predicted class labels."""
        self.eval()
        with torch.inference_mode():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def predict_proba(self, x):
        """Get class probabilities."""
        self.eval()
        with torch.inference_mode():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities


class DigitResNet(nn.Module):
    """
    ResNet-18 based digit classifier adapted for 32x32 inputs.

    Standard ResNet-18 uses a 7x7 stride-2 conv + maxpool as its stem,
    which aggressively downsamples to 56x56 (designed for 224x224 inputs).
    On 32x32 images this crushes spatial info to 8x8 before the residual
    blocks even start. We replace the stem with a 3x3 stride-1 conv and
    skip the maxpool, preserving spatial resolution for small inputs.

    The residual blocks (layer1-4) retain their pretrained ImageNet weights.

    Input: 32x32 RGB images (3 channels)
    Output: 10 class probabilities (digits 0-9)
    """

    def __init__(self, in_channels=3, pretrained=True):
        super(DigitResNet, self).__init__()

        self.in_channels = in_channels

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Replace the 7x7 stride-2 conv with 3x3 stride-1 for 32x32 inputs.
        # This is randomly initialized; pretrained weights only apply to
        # the residual blocks below.
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # No maxpool — go straight into residual blocks

        # Pretrained residual blocks
        self.layer1 = resnet.layer1  # 64ch,  32x32 -> 32x32
        self.layer2 = resnet.layer2  # 128ch, 32x32 -> 16x16
        self.layer3 = resnet.layer3  # 256ch, 16x16 -> 8x8
        self.layer4 = resnet.layer4  # 512ch, 8x8   -> 4x4

        self.avgpool = resnet.avgpool  # -> 512x1x1

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 10),
        )

        # Group pretrained vs new params for differential LR
        self.features = nn.ModuleList([
            self.bn1, self.layer1, self.layer2, self.layer3, self.layer4
        ])
        self.new_params = nn.ModuleList([self.conv1, self.classifier])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # No maxpool for 32x32 inputs
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def pretrained_parameters(self):
        """Parameters from pretrained residual blocks (use lower LR)."""
        for module in self.features:
            yield from module.parameters()

    def new_parameters(self):
        """Newly initialized parameters (use higher LR)."""
        for module in self.new_params:
            yield from module.parameters()

    def predict(self, x):
        """Get predicted class labels."""
        self.eval()
        with torch.inference_mode():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def predict_proba(self, x):
        """Get class probabilities."""
        self.eval()
        with torch.inference_mode():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities
