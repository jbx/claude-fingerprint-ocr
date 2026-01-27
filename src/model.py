"""
CNN model architecture for digit classification.
Designed to work with 8x8 grayscale digit images from the UCI dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for digit classification.

    Input: 8x8 grayscale images (1 channel)
    Output: 10 class probabilities (digits 0-9)
    """

    def __init__(self):
        super(DigitCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        # After conv layers on 8x8 input: 8x8 -> 4x4 -> 2x2 -> 1x1
        self.fc1 = nn.Linear(128 * 1 * 1, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # First block: conv -> bn -> relu -> pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 4x4 -> 2x2

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 2x2 -> 1x1

        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        """Get predicted class labels."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def predict_proba(self, x):
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities
