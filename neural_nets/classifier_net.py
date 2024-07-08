"""Defines ClassifierNet, a simple net for classifying MNIST or CIFAR10"""


import torch
from torch import nn
import torch.nn.functional as F


class ClassifierNet(nn.Module):
    """Simple Classifier Neural Net, can be configured to accept MNIST (1x28x28) or
       CIFAR10 (3x32x32) tensors. Outputs categorical distribution over 10 states."""
    def __init__(self, mnist, num_classes=10):
        super().__init__()
        if mnist:
            in_channels = 1
            mid_channels = 9216
        else:
            in_channels = 3
            mid_channels = 12544
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(mid_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)

    # pylint: disable=C0103,C0116
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # pylint: disable=E1101
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        distribution = torch.distributions.categorical.Categorical(logits=x)
        return distribution
