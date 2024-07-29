"""Defines ClassifierNet, a simple net for classifying MNIST or CIFAR10"""


import torch
from torch import nn
import torch.nn.functional as F


class ClassifierNet(nn.Module):
    """A simple classifier type neural net.

    Can be configured to accept MNIST or CIFAR10 sized input tensors.

    Args:
        mnist (Bool): If true accept MNIST shaped tensor, otherwise CIFAR10.
        num_classes (Integer): size of output tensor.

    Example:
        >>> classifier_net = ClassifierNet(True, num_classes=8)
        >>> classifier_net(torch.rand([32, 1, 28, 28])).shape
        torch.Size([32, 8])
    """
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
        return x  # We do not return a distribution, as some clients will want Categorical, OneHot, Multivariate...


import doctest
doctest.testmod()
