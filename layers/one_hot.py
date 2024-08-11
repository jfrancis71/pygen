from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return F.one_hot(x, self.num_classes).float()