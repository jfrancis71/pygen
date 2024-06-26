"""
Defines Categorical layer.
"""


import torch


class Categorical(torch.nn.Module):
    """Categorical is a layer which accepts a tensor as input and returns a probability
    distribution. Suitable for use directly in, eg Sequential, presumably as final layer.
    """
    def __init__(self):
        super().__init__()

    # pylint: disable=C0116
    def forward(self, logits):
        return torch.distributions.categorical.Categorical(logits=logits)
