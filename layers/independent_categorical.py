"""
Defines IndependentCategorical layer.
"""


import math
import torch


class IndependentCategorical(torch.nn.Module):
    """Layer which accepts a tensor and returns a Categorical probability distribution.

    Examples:
        >>> categorical_layer = IndependentCategorical(event_shape=[], num_classes=10)
        >>> distribution = categorical_layer(torch.rand([7, 10]))
        >>> distribution.batch_shape
        torch.Size([7])
        >>> multivariate_layer = IndependentCategorical(event_shape=[5], num_classes=10)
        >>> multivariate_distribution = multivariate_layer(torch.rand([7, 5*10]))
        >>> multivariate_distribution.sample([12]).shape
        torch.Size([12, 7, 5])
    """
    def __init__(self, event_shape, num_classes):
        super().__init__()
        self.event_shape = event_shape
        self.num_classes = num_classes

    def params_size(self):
        return math.prod(self.event_shape)*self.num_classes

    def forward(self, logits):
        batch_shape = list(logits.shape[:-1])
        reshape_logits = logits.reshape(batch_shape + self.event_shape + [self.num_classes])
        base_distribution = torch.distributions.categorical.Categorical(logits=reshape_logits)
        return torch.distributions.independent.Independent(
            base_distribution,
            reinterpreted_batch_ndims=len(self.event_shape))


import doctest
doctest.testmod()
