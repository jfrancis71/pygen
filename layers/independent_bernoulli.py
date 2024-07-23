"""Defines IndependentBernoulli Layer"""


import math
import torch


class IndependentBernoulli(torch.nn.Module):
    """Layer which accepts a tensor and returns an independent bernoulli probability distribution.

    Example::

        >>> independent_bernoulli_layer = IndependentBernoulli([3, 32, 32])
        >>> independent_bernoulli_distribution = independent_bernoulli_layer(torch.rand([7, 3*32*32]))
        >>> independent_bernoulli_distribution.batch_shape
        torch.Size([7])
        >>> independent_bernoulli_distribution.sample([5]).shape
        torch.Size([5, 7, 3, 32, 32])
    """
    def __init__(self, event_shape):
        super().__init__()
        if not isinstance(event_shape, list):
            raise RuntimeError(f"event_shape is {event_shape}, but was expecting a list.")
        self.event_shape = event_shape

    def params_size(self):
        """return number of parameters required to describe distribution"""
        return math.prod(self.event_shape)

    # pylint: disable=C0116
    def forward(self, logits):
        batch_shape = list(logits.shape[:-1])
        reshape_logits = logits.reshape(batch_shape + self.event_shape)
        base_distribution = torch.distributions.bernoulli.Bernoulli(logits=reshape_logits)
        return torch.distributions.independent.Independent(
            base_distribution,
            reinterpreted_batch_ndims=len(self.event_shape))


import doctest
doctest.testmod()
