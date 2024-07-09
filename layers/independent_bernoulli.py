"""Defines IndependentBernoulli Layer"""


import math
import torch


class IndependentBernoulli(torch.nn.Module):
    """IndependentBernoulli layer accepts a tensor describing the logits parameters of the
       bernoulli variable components and returns an independent bernoulli distribution.
       You need to specify event_shape to indicate what a sample from this distribution
       looks like, eg event_shape = [1,28,28] would describe an MNIST image where all pixels
       are independent bernoulli variables.
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
