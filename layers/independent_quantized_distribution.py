"""Defines IndependentQuantizedDistribution Layer"""


import math
import torch
import pygen.distributions.quantized_distribution as qd


class IndependentQuantizedDistribution(torch.nn.Module):
    """IndependentQuantizedDistribution layer accepts a tensor describing the logits parameters
       of the quantized variable components and returns an independent quantized distribution.
       You need to specify event_shape to indicate what a sample from this distribution
       looks like, eg event_shape = [3,32,32] would describe a CIFAR10 image where all pixels
       are independent quantized variables.
    """
    def __init__(self, event_shape, num_buckets=8):
        super().__init__()
        self.event_shape = event_shape
        self.num_buckets = num_buckets

    def params_size(self):
        """return number of parameters required to describe distribution"""
        return math.prod(self.event_shape)*self.num_buckets

    # pylint: disable=C0116
    def forward(self, logits):  # logits, eg. B, Y, X, P where batch_shape would be B, Y, X
        batch_shape = list(logits.shape[:-1])
        reshape_logits = logits.reshape(batch_shape + self.event_shape + [self.num_buckets])
        base_distribution = qd.QuantizedDistribution(logits=reshape_logits)
        return torch.distributions.independent.Independent(
            base_distribution,
            reinterpreted_batch_ndims=len(self.event_shape))
