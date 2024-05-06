import math
import torch
import pygen.distributions.quantized_distribution as qd


class IndependentQuantizedDistribution(torch.nn.Module):
    def __init__(self, event_shape, num_buckets=8):
        super().__init__()
        self.event_shape = event_shape
        self.num_buckets = num_buckets

    def params_size(self):
        return math.prod(self.event_shape)*self.num_buckets

    def forward(self, logits):  # logits, eg. B, Y, X, P where batch_shape would be B, Y, X
        batch_shape = list(logits.shape[:-1])
        reshape_logits = logits.reshape(batch_shape + self.event_shape + [self.num_buckets])
        base_distribution = qd.QuantizedDistribution(logits=reshape_logits)
        return torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims=len(self.event_shape))
