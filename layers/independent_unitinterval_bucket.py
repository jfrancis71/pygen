import math
import torch
import pygen.distributions.unitinterval_bucket as bucket_dist


class IndependentUnitIntervalBucket(torch.nn.Module):
    def __init__(self, event_shape):
        super().__init__()
        self.event_shape = event_shape

    def params_size(self):
        return math.prod(self.event_shape)*8

    def forward(self, logits):  # logits B, Y, X, P
        batch_shape = list(logits.shape[:-len(self.event_shape)])
        reshape_logits = logits.reshape(batch_shape + self.event_shape + [8])
        base_distribution = bucket_dist.UnitIntervalBucket(logits=reshape_logits)
        return torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims=len(self.event_shape))
