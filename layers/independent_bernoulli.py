import math
import torch


class IndependentBernoulli(torch.nn.Module):
    def __init__(self, event_shape):
        super().__init__()
        self.event_shape = event_shape

    def params_size(self):
        return math.prod(self.event_shape)

    def forward(self, logits):
        base_distribution = torch.distributions.bernoulli.Bernoulli(logits=logits)
        return torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims=len(self.event_shape))
