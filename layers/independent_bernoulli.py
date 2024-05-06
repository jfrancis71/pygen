import math
import torch


class IndependentBernoulli(torch.nn.Module):
    def __init__(self, event_shape):
        super().__init__()
        self.event_shape = event_shape

    def params_size(self):
        return math.prod(self.event_shape)

    def forward(self, logits):
        batch_shape = list(logits.shape[:-1])
        reshape_logits = logits.reshape(batch_shape + self.event_shape)
        base_distribution = torch.distributions.bernoulli.Bernoulli(logits=reshape_logits)
        return torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims=len(self.event_shape))
