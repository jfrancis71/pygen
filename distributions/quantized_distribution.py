from torch.distributions.categorical import Categorical
import torch


class QuantizedDistribution:
    """Represents a continuous distribution on interval (0,1) which has been"""
    """discretized into num_buckets where num_buckets is size of last logits tensor"""
    def __init__(self, logits):
        self.logits = logits
        self.num_buckets = self.logits.shape[-1]
        self.event_shape = torch.Size([])
        self.batch_shape = self.logits.shape[:-1]
        self.log_buckets = torch.log(torch.tensor(self.num_buckets))

    def log_prob(self, x):
        quantized_samples = torch.clamp((x * self.num_buckets).floor(), 0, self.num_buckets - 1)
        return Categorical(logits=self.logits).log_prob(quantized_samples)

    def sample(self, sample_shape=torch.Size()):
        floor = Categorical(logits=self.logits).sample(sample_shape)/self.num_buckets
        samples = floor + torch.rand_like(floor)/(self.num_buckets*2.0)
        return samples
