from torch.distributions.categorical import Categorical
import torch


class UnitIntervalBucket:
    def __init__(self, logits):
        self.logits = logits
        self.num_buckets = self.logits.shape[-1]
        self.event_shape = torch.Size([])
        self.batch_shape = self.logits.shape[:-1]

    def log_prob(self, x):
        quantized_samples = torch.clamp((x * self.num_buckets).floor(), 0, self.num_buckets - 1)
        return Categorical(logits=self.logits).log_prob(quantized_samples)

    def sample(self, sample_shape=torch.Size()):  # TODO should really be a uniform sample within the bucket
        return Categorical(logits=self.logits).sample(sample_shape)/self.num_buckets + 1.0/(self.num_buckets*2.0)
