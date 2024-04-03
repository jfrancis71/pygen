import torch
import torch.nn as nn
from pygen.distributions.unitinterval_bucket import UnitIntervalBucket
import pixelcnn_pp.model as pixelcnn_model


class _PixelCNN(nn.Module):
    def __init__(self, event_shape, params, nr_resnet=3):
        super().__init__()
        self.event_shape = event_shape
        self.batch_shape = []
        self.pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
            input_channels=self.event_shape[0], nr_params=params, nr_conditional=None)

    def log_prob(self, samples):
        if samples.size()[1:4] != torch.Size(self.event_shape):
            raise RuntimeError("sample shape {}, but event_shape has shape {}"
                .format(samples.shape[1:4], self.event_shape))
        logits = self.pixelcnn_net((samples*2.0)-1.0, conditional=None)
        return self._log_prob(logits, samples).sum((1, 2, 3))

    def sample(self, sample_shape=None):
        if sample_shape is None:
            batch_shape = [1]
        else:
            batch_shape = sample_shape
        with torch.no_grad():
            sample = torch.zeros(batch_shape+self.event_shape, device=next(self.parameters()).device)
            for y in range(self.event_shape[1]):
                for x in range(self.event_shape[2]):
                    logits = self.pixelcnn_net((sample*2)-1, sample=True, conditional=None)
                    pixel_sample = self._sample(logits)
                    sample[:, :, y, x] = pixel_sample[:, :, y, x]
        if sample_shape is None:
            return sample[0]
        else:
            return sample


class PixelCNNBernoulliDistribution(_PixelCNN):
    def __init__(self, event_shape, nr_resnet=3):
        super().__init__(event_shape, 1, nr_resnet=nr_resnet)

    def _log_prob(self, logits, samples):
        return torch.distributions.bernoulli.Bernoulli(logits=logits).log_prob(samples)

    def _sample(self, logits):
        return torch.distributions.bernoulli.Bernoulli(logits=logits).sample()


class PixelCNNUnitIntervalBucketDistribution(_PixelCNN):
    def __init__(self, event_shape, nr_resnet=3):
        super().__init__(event_shape, 24, nr_resnet=nr_resnet)

    def _log_prob(self, logits, samples):
        B, C, Y, X = samples.shape
        reshape_logits = logits.permute(0, 2, 3, 1).reshape(B, Y, X, 3, 8).permute(0, 3, 1, 2, 4)
        return UnitIntervalBucket(logits=reshape_logits).log_prob(samples)

    def _sample(self, logits):
        B, C, Y, X = logits.shape
        reshape_logits = logits.permute(0, 2, 3, 1).reshape(B, Y, X, 3, 8).permute(0, 3, 1, 2, 4)
        return UnitIntervalBucket(logits=reshape_logits).sample()
