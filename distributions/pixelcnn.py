import torch
import torch.nn as nn
from pygen.distributions.unitinterval_bucket import UnitIntervalBucket
import pixelcnn_pp.model as pixelcnn_model
import pygen.layers.independent_bernoulli as bernoulli_layer
import pygen.layers.independent_unitinterval_bucket as unitinterval_bucket_layer


class _PixelCNN(nn.Module):
    def __init__(self, event_shape, layer, nr_resnet=3):
        super().__init__()
        self.event_shape = event_shape
        self.layer = layer
        self.batch_shape = []
        self.pixelcnn_net = pixelcnn_model.PixelCNN(nr_resnet=nr_resnet, nr_filters=160,
            input_channels=self.event_shape[0], nr_params=layer.params_size(), nr_conditional=None)

    def log_prob(self, samples):
        if samples.size()[1:4] != torch.Size(self.event_shape):
            raise RuntimeError("sample shape {}, but event_shape has shape {}"
                .format(samples.shape[1:4], self.event_shape))
        logits = self.pixelcnn_net((samples*2.0)-1.0, conditional=None)
        layer_logits = logits.permute(0, 2, 3, 1)  # B, Y, X, P where P for parameters
        permute_samples = samples.permute(0, 2, 3, 1)  # B, Y, X, C
        return self.layer(layer_logits).log_prob(permute_samples).sum(axis=[1, 2])

    def sample(self, sample_shape=None):
        if sample_shape is None:
            batch_shape = [1]
        else:
            batch_shape = sample_shape
        with torch.no_grad():
            sample = torch.zeros(batch_shape+self.event_shape, device=next(self.parameters()).device)
            for y in range(self.event_shape[1]):
                for x in range(self.event_shape[2]):
                    logits = self.pixelcnn_net((sample*2)-1, sample=True, conditional=None)[:, :, y, x]
                    pixel_sample = self.layer(logits).sample()
                    sample[:, :, y, x] = pixel_sample
        if sample_shape is None:
            return sample[0]
        else:
            return sample


class PixelCNNBernoulliDistribution(_PixelCNN):
    def __init__(self, event_shape, nr_resnet=3):
        super().__init__(event_shape, bernoulli_layer.IndependentBernoulli(event_shape=event_shape[:1]), nr_resnet=nr_resnet)


class PixelCNNUnitIntervalBucketDistribution(_PixelCNN):
    def __init__(self, event_shape, nr_resnet=3):
        super().__init__(event_shape, unitinterval_bucket_layer.IndependentUnitIntervalBucket(event_shape=event_shape[:1]), nr_resnet=nr_resnet)
