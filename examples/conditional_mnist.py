import argparse
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torchvision
import pygen.train.train as train
import pygen.train.callbacks as callbacks
import pygen.layers.independent_bernoulli as bernoulli_layer


class ConditionalDigitDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros([10, 1, 28, 28], requires_grad=True))
        self.layer = bernoulli_layer.IndependentBernoulli(event_shape=[1,28,28])

    def forward(self, x):
        return self.layer(self.logits[x])


parser = argparse.ArgumentParser(description='PyGen MNIST PixelCNN')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: (x > 0.5).float()])
dataset = torchvision.datasets.MNIST(ns.datasets_folder, train=True, download=True, transform=transform)
train_dataset, validation_dataset = random_split(dataset, [50000, 10000])
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callback = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "conditional_generated_images"),
    callbacks.TBTotalLogProbCallback(tb_writer, "train_epoch_log_prob"),
    ])
conditional_digit_distribution = ConditionalDigitDistribution()
train.LayerTrainer(
    conditional_digit_distribution.to(ns.device),
    train_dataset,
    batch_end_callback=callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callback, reverse_inputs=True).train()
