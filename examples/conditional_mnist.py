"""Simple example program, generates MNIST images conditioned on a digit, uses pygen."""


import argparse
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torchvision
from pygen.train import train
from pygen.train import callbacks
import pygen.layers.independent_bernoulli as bernoulli_layer


class ConditionalDigitDistribution(nn.Module):
    """Layer type which accepts a digit index and returns a probability distribution
       over an image, conditioned on that digit.
    """
    def __init__(self):
        super().__init__()
        # pylint: disable=E1101
        self.linear = nn.Linear(10, 1*28*28)
        self.layer = bernoulli_layer.IndependentBernoulli(event_shape=[1,28,28])

    # pylint: disable=C0103, C0116
    def forward(self, x):
        return self.layer(self.linear(x))


parser = argparse.ArgumentParser(description='PyGen Conditional MNIST PixelCNN')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    lambda x: (x > 0.5).float()])
dataset = torchvision.datasets.MNIST(ns.datasets_folder, train=True, download=True,
    transform=transform)
train_dataset, validation_dataset = random_split(dataset, [50000, 10000])
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callback = callbacks.callback_compose([
    callbacks.TBConditionalImagesCallback(tb_writer, "conditional_generated_images", num_labels=10),
    callbacks.TBTotalLogProbCallback(tb_writer, "train_epoch_log_prob"),
    callbacks.TBDatasetLogProbCallback(tb_writer, "validation_log_prob",
        validation_dataset)
    ])
conditional_digit_distribution = ConditionalDigitDistribution()
train.LayerTrainer(
    conditional_digit_distribution.to(ns.device),
    train_dataset,
    batch_end_callback=callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callback, reverse_inputs=True, dummy_run=ns.dummy_run, num_classes=10).train()
