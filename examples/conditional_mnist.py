"""Simple example program, generates MNIST images conditioned on a digit, uses pygen."""


import argparse
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pygen.train import train
from pygen.train import callbacks
import pygen.layers.independent_bernoulli as bernoulli_layer


parser = argparse.ArgumentParser(description='PyGen Conditional MNIST PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float(), train.DevicePlacement()])
dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
train_dataset, validation_dataset = random_split(dataset, [50000, 10000])
torch.set_default_device(ns.device)
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBConditionalImages(tb_writer, "conditional_generated_images", num_labels=10),
    callbacks.TBEpochLogMetrics(tb_writer),
    callbacks.TBDatasetMetricsLogging(tb_writer, "validation", validation_dataset)
    ])
conditional_digit_distribution = nn.Sequential(nn.Linear(10, 1*28*28),
    bernoulli_layer.IndependentBernoulli(event_shape=[1,28,28]))
train.train(conditional_digit_distribution, train_dataset, train.OneHotLayerTrainer(10),
    batch_end_callback=callbacks.TBBatchLogMetrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
