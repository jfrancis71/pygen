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


parser = argparse.ArgumentParser(description='PyGen Conditional MNIST PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
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
    callbacks.TBConditionalImages(tb_writer, "conditional_generated_images", num_labels=10),
    callbacks.TBEpochLogProb(tb_writer, "train_epoch_log_prob"),
    callbacks.TBDatasetLogProb(tb_writer, "validation_log_prob",
        validation_dataset)
    ])
conditional_digit_distribution = nn.Sequential(nn.Linear(10, 1*28*28),
    bernoulli_layer.IndependentBernoulli(event_shape=[1,28,28]))
train.LayerTrainer(
    conditional_digit_distribution.to(ns.device),
    train_dataset,
    batch_end_callback=callbacks.TBBatchLogProb(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callback, reverse_inputs=True, dummy_run=ns.dummy_run, num_classes=10).train()
