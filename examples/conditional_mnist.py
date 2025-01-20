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
import pygen.layers.one_hot as one_hot


parser = argparse.ArgumentParser(description='PyGen Conditional MNIST PixelCNN')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

torch.set_default_device(ns.device)
transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float(), train.DevicePlacement()])
dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
train_dataset, validation_dataset = random_split(dataset, [50000, 10000],
    generator=torch.Generator(device=torch.get_default_device()))
tb_writer = SummaryWriter(ns.tb_folder)
conditional_digit_distribution = nn.Sequential(
    one_hot.OneHot(10),
    nn.Linear(10, 1*28*28),
    bernoulli_layer.IndependentBernoulli(event_shape=[1, 28, 28]))
epoch_end_callbacks = [
    callbacks.log_image_cb(callbacks.demo_conditional_images(conditional_digit_distribution, torch.arange(10), 2),
                           tb_writer=tb_writer, folder=ns.images_folder, name="train_images"),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)]
train.train(conditional_digit_distribution, train_dataset, train.layer_objective(reverse_inputs=True),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=callbacks.callback_compose(epoch_end_callbacks), dummy_run=ns.dummy_run)
