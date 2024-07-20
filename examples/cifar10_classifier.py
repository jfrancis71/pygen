"""Simple example program for CIFAR10 classification using pygen."""


import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pygen.train import train
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.categorical as layer_categorical


parser = argparse.ArgumentParser(description='PyGen CIFAR10 Classifier')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--use_scheduler", action="store_true")
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(), train.DevicePlacement()])
dataset = datasets.CIFAR10(ns.datasets_folder, train=True, download=False, transform=transform)
train_dataset, validation_dataset = random_split(dataset, [45000, 5000])
torch.set_default_device(ns.device)
# Grab some example images to use in the tensorboard example classifications callback.
example_train_images = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=25)))[0]
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=25)))[0]
class_labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBClassifyImages(tb_writer, "train_images", example_train_images, class_labels),
    callbacks.TBClassifyImages(tb_writer, "validation_images", example_valid_images, class_labels),
    callbacks.TBEpochLogMetrics(tb_writer),
    callbacks.TBDatasetMetricsLogging(tb_writer, "validation", validation_dataset),
    ])
cifar10_recognizer = torch.nn.Sequential(classifier_net.ClassifierNet(mnist=False), layer_categorical.Categorical())
train.train(cifar10_recognizer, train_dataset, train.classifier_trainer,
    batch_end_callback=callbacks.TBBatchLogMetrics(tb_writer),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run)
