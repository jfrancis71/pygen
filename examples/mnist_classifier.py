"""Simple example program for MNIST classification using pygen."""


import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision
from pygen.train import train
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.categorical as layer_categorical


parser = argparse.ArgumentParser(description='PyGen MNIST Classifier')
parser.add_argument("--datasets_folder", default=".")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.MNIST(ns.datasets_folder, train=True, download=True,
    transform=transform)
train_dataset, validation_dataset = random_split(dataset, [50000, 10000])
class_labels = [f"{num}" for num in range(10)]
tb_writer = SummaryWriter(ns.tb_folder)
epoch_end_callbacks = callbacks.callback_compose([
    callbacks.TBClassifyImagesCallback(tb_writer, "train_images", train_dataset, class_labels),
    callbacks.TBClassifyImagesCallback(tb_writer, "validation_images",
        validation_dataset, class_labels),
    callbacks.TBTotalLogProbCallback(tb_writer, "train_epoch_log_prob"),
    callbacks.TBDatasetLogProbLayerCallback(tb_writer, "validation_log_prob", validation_dataset),
    callbacks.TBAccuracyCallback(tb_writer, "train_accuracy", train_dataset),
    callbacks.TBAccuracyCallback(tb_writer, "validation_accuracy", validation_dataset)])
digit_recognizer = classifier_net.ClassifierNet(mnist=True)
train.LayerTrainer(digit_recognizer.to(ns.device), train_dataset,
    batch_end_callback=callbacks.TBBatchLogProbCallback(tb_writer, "batch_log_prob"),
    epoch_end_callback=epoch_end_callbacks, dummy_run=ns.dummy_run).train()
