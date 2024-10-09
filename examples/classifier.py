"""Simple classification program (MNIST or CIFAR10) using pygen."""


import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pygen.train import train
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.independent_categorical as layer_categorical


parser = argparse.ArgumentParser(description='PyGen Classifier')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
parser.add_argument("--use_scheduler", action="store_true")
parser.add_argument("--dummy_run", action="store_true")
ns = parser.parse_args()

torch.set_default_device(ns.device)
transform = transforms.Compose([transforms.ToTensor(), train.DevicePlacement()])
if ns.dataset == "mnist":
    dataset = datasets.MNIST(ns.datasets_folder, train=True, download=False, transform=transform)
    mnist = True
    data_split = [55000, 5000]
elif ns.dataset == "cifar10":
    dataset = datasets.CIFAR10(ns.datasets_folder, train=True, download=False, transform=transform)
    mnist = False
    data_split = [45000, 5000]
else:
    raise RuntimeError(f"{ns.dataset} not recognized.")
train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
# Grab some example images to use in the tensorboard example classifications callback.
example_train_images = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=25)))[0]
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=25)))[0]
tb_writer = SummaryWriter(ns.tb_folder)
classifier = torch.nn.Sequential(
    classifier_net.ClassifierNet(mnist=mnist),
    layer_categorical.IndependentCategorical(event_shape=[], num_classes=10))
epoch_end_callbacks = [
    callbacks.tb_log_image(tb_writer, "train_images",
        callbacks.demo_classify_images(classifier, example_train_images, dataset.classes)),
    callbacks.tb_log_image(tb_writer, "valid_images",
                           callbacks.demo_classify_images(classifier, example_valid_images, dataset.classes)),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)]
if ns.images_folder is not None:
    epoch_end_callbacks.append(
        callbacks.file_log_image(ns.images_folder,"train",
            callbacks.demo_classify_images(classifier, example_train_images, dataset.classes)))
    epoch_end_callbacks.append(
        callbacks.file_log_image(ns.images_folder,"valid",
            callbacks.demo_classify_images(classifier, example_valid_images, dataset.classes)))
train.train(classifier, train_dataset, train.layer_objective(track_accuracy=True),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=callbacks.callback_compose(epoch_end_callbacks), dummy_run=ns.dummy_run)
