# Preamble
#

import argparse
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from pygen.train import train
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.independent_categorical as layer_categorical


parser = argparse.ArgumentParser(description='PyGen Classifier')
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", default=10, type=int)
args = parser.parse_args()

torch.set_default_device(args.device)
transform = Compose([ToTensor(), train.DevicePlacement()])
dataset = MNIST(args.datasets_folder, train=True, download=True, transform=transform)
data_split = [55000, 5000]
train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
# Grab some example images to use in the tensorboard example classifications callback.
example_valid_images = next(iter(
    torch.utils.data.DataLoader(validation_dataset, batch_size=25)))[0]
tb_writer = SummaryWriter(args.tb_folder)
classifier = torch.nn.Sequential(
    classifier_net.ClassifierNet(mnist=True),
    layer_categorical.IndependentCategorical(event_shape=[], num_classes=10))
epoch_end_callbacks = [
    callbacks.log_image_cb(
        callbacks.demo_classify_images(
            classifier[0], example_valid_images, dataset.classes),
        tb_writer=tb_writer, folder=args.images_folder, name="valid_images"),
    callbacks.tb_epoch_log_metrics(tb_writer),
    callbacks.tb_dataset_metrics_logging(tb_writer, "validation", validation_dataset)
]
train.train(classifier, train_dataset, train.layer_objective(track_accuracy=True),
    batch_end_callback=callbacks.tb_batch_log_metrics(tb_writer),
    epoch_end_callback=callbacks.callback_compose(epoch_end_callbacks),
    max_epoch=args.max_epoch)
