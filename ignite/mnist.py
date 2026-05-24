# This example is from https://github.com/pytorch/ignite/tree/master
# See above for license.

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from train import callbacks
import layers.independent_categorical as layer_categorical
from neural_nets import classifier_net


parser = ArgumentParser()
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--max_epoch", type=int, default=10, help="number of epochs to train (default: 10)")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
dataset = datasets.MNIST(args.datasets_folder, train=True, download=True, transform=data_transform)
data_split = [55000, 5000]
train_dataset, validation_dataset = random_split(dataset, data_split,
    generator=torch.Generator(device=torch.get_default_device()))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
model = nn.Sequential(classifier_net.ClassifierNet(mnist=True), nn.LogSoftmax(dim=-1))
model.to(args.device)  # Move model before creating optimizer
optimizer = SGD(model.parameters(), lr=.001)
criterion = nn.NLLLoss()
trainer = create_supervised_trainer(model, optimizer, criterion, device=args.device)
trainer.logger = setup_logger("trainer")

val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=args.device)
evaluator.logger = setup_logger("evaluator")

example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset, batch_size=25)))[0]

tb_writer = SummaryWriter(args.tb_folder)

classifier = torch.nn.Sequential(
    model,
    layer_categorical.IndependentCategorical(event_shape=[], num_classes=10)
)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_nll = metrics["nll"]
    tb_writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
    tb_writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_nll = metrics["nll"]
    tb_writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)
    tb_writer.add_scalar("valiation/avg_loss", avg_nll, engine.state.epoch)
    image = callbacks.demo_classify_images(classifier, example_valid_images, dataset.classes)()
    if tb_writer is not None:
        tb_writer.add_image("valid_images", image, engine.state.epoch)

trainer.run(train_loader, max_epochs=args.max_epoch)
