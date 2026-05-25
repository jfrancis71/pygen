# This example is from https://github.com/pytorch/ignite/tree/master
# See above for license.

import argparse
import torch
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
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.independent_categorical as layer_categorical


parser = argparse.ArgumentParser()
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", type=int, default=10, help="number of epochs to train (default: 10)")
args = parser.parse_args()

transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
dataset = datasets.MNIST(args.datasets_folder, train=True, download=True, transform=transform)
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
def log_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    tb_writer.add_scalar("training/avg_accuracy", metrics["accuracy"], engine.state.epoch)
    tb_writer.add_scalar("training/avg_loss", metrics["nll"], engine.state.epoch)
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    tb_writer.add_scalar("validation/avg_accuracy", metrics["accuracy"], engine.state.epoch)
    tb_writer.add_scalar("validation/avg_loss", metrics["nll"], engine.state.epoch)
    image = callbacks.demo_classify_images(classifier, example_valid_images, dataset.classes)()
    if tb_writer is not None:
        tb_writer.add_image("valid_images", image, engine.state.epoch)

trainer.run(train_loader, max_epochs=args.max_epoch)
