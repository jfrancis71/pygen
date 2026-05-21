# This example is from https://github.com/pytorch/ignite/tree/master
# See above for license.

from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


parser = ArgumentParser()
parser.add_argument("--max_epoch", type=int, default=10, help="number of epochs to train (default: 10)")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(
        MNIST(download=True, root=".", transform=data_transform, train=True), batch_size=32, shuffle=True
    )
val_loader = DataLoader(
        MNIST(download=False, root=".", transform=data_transform, train=False), batch_size=32, shuffle=False
    )
model = Net()
model.to(args.device)  # Move model before creating optimizer
optimizer = SGD(model.parameters(), lr=.001)
criterion = nn.NLLLoss()
trainer = create_supervised_trainer(model, optimizer, criterion, device=args.device)
trainer.logger = setup_logger("trainer")

val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=args.device)
evaluator.logger = setup_logger("evaluator")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_nll = metrics["nll"]
    print(
        f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
    )

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_nll = metrics["nll"]
    print(
        f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
    )

@trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
def log_time(engine):
    print(f"{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds")

trainer.run(train_loader, max_epochs=args.max_epoch)
