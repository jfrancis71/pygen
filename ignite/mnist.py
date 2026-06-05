# This example is from https://github.com/pytorch/ignite/tree/master
# See above for license.

import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from pygen.train import callbacks
from pygen.neural_nets import classifier_net
import pygen.layers.independent_categorical as layer_categorical


parser = argparse.ArgumentParser()
parser.add_argument("--datasets_folder", default="~/datasets")
parser.add_argument("--tb_folder", default=None)
parser.add_argument("--images_folder", default=None)
parser.add_argument("--device", default="cpu")
parser.add_argument("--max_epoch", type=int, default=10, help="number of epochs to train (default: 10)")
args = parser.parse_args()

dataset = MNIST(args.datasets_folder, train=True, download=True,
    transform=ToTensor())
data_split = [55000, 5000]
train_dataset, validation_dataset = random_split(dataset, data_split)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
model = classifier_net.ClassifierNet(mnist=True)
model.to(args.device)  # Move model before creating optimizer
optimizer = Adam(model.parameters(), lr=.001)
criterion = torch.nn.CrossEntropyLoss()
trainer = create_supervised_trainer(model, optimizer, criterion, device=args.device)
trainer.logger = setup_logger("trainer")
val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=args.device)
evaluator.logger = setup_logger("evaluator")
example_valid_images = next(iter(torch.utils.data.DataLoader(validation_dataset,
    batch_size=25)))[0].to(args.device)
tb_writer = SummaryWriter(args.tb_folder)

@trainer.on(Events.EPOCH_COMPLETED)
def log_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    tb_writer.add_scalar("training/avg_accuracy", metrics["accuracy"],
        engine.state.epoch)
    tb_writer.add_scalar("training/avg_loss", metrics["nll"], engine.state.epoch)
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    tb_writer.add_scalar("validation/avg_accuracy", metrics["accuracy"],
        engine.state.epoch)
    tb_writer.add_scalar("validation/avg_loss", metrics["nll"], engine.state.epoch)
    trainer_state = Warning() # bit of a hack here.
    trainer_state.epoch_num = engine.state.epoch
    callbacks.log_image_cb(
        callbacks.demo_classify_images(
            model, example_valid_images, dataset.classes),
        tb_writer=tb_writer, folder=args.images_folder, name="valid_images")(
            trainer_state)

trainer.run(train_loader, max_epochs=args.max_epoch)
