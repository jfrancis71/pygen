"""
train module provides two classes: DistributionTrainer and LayerTrainer
for training distributions and layers respectively.
Layers are inspired by the TensorFlow probability distributions package.
A Layer object accepts a tensor and returns a probability distribution.
Example: An MNIST classifier which takes an image as a tensor and returns
a Softmax distribution over digits can be implemented as a layer object.
"""


import math
import numpy as np
import torch
import torch.optim.lr_scheduler
import torch.nn as nn


# Need as ToTensor in transforms ignores the global default device option.
class DevicePlacement:
    def __call__(self, x):
        return x.to(torch.get_default_device())


# Robbins Monro, Ref 1:
def rm_scheduler(epoch):
    """Robins Monro compliant scheduler.
       See "What is an adaptive step size in parameter estimation", YouTube,
       Ian explains signals, systems and digital comms, June 20, 2022
    """
    return 1 / math.sqrt(1 + epoch)


def layer_objective(reverse_inputs=False, track_accuracy=False):
    """Computes trainable(batch[0]).log_prob(batch[1]) (reversed order if reverse_inputs is True)
    returns this as the 1st element of a tuple, second element is metrics containing the same.
    trainable is a layer object taking a tensor and returning a distribution.
    """
    def _fn(trainable, batch):
        if reverse_inputs is False:
            conditional, value = batch[0], batch[1]
        else:
            conditional, value = batch[1], batch[0]
        distribution = trainable(conditional)
        log_prob_mean = (distribution.log_prob(value)).mean()
        if track_accuracy:
            accuracy = (distribution.sample() == value).float().mean()
            return log_prob_mean, np.array(
                (log_prob_mean.cpu().detach().numpy(), accuracy.cpu().detach().numpy()),
                dtype=[('log_prob', 'float32'), ('accuracy', 'float32')])
        else:
            return log_prob_mean, np.array((log_prob_mean.cpu().detach().numpy()), dtype=[('log_prob', 'float32')])
    return _fn


class TrainerState:
    """Maintains trainer state so that it is accessible from callbacks."""
    def __init__(self):
        self.trainable = None
        self.epoch_num = None
        self.batch_num = None
        self.batch_metrics = None
        self.metrics_history = None
        self.batch_objective_fn = None


def train(trainable, dataset, batch_objective_fn, batch_size=32, max_epoch=10, batch_end_callback=None,
          epoch_end_callback=None, optimizer=None, scheduler=None, dummy_run=False, model_path=None, epoch_regularizer=False):
    """trains a trainable.

    Examples:
        >>> import pygen.layers.independent_categorical as layer_categorical
        >>> trainable = nn.Sequential(nn.Flatten(), nn.Linear(1*5*5, 10), layer_categorical.IndependentCategorical(event_shape=[], num_classes=10))
        >>> images, labels = (torch.rand([64, 1, 5, 5]), torch.randint(0, 10, [64]))
        >>> dataset = torch.utils.data.StackDataset(images, labels)
        >>> train(trainable, dataset, layer_objective(), max_epoch=1)
        Epoch: 0
    """
    trainer_state = TrainerState()
    if dummy_run:
        dataset = torch.utils.data.Subset(dataset, range(batch_size))
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, generator=torch.Generator(device=torch.get_default_device()),
        drop_last=True)
    if optimizer is None:
        opt = torch.optim.Adam(trainable.parameters(), maximize=True, lr=.001)
        if scheduler is not None:
            raise RuntimeError("Cannot pass in a scheduler without an optimizer. \
            The scheduler needs to know what optimizer it is scheduling for.")
    else:
        opt = optimizer
#   scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=rm_scheduler)
    trainer_state.batch_num = 0
    trainer_state.trainable = trainable
    trainer_state.batch_objective_fn = batch_objective_fn
    for trainer_state.epoch_num in range(max_epoch):
        print("Epoch:", trainer_state.epoch_num)
        trainer_state.metrics_history = []
        for (_, batch) in enumerate(dataloader):
            trainable.zero_grad()
            objective, trainer_state.batch_metrics = batch_objective_fn(trainable, batch)
            trainer_state.metrics_history.append(trainer_state.batch_metrics)
            if epoch_regularizer is True:
                objective -= trainable.epoch_regularizer_penalty(batch) / (trainer_state.epoch_num+1)
            objective.backward()
            opt.step()
            trainer_state.batch_num += 1
            if batch_end_callback is not None:
                batch_end_callback(trainer_state)
        if epoch_end_callback is not None:
            epoch_end_callback(trainer_state)
        if model_path is not None:
            torch.save(trainable.state_dict(), model_path)
        if scheduler:
            scheduler.step()


import doctest
doctest.testmod()
