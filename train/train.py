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


class OneHotLayerTrainer:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, trainable, batch):
        conditional = torch.nn.functional.one_hot(batch[1], self.num_classes).float()
        distribution = trainable(conditional)
        log_prob_mean = (distribution.log_prob(batch[0])).mean()
        return log_prob_mean, np.array((log_prob_mean.cpu().detach().numpy()), dtype=[('log_prob', 'float32')])


def classifier_trainer(trainable, batch):
    conditional = batch[0]
    value = batch[1]
    distribution = trainable(conditional)
    log_prob_mean = (distribution.log_prob(value)).mean()
    accuracy = (distribution.sample() == value).float().mean()
    return log_prob_mean, np.array(
        (log_prob_mean.cpu().detach().numpy(), accuracy.cpu().detach().numpy()),
        dtype=[('log_prob', 'float32'), ('accuracy', 'float32')])


class TrainingLoopInfo:
    def __init__(self):
        self.trainable = None
        self.epoch_num = None
        self.batch_num = None
        self.batch_metrics = None
        self.metrics_history = None
        self.batch_objective_fn = None


def train(trainable, dataset, batch_objective_fn, batch_size=32, max_epoch=10, batch_end_callback=None,
          epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None, epoch_regularizer=False):
    training_loop_info = TrainingLoopInfo()
    if dummy_run:
        dataset = torch.utils.data.Subset(dataset, range(self.batch_size))
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=True, generator=torch.Generator(device=torch.get_default_device()),
        drop_last=True)
    opt = torch.optim.Adam(trainable.parameters(), maximize=True, lr=.001)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=rm_scheduler)
    else:
        scheduler = None
    training_loop_info.batch_num = 0
    training_loop_info.trainable = trainable
    training_loop_info.batch_objective_fn = batch_objective_fn
    for training_loop_info.epoch_num in range(max_epoch):
        print("Epoch: ", training_loop_info.epoch_num)
        training_loop_info.metrics_history = []
        for (_, batch) in enumerate(dataloader):
            trainable.zero_grad()
            objective, training_loop_info.batch_metrics = batch_objective_fn(trainable, batch)
            training_loop_info.metrics_history.append(training_loop_info.batch_metrics)
            if epoch_regularizer is True:
                objective += trainable.epoch_regularizer_penalty(batch) / (training_loop_info.epoch_num+1)
            objective.backward()
            opt.step()
            training_loop_info.batch_num += 1
            if batch_end_callback is not None:
                batch_end_callback(training_loop_info)
        if epoch_end_callback is not None:
            epoch_end_callback(training_loop_info)
        if model_path is not None:
            torch.save(trainable.state_dict(), model_path)
        if scheduler:
            scheduler.step()
