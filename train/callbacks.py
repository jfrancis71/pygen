"""
callbacks is a module defining functors which can be passed to a Trainer
to monitor Training session.
Callbacks are generally implemented as functors so they can be configured, eg
the tensorboard writer, or the name of tensorboard string to log with.
Callbacks when called are passed the trainer as an argument so they can
inspect any trainer state for logging.
"""


import numpy as np
import numpy.lib.recfunctions as np_recfunctions
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
import pygen.layers.categorical as layers_categorical
import pygen.layers.independent_bernoulli as layers_bernoulli
import pygen.train.train as train


def make_labelled_images_grid(images, labels):
    """make a grid of labelled images.
    images is a list or tensor of 25 images, labels is a list of 25 strings.
    returns an image tensor of labelled images organised into 5x5 grid.

    >>> images, labels = torch.zeros(25, 1, 28, 28), [str(idx) for idx in range(25)]
    >>> len(make_labelled_images_grid(images, labels).shape)
    3
    """
    if images.shape[0] != 25:
        raise RuntimeError(f"images batch shape is {images.shape[0]} expected 25.")
    if len(labels) != 25:
        raise RuntimeError(f"len(labels) is {len(labels)} expected 25.")
    plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=labels[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        num_channels = images[i].shape[0]
        if num_channels == 1:
            image = images[i][0]
            cmap = 'gray'
        else:
            if num_channels == 3:
                image = images[i].permute(1, 2, 0)
                cmap = None
            else:
                raise ValueError(f"Unknow image type with num_channels={num_channels}")
        plt.imshow(image.cpu(), cmap=cmap)
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    width, height = canvas.get_width_height()
    data = np.array(data).reshape(height, width, 4)  # pylint: disable=E1121
    return data[:, :, :3].transpose(2, 0, 1)


def tb_classify_images(tb_writer, tb_name, images, categories):
    """Classify images using the trainable and tensorboard log the result organised in a 5x5 grid.
    images should be a tensor of batch size 25.
    categories should be a list of the dataset categories, eg for CIFAR-10 ["aeroplane", "car", ...]

    >>> images = torch.ones([25, 1, 5, 5])
    >>> dataset_class_labels = [str(category) for category in range(10)]
    >>> trainable = nn.Sequential(nn.Flatten(), nn.Linear(1*5*5, 10), layers_categorical.Categorical())
    >>> callback = tb_classify_images(None, "", images, dataset_class_labels)
    >>> training_loop_info = type('TrainingLoopInfo', (object,), {'trainable': trainable})()
    >>> callback(training_loop_info)
    """
    def cb_tb_classify_images(training_loop_info):
        classifier = training_loop_info.trainable
        label_indices = classifier(images).sample()
        labels = [categories[idx.to("cpu").item()] for idx in label_indices]
        labelled_images = make_labelled_images_grid(images, labels)
        if tb_writer is not None:
            tb_writer.add_image(tb_name, labelled_images, training_loop_info.epoch_num)
    return cb_tb_classify_images


def tb_conditional_images(tb_writer, tb_name, num_labels):
    """Produces a num_labels x 2 grid of images where each row is an image generated conditioned on
       the corresponding class label, and there are two examples per row.
       Suitable for trainables that are Layer objects accepting a one hot vector
       and returning a probability distribution over an image.
       eg a trainable accepting one hot vector with position 2 = 1, returning 1x28x28 probability distributions
       over digit 2.

    >>> trainable = nn.Sequential(nn.Linear(10, 1*8*8), layers_bernoulli.IndependentBernoulli(event_shape=[1, 8, 8]))
    >>> callback = tb_conditional_images(None, "", 10)
    >>> trainer = type('Trainer', (object,), {'trainable': trainable})()
    >>> callback(trainer)
    """
    def cb_tb_conditional_images(training_loop_info):
        sample_size = 2
        identity = torch.eye(num_labels)
        images = training_loop_info.trainable(identity).sample([sample_size])
        imglist = images.permute([1, 0, 2, 3, 4]).flatten(end_dim=1)  # Transpose the sample and batch dims
        grid_image = make_grid(imglist, padding=10, nrow=2, value_range=(0.0, 1.0))
        if tb_writer is not None:
            tb_writer.add_image(tb_name, grid_image, training_loop_info.epoch_num)
    return cb_tb_conditional_images


def tb_log_metrics(tb_writer, tb_name, metrics, step):
    for key in metrics.dtype.fields.keys():
        tb_writer.add_scalar(tb_name + "_" + key, metrics[key], step)


def reduce_metrics_history(metrics_history):
    keys = metrics_history[-1].dtype.fields.keys()
    stacked_history = np_recfunctions.stack_arrays(metrics_history)
    metric_means = [stacked_history[key].mean() for key in keys]
    metrics_epoch = np.array(tuple(metric_means), dtype=metrics_history[-1].dtype)
    return metrics_epoch


def tb_batch_log_metrics(tb_writer):
    return lambda training_loop_info: tb_log_metrics(tb_writer, "batch", training_loop_info.batch_metrics,
        training_loop_info.batch_num)


def tb_epoch_log_metrics(tb_writer):
    def cb_tb_epoch_log_metrics(training_loop_info):
        metrics_epoch = reduce_metrics_history(training_loop_info.metrics_history)
        tb_log_metrics(tb_writer, "train_epoch", metrics_epoch, training_loop_info.epoch_num)
    return cb_tb_epoch_log_metrics


def tb_dataset_metrics_logging(tb_writer, tb_name, dataset, batch_size=32):
    def cb_tb_dataset_metrics_logging(training_loop_info):
        dataloader = DataLoader(dataset, collate_fn=None,
            generator=torch.Generator(device=torch.get_default_device()),
            batch_size=batch_size, shuffle=True, drop_last=True)
        dataset_iter = iter(dataloader)
        metrics_history = []
        for batch in dataset_iter:
            log_prob, batch_metrics = training_loop_info.batch_objective_fn(training_loop_info.trainable, batch)
            metrics_history.append(batch_metrics)
        metrics_epoch = reduce_metrics_history(training_loop_info.metrics_history)
        tb_log_metrics(tb_writer, tb_name + "_epoch", metrics_epoch, training_loop_info.epoch_num)
    return cb_tb_dataset_metrics_logging


def callback_compose(list_callbacks):
    """Strings a list of callbacks into one callback so you can have multiple callbacks
       for eg an epoch end callback.
    """
    def call_callbacks(trainer):
        for func in list_callbacks:
            func(trainer)
    return call_callbacks


import doctest
doctest.testmod()
