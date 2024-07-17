"""
callbacks is a module defining functors which can be passed to a Trainer
to monitor Training session.
Callbacks are generally implemented as functors so they can be configured, eg
the tensorboard writer, or the name of tensorboard string to log with.
Callbacks when called are passed the trainer as an argument so they can
inspect any trainer state for logging.
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
import pygen.layers.categorical as layers_categorical
import pygen.layers.independent_bernoulli as layers_bernoulli


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
    plt.figure(figsize=(10,10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=labels[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = None
        num_channels = images[i].shape[0]
        if num_channels == 1:
            image = images[i][0]
            cmap = 'gray'
        else:
            if num_channels == 3:
                image = images[i].permute(1,2,0)
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


class TBClassifyImages():
    """Classify images using the trainable and tensorboard log the result organised in a 5x5 grid.
    images should be a tensor of batch size 25.
    categories should be a list of the dataset categories, eg for CIFAR-10 ["aeroplane", "car", ...]

    >>> images = torch.ones([25, 1, 5, 5])
    >>> dataset_class_labels = [str(category) for category in range(10)]
    >>> trainable = nn.Sequential(nn.Flatten(), nn.Linear(1*5*5, 10), layers_categorical.Categorical())
    >>> callback = TBClassifyImages(None, "", images, dataset_class_labels)
    >>> trainer = type('Trainer', (object,), {'trainable': trainable})()
    >>> callback(trainer)
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name, images, categories):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.images = images
        self.categories = categories

    def __call__(self, trainer):
        classifier = trainer.trainable
        device = next(trainer.trainable.parameters()).device
        images = self.images.to(device)
        label_indices = classifier(images).sample()
        labels = [self.categories[idx.to("cpu").item()] for idx in label_indices]
        labelled_images = make_labelled_images_grid(images, labels)
        if self.tb_writer is not None:
            self.tb_writer.add_image(self.tb_name, labelled_images, trainer.epoch)


class TBSampleImages():
    """Creates a 4x4 grid of images by sampling the trainable.

    >>> callback = TBSampleImages(None, "")
    >>> base_distribution = torch.distributions.bernoulli.Bernoulli(logits=torch.zeros([1, 8, 8]))
    >>> distribution = torch.distributions.independent.Independent(base_distribution, reinterpreted_batch_ndims=3)
    >>> trainer = type('Trainer', (object,), {'trainable': distribution})()
    >>> callback(trainer)
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        imglist = trainer.trainable.sample([16])
        grid_image = make_grid(imglist, padding=10, nrow=4, value_range=(0.0, 1.0))
        if self.tb_writer is not None:
            self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBConditionalImages():
    """Produces a num_labels x 2 grid of images where each row is an image generated conditioned on
       the corresponding class label, and there are two examples per row.
       Suitable for trainables that are Layer objects accepting a one hot vector
       and returning a probability distribution over an image.
       eg a trainable accepting one hot vector with position 2 = 1, returning 1x28x28 probability distributions
       over digit 2.

    >>> trainable = nn.Sequential(nn.Linear(10, 1*8*8), layers_bernoulli.IndependentBernoulli(event_shape=[1, 8, 8]))
    >>> callback = TBConditionalImages(None, "", 10)
    >>> trainer = type('Trainer', (object,), {'trainable': trainable})()
    >>> callback(trainer)
    """
    def __init__(self, tb_writer, tb_name, num_labels):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.num_labels = num_labels

    def __call__(self, trainer):
        sample_size = 2
        device = next(trainer.trainable.parameters()).device
        identity = torch.eye(self.num_labels, device=device)
        images = trainer.trainable(identity).sample([sample_size])
        imglist = images.permute([1, 0, 2, 3, 4]).flatten(end_dim=1)  # Transpose the sample and batch dims
        grid_image = make_grid(imglist, padding=10, nrow=2, value_range=(0.0, 1.0))
        if self.tb_writer is not None:
            self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBBatchLogProb():
    """Logs the batch log_prob.
       As it applies to the trainer, not the trainable, it is applicable to either
       Layer or Distribution trainables.
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        self.tb_writer.add_scalar(self.tb_name, trainer.log_prob_item, trainer.batch_num)


class TBEpochLogProb():
    """Logs the total log_prob for the epoch.
       As it applies to the trainer, not the trainable, it is applicable to either
       Layer or Distribution trainables.
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        self.tb_writer.add_scalar(self.tb_name,
            trainer.total_log_prob/trainer.batch_len, trainer.epoch)


class TBDatasetLogProb():
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.batch_size = batch_size
        self.dataset = dataset

    def __call__(self, trainer):
        dataloader = DataLoader(self.dataset, collate_fn=None,
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        log_prob = 0.0
        size = 0
        for (_, batch) in enumerate(dataloader):
            log_prob += trainer.batch_log_prob(batch).mean().item()
            size += 1
        self.tb_writer.add_scalar(self.tb_name, log_prob/size, trainer.epoch)


class TBAccuracy():
    """This is for classification trainables, ie Layer trainables which return
       a Categorical distribution, and returns percentage accuracy over
       a dataset, presumably validation_dataset.
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.batch_size = batch_size
        self.dataset = dataset

    def __call__(self, trainer):
        dataloader = DataLoader(self.dataset, collate_fn=None,
            batch_size=self.batch_size, shuffle=True, drop_last=True)
        correct = 0.0
        size = 0
        for (_, batch) in enumerate(dataloader):
            correct += (trainer.trainable(batch[0].to(trainer.device)).sample().cpu() \
                ==batch[1]).sum().item()
            size += self.batch_size
        self.tb_writer.add_scalar(self.tb_name, correct/size, trainer.epoch)


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