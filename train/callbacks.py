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
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid


def labelled_images_grid(images, labels):
    """images is a list or tensor of 25 images, labels is a list of 25 text objects.
       returns a tensor which represents these images and labels organised into a 5x5 grid.
    """
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
        plt.imshow(image, cmap=cmap)
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    width, height = canvas.get_width_height()
    data = np.array(data).reshape(height, width, 4)  # pylint: disable=E1121
    return data[:, :, :3].transpose(2, 0, 1)


class TBClassifyImagesCallback():
    """You initialise with the dataset eg validation_dataset and the class labels (in text string)
       for the dataset, ie the mappings from the numerical Categorical index to the text string.
       It returns 5x5 image grid with labelling using the trainable object.
       Suitable for trainables that are Layer objects mapping from images to a
       Categorical distribution.
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name, dataset, class_labels):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.dataset = dataset
        self.class_labels = class_labels

    def __call__(self, trainer):
        images = torch.stack([self.dataset[i][0].to(trainer.device) for i in range(25)])  # pylint: disable=E1101
        labels = [self.class_labels[idx.to("cpu").item()]
            for idx in trainer.trainable(images).sample()]
        labelled_images = labelled_images_grid(images.to("cpu"), labels)
        self.tb_writer.add_image(self.tb_name, labelled_images, trainer.epoch)


class TBImagesCallback():
    """Creates a 4x4 grid of images using the trainable.
       Suitable for trainables that are distributions where the probability distribution
       is over images.
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        batch_size = 16
        imglist = [trainer.trainable.sample([batch_size]) for _ in range(16 // batch_size)]
        imglist = torch.clip(torch.cat(imglist, axis=0), 0.0, 1.0)  # pylint: disable=E1101
        grid_image = make_grid(imglist, padding=10, nrow=4)
        self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBConditionalImagesCallback():
    """Produces a 10x2 grid of images where each row is an image generated conditioned on
       the corresponding class label, and there are two examples per row.
       Suitable for trainables that are Layer objects accepting a one got vector
       and returning a probability distribution over an image.
       eg a trainable accepting one hot vector with position 2 = 1, returning 1x28x28 probability distributions
       over digit 2.
    """
    # pylint: disable=R0903
    def __init__(self, tb_writer, tb_name, num_labels):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.num_labels = num_labels

    def __call__(self, trainer):
        sample_size = 2
        imglist = [trainer.trainable(
            torch.nn.functional.one_hot(torch.tensor(label_idx, device=trainer.device), self.num_labels).float()).sample([sample_size]) for label_idx in range(self.num_labels)]
        imglist = torch.clip(torch.cat(imglist, axis=0), 0.0, 1.0)  # pylint: disable=E1101
        grid_image = make_grid(imglist, padding=10, nrow=2)
        self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBBatchLogProbCallback():
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


class TBTotalLogProbCallback():
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


class TBDatasetLogProbCallback():
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


class TBAccuracyCallback():
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
