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
from torchvision.utils import make_grid
from torchvision.utils import save_image


def log_image_cb(make_image_fn, tb_writer=None, folder=None, name=None):
    def _fn(trainer_state):
        image = make_image_fn()
        if tb_writer is not None:
            tb_writer.add_image(name, image, trainer_state.epoch_num)
        if folder is not None:
            save_image(image, folder + "/" + name + "_" + str(trainer_state.epoch_num) + ".png")
    return _fn


def make_labelled_images_grid(images, labels):
    """Make a grid of labelled images.

    Args:
        images: a list or tensor of 25 images
        labels: a list of 25 strings.

    Returns:
        image tensor of labelled images organised into 5x5 grid.

    Examples:
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
    return torch.tensor(data[:, :, :3].transpose(2, 0, 1)).float()/255.


def demo_classify_images(classifier, images, categories):
    """Classify images using the trainable and tensorboard log the result organised in a 5x5 grid.

    Args:
        classifier: nn.Module accepting a tensor and returning a categorical distribution.
        images: a tensor of batch size 25.
        categories: a list of the dataset categories, eg for CIFAR-10 ["aeroplane", "car", ...]

    Example:
        >>> import pygen.layers.independent_categorical as independent_categorical
        >>> images = torch.ones([25, 1, 5, 5])
        >>> dataset_class_labels = [str(category) for category in range(10)]
        >>> trainable = nn.Sequential(nn.Flatten(), nn.Linear(1*5*5, 10), independent_categorical.IndependentCategorical(event_shape=[], num_classes=10))
        >>> image_demo_fn = demo_classify_images(trainable, images, dataset_class_labels)
        >>> len(image_demo_fn())
        3
    """
    def _fn():
        label_indices = classifier(images).sample()
        labels = [categories[idx.to("cpu").item()] for idx in label_indices]
        labelled_images = make_labelled_images_grid(images, labels)
        return labelled_images
    return _fn


def demo_conditional_images(trainable, conditionals, num_samples=2):
    """Samples a layer type trainable using the input conditionals.

       Samples each conditional num_samples times and returns an image with the results assembled in a grid.
       Suitable for, e.g., generating MNIST images conditioned on knowing the digit class.

    Args:
        trainable: a layer type object accepting an input and returning a probability distribution over images.
        num_samples: number of samples for each conditional
        conditionals: tensor describing the inputs to the trainable.

    Examples:
        >>> import pygen.layers.independent_bernoulli as layers_bernoulli
        >>> trainable = nn.Sequential(nn.Linear(10, 1*8*8), layers_bernoulli.IndependentBernoulli(event_shape=[1, 8, 8]))
        >>> image_demo_fn = demo_conditional_images(trainable, torch.eye(10), 2)
        >>> len(image_demo_fn())
        3
    """
    def _fn():
        images = trainable(conditionals).sample([num_samples])
        imglist = images.permute([1, 0, 2, 3, 4]).flatten(end_dim=1)  # Transpose the sample and batch dims
        grid_image = make_grid(imglist, padding=10, nrow=num_samples, value_range=(0.0, 1.0))
        return grid_image
    return _fn


def tb_log_metrics(tb_writer, tb_name, metrics, step):
    for key in metrics.dtype.fields.keys():
        if tb_writer is not None:
            tb_writer.add_scalar(tb_name + "_" + key, metrics[key], step)


def reduce_metrics_history(metrics_history):
    keys = metrics_history[-1].dtype.fields.keys()
    stacked_history = np_recfunctions.stack_arrays(metrics_history)
    metric_means = [stacked_history[key].mean() for key in keys]
    metrics_epoch = np.array(tuple(metric_means), dtype=metrics_history[-1].dtype)
    return metrics_epoch


def tb_batch_log_metrics(tb_writer):
    """Tensorboard log the trainer_state's batch_metrics with the batch number."""
    return lambda trainer_state: tb_log_metrics(tb_writer, "batch", trainer_state.batch_metrics,
        trainer_state.batch_num)


def tb_epoch_log_metrics(tb_writer):
    """Tensorboard log the mean of each of the metrics in trainer_state's metrics_history with the epoch number.
    """
    def _fn(trainer_state):
        metrics_epoch = reduce_metrics_history(trainer_state.metrics_history)
        tb_log_metrics(tb_writer, "train_epoch", metrics_epoch, trainer_state.epoch_num)
    return _fn


def tb_dataset_metrics_logging(tb_writer, tb_name, dataset, batch_size=32):
    """Runs the trainable over a dataset (eg validation) and logs the resulting metrics.

    Examples:
        >>> import pygen.train.train as train
        >>> import pygen.layers.independent_categorical as independent_categorical
        >>> trainable = nn.Sequential(nn.Flatten(), nn.Linear(1*5*5, 10), independent_categorical.IndependentCategorical(event_shape=[], num_classes=10))
        >>> trainer_state = type('TrainerState', (object,), {'trainable': trainable})()
        >>> trainer_state.batch_objective_fn = train.layer_objective()
        >>> trainer_state.epoch_num = 0
        >>> images, labels = (torch.rand([64, 1, 5, 5]), torch.randint(0, 10, [64]))
        >>> dataset = torch.utils.data.StackDataset(images, labels)
        >>> tb_dataset_metrics_logging(None, "", dataset)(trainer_state)
    """
    def _fn(trainer_state):
        dataloader = DataLoader(dataset, collate_fn=None,
            generator=torch.Generator(device=torch.get_default_device()),
            batch_size=batch_size, shuffle=True, drop_last=True)
        dataset_iter = iter(dataloader)
        metrics_history = []
        for batch in dataset_iter:
            with torch.no_grad():
                log_prob, batch_metrics = trainer_state.batch_objective_fn(trainer_state.trainable, batch)
            metrics_history.append(batch_metrics)
        metrics_epoch = reduce_metrics_history(metrics_history)
        tb_log_metrics(tb_writer, tb_name + "_epoch", metrics_epoch, trainer_state.epoch_num)
    return _fn


def tb_log_gradients(tb_writer):
    def _fn(trainer_state):
        for name, layer in trainer_state.trainable.named_parameters():
            tb_writer.add_histogram(name + "_grad", layer.grad.cpu(), trainer_state.epoch_num)
    return _fn


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
