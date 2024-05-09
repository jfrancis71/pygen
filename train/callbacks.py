import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

def image_grid(images, labels):
    """Return a 5x5 grid of images with labels."""
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
                raise("Unknown image with channels", num_channels)
        plt.imshow(image, cmap=cmap)
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    width, height = canvas.get_width_height()
    data = np.array(data).reshape(height, width, 4)
    return data[:, :, :3].transpose(2, 0, 1)


class TBClassifyImagesCallback():
    def __init__(self, tb_writer, tb_name, dataset, class_labels):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.dataset = dataset
        self.class_labels = class_labels

    def __call__(self, trainer):
        images = torch.stack([self.dataset[i][0].to(trainer.device) for i in range(25)])
        labels = [self.class_labels[idx.to("cpu").item()] for idx in trainer.trainable(images).sample()]
        labelled_images = image_grid(images.to("cpu"), labels)
        self.tb_writer.add_image(self.tb_name, labelled_images, trainer.epoch)


class TBImagesCallback():
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        batch_size = 16
        imglist = [trainer.trainable.sample([batch_size]) for _ in range(16 // batch_size)]
        imglist = torch.clip(torch.cat(imglist, axis=0), 0.0, 1.0)
        grid_image = torchvision.utils.make_grid(imglist, padding=10, nrow=4)
        self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBConditionalImagesCallback():
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        batch_size = 2
        imglist = [trainer.trainable(torch.tensor(label_idx, device=trainer.device)).sample([batch_size]) for label_idx in range(10)]
        imglist = torch.clip(torch.cat(imglist, axis=0), 0.0, 1.0)
        grid_image = torchvision.utils.make_grid(imglist, padding=10, nrow=2)
        self.tb_writer.add_image(self.tb_name, grid_image, trainer.epoch)


class TBBatchLogProbCallback():
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        self.tb_writer.add_scalar(self.tb_name, trainer.log_prob_item, trainer.batch_num)


class TBTotalLogProbCallback():
    def __init__(self, tb_writer, tb_name):
        self.tb_writer = tb_writer
        self.tb_name = tb_name

    def __call__(self, trainer):
        self.tb_writer.add_scalar(self.tb_name, trainer.total_log_prob/trainer.batch_len, trainer.epoch)


class _TBDatasetLogProbCallback():
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.batch_size = batch_size
        self.dataset = dataset

    def __call__(self, trainer):
        dataloader = torch.utils.data.DataLoader(self.dataset, collate_fn=None, batch_size=self.batch_size, shuffle=True,
                                             drop_last=True)
        log_prob = 0.0
        size = 0
        for (_, batch) in enumerate(dataloader):
            log_prob += self.batch_log_prob(trainer, batch)
            size += 1
        self.tb_writer.add_scalar(self.tb_name, log_prob/size, trainer.epoch)


class TBDatasetLogProbDistributionCallback(_TBDatasetLogProbCallback):
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        super().__init__(tb_writer, tb_name, dataset, batch_size)

    def batch_log_prob(self, trainer, batch):
        return (trainer.trainable.log_prob(batch[0].to(trainer.device)).mean()).item()


class TBDatasetLogProbLayerCallback(_TBDatasetLogProbCallback):
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32, reverse_inputs=False):
        super().__init__(tb_writer, tb_name, dataset, batch_size)
        self.reverse_inputs = reverse_inputs

    def batch_log_prob(self, trainer, batch):
        if not self.reverse_inputs:
            return (trainer.trainable(batch[0].to(trainer.device)).log_prob(batch[1].to(trainer.device)).mean()).item()
        else:
            return (trainer.trainable(batch[1].to(trainer.device)).log_prob(batch[0].to(trainer.device)).mean()).item()


class TBAccuracyCallback():
    def __init__(self, tb_writer, tb_name, dataset, batch_size=32):
        self.tb_writer = tb_writer
        self.tb_name = tb_name
        self.batch_size = batch_size
        self.dataset = dataset

    def __call__(self, trainer):
        dataloader = torch.utils.data.DataLoader(self.dataset, collate_fn=None, batch_size=self.batch_size, shuffle=True,
                                             drop_last=True)
        correct = 0.0
        size = 0
        for (_, batch) in enumerate(dataloader):
            correct += (trainer.trainable(batch[0].to(trainer.device)).sample().cpu()==batch[1]).sum().item()
            size += self.batch_size
        self.tb_writer.add_scalar(self.tb_name, correct/size, trainer.epoch)


def callback_compose(list_callbacks):
    def call_callbacks(trainer):
        for fn in list_callbacks:
            fn(trainer)
    return call_callbacks
