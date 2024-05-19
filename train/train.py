"""
train module provides two classes: DistributionTrainer and LayerTrainer
for training distributions and layers respectively.
Layers are inspired by the TensorFlow probability distributions package.
A Layer object accepts a tensor and returns a probability distribution.
Example: An MNIST classifier which takes an image as a tensor and returns
a Softmax distribution over digits can be implemented as a layer object.
"""


import math
import torch
import torch.optim.lr_scheduler


# Robbins Monro, Ref 1:
def rm_scheduler(epoch):
    """Robins Monro compliant scheduler.
       See "What is an adaptive step size in parameter estimation", youtube,
       Ian explains signals, systems and digital comms, June 20, 2022
    """
    return 1 / math.sqrt(1 + epoch)


class _Trainer():
    # pylint: disable=R0902
    # pylint: disable=R0913
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10,
            batch_end_callback=None, epoch_end_callback=None, use_scheduler=False,
            dummy_run=False, model_path=None):
        self.trainable = trainable
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.batch_end_callback = batch_end_callback
        self.epoch_end_callback = epoch_end_callback
        self.use_scheduler = use_scheduler
        self.dummy_run = dummy_run
        self.model_path = model_path
        self.device = next(self.trainable.parameters()).device
        self.batch_num = None
        self.epoch = None
        self.total_log_prob = None
        self.batch_len = None
        self.log_prob_item = None

    def train(self):
        """train() starts the training session.
           Note this will set various member attributes while running so they can be
           accessed by callbacks.
        """
        if self.dummy_run:
            dataset = torch.utils.data.Subset(self.dataset, range(self.batch_size))
        else:
            dataset = self.dataset
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=None,
            batch_size=self.batch_size, shuffle=True,
                                             drop_last=True)
        opt = torch.optim.Adam(self.trainable.parameters(), lr=.001)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=rm_scheduler)
        else:
            scheduler = None
        self.batch_num = 0
        for self.epoch in range(self.max_epoch):
            print("Epoch: ", self.epoch)
            self.total_log_prob = 0.0
            self.batch_len = 0
            for (_, batch) in enumerate(dataloader):
                self.trainable.zero_grad()
                log_prob = torch.mean(self.batch_log_prob(batch))  # pylint: disable=E1101
                loss = -log_prob
                loss.backward()
                opt.step()
                self.log_prob_item = log_prob.item()
                self.total_log_prob += self.log_prob_item
                self.batch_num += 1
                self.batch_len += 1
                if self.batch_end_callback is not None:
                    self.batch_end_callback(self)
            if self.epoch_end_callback is not None:
                self.epoch_end_callback(self)
            if self.model_path is not None:
                torch.save(self.trainable.state_dict(), self.model_path)
            if scheduler:
                scheduler.step()

    # pylint: disable=C0116
    def batch_log_prob(self, batch):
        raise NotImplementedError("Unimplemented, Abstract Base Class")


class DistributionTrainer(_Trainer):
    """DistributionTrainer trains a trainable (a learnable probability distribution)
       This distribution should support a log_prob method.
       The train method takes a batch of tuples from the dataset and passes the first element
       of this tuple to the log_prob method.
       Example: Used to train generative models, eg PixelCNN.
    """
    # pylint: disable=R0913
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super().__init__(
            trainable, dataset, batch_size, max_epoch, batch_end_callback,
            epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run,
            model_path=model_path)

    def batch_log_prob(self, batch):
        return self.trainable.log_prob(batch[0].to(self.device))


class LayerTrainer(_Trainer):
    """LayerTrainer trains a trainable layer (ie a probability distribution conditioned on input)
       The trainable should be an object which accepts a tensor input and returns a probability
       distribution.
       The first element of the batch from the dataset is the input to the trainable and the
       second element is the sample from the distribution. (unless reverse_inputs is True in
       which case it is the other way around).
       Example: Could be used to train MNIST classifier, or a PixelCNN conditioned on digit
       identity (with reverse_inputs set to True).
    """
    # pylint: disable=R0913
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False,
                 reverse_inputs=False, model_path=None):
        super().__init__(trainable, dataset, batch_size, max_epoch,
            batch_end_callback, epoch_end_callback, use_scheduler=use_scheduler,
            dummy_run=dummy_run, model_path=model_path)
        self.reverse_inputs = reverse_inputs

    def batch_log_prob(self, batch):
        if not self.reverse_inputs:
            return self.trainable(batch[0].to(self.device)).log_prob(batch[1].to(self.device))
        return self.trainable(batch[1].to(self.device)).log_prob(batch[0].to(self.device))
