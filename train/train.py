import torch
import torch.optim.lr_scheduler
import math


# Robbins Monro, Ref 1:
rm_scheduler_fn = lambda epoch: 1 / math.sqrt(1 + epoch)


class _Trainer():
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None, epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        self.trainable = trainable
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.batch_end_callback = batch_end_callback
        self.epoch_end_callback = epoch_end_callback
        self.use_scheduler = use_scheduler
        self.dummy_run = dummy_run
        self.model_path = model_path

    def train(self):
        self.device = next(self.trainable.parameters()).device
        if self.dummy_run:
            dataset = torch.utils.data.Subset(self.dataset, range(self.batch_size))
        else:
           dataset = self.dataset
        dataloader = torch.utils.data.DataLoader(dataset, collate_fn=None, batch_size=self.batch_size, shuffle=True,
                                             drop_last=True)
        opt = torch.optim.Adam(self.trainable.parameters(), lr=.001)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=rm_scheduler_fn)
        else:
            scheduler = None
        self.batch_num = 0
        for self.epoch in range(self.max_epoch):
            print("Epoch: ", self.epoch)
            self.total_log_prob = 0.0
            self.batch_len = 0
            for (_, batch) in enumerate(dataloader):
                self.trainable.zero_grad()
                log_prob = torch.mean(self.batch_log_prob(batch))
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

    def log_prob(self, batch):
        raise("Unimplemented, Abstract Base Class")


class DistributionTrainer(_Trainer):
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, model_path=None):
        super(DistributionTrainer, self).__init__(trainable, dataset, batch_size, max_epoch, batch_end_callback,
                                           epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run, model_path=model_path)

    def batch_log_prob(self, batch):
        return self.trainable.log_prob(batch[0].to(self.device))


class LayerTrainer(_Trainer):
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None, use_scheduler=False, dummy_run=False, reverse_inputs=False, model_path=None):
        super(LayerTrainer, self).__init__(trainable, dataset, batch_size, max_epoch, batch_end_callback,
                 epoch_end_callback, use_scheduler=use_scheduler, dummy_run=dummy_run, model_path=model_path)
        self.reverse_inputs = reverse_inputs

    def batch_log_prob(self, batch):
        if not self.reverse_inputs:
            return self.trainable(batch[0].to(self.device)).log_prob(batch[1].to(self.device))
        else:
            return self.trainable(batch[1].to(self.device)).log_prob(batch[0].to(self.device))


# Ref 1: "What is an adaptive step size in parameter estimation", youtube, ian explains signals, systems and digital comms, June 20, 2022
