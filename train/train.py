import torch


class _Trainer():
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None, epoch_end_callback=None):
        self.trainable = trainable
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.batch_end_callback = batch_end_callback
        self.epoch_end_callback = epoch_end_callback

    def train(self):
        self.device = next(self.trainable.parameters()).device
        dataloader = torch.utils.data.DataLoader(self.dataset, collate_fn=None, batch_size=self.batch_size, shuffle=True,
                                             drop_last=True)
        opt = torch.optim.Adam(self.trainable.parameters(), lr=.001)
        self.batch_num = 0
        for self.epoch in range(self.max_epoch):
            print("Epoch: ", self.epoch)
            self.total_log_prob = 0.0
            self.batch_len = 0
            for (_, batch) in enumerate(dataloader):
                self.trainable.zero_grad()
                log_prob = torch.mean(self.log_prob(batch))
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

    def log_prob(self, batch):
        raise("Unimplemented, Abstract Base Class")


class DistributionTrainer(_Trainer):
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None):
        super(DistributionTrainer, self).__init__(trainable, dataset, batch_size, max_epoch, batch_end_callback,
                                           epoch_end_callback)

    def log_prob(self, batch):
        return self.trainable.log_prob(batch[0].to(self.device))


class LayerTrainer(_Trainer):
    def __init__(self, trainable, dataset, batch_size=32, max_epoch=10, batch_end_callback=None,
                 epoch_end_callback=None):
        super(LayerTrainer, self).__init__(trainable, dataset, batch_size, max_epoch, batch_end_callback,
                 epoch_end_callback)

    def log_prob(self, batch):
        return self.trainable(batch[0].to(self.device)).log_prob(batch[1].to(self.device))
