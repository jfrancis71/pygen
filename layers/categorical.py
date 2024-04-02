import torch


class Categorical(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.distributions.categorical.Categorical(logits=logits)
