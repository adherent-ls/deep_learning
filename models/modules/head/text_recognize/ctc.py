from torch import nn


class CTC(nn.Module):
    def __init__(self, in_channel, num_class):
        super().__init__()
        self.linear = nn.Linear(in_channel, num_class)

    def forward(self, x):
        return self.linear(x)
