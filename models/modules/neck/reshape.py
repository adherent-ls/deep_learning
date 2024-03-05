from torch import nn


class Reshape(nn.Module):
    def __init__(self, shape, permute=(0, 2, 1)):
        super().__init__()
        self.shape = shape
        self.permute = permute

    def forward(self, data):
        data = data.reshape(*self.shape)
        data = data.permute(*self.permute)
        return data
