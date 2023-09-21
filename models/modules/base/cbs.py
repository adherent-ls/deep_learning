from torch import nn


class CB(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0, stride=1):
        super(CB, self).__init__()
        self.forward_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, data):
        return self.forward_layer(data)


class CBS(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0, stride=1):
        super(CBS, self).__init__()
        self.forward_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.SiLU()
        )

    def forward(self, data):
        return self.forward_layer(data)
