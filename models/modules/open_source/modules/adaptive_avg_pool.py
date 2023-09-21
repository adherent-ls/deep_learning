from torch import nn



class AdaptiveAvgPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveAvgPool2d, self).__init__()
        self.net = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        y = self.net(x.permute(0, 3, 1, 2))
        y = y.squeeze(3)
        return y
