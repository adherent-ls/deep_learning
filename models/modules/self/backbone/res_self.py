from torch import nn

from base.model.base_model import BaseModel
from models.modules.base.cbs import CB, CBS


class BottleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, down_sample=True):
        super(BottleBlock, self).__init__()
        hidden_channel = out_channel // 64
        self.main_stream = nn.Sequential(
            CBS(in_channel, hidden_channel, kernel_size=1),
            CBS(hidden_channel, hidden_channel, kernel_size=3, padding=1, stride=stride),
            CB(hidden_channel, out_channel, kernel_size=1)
        )

        self.short_cut = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.activate = nn.BatchNorm2d(out_channel)

        if down_sample:
            self.down_sample = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=2)

    def forward(self, data):
        main_stream_data = self.main_stream(data)
        short_cut_data = self.short_cut(data)
        data = main_stream_data + short_cut_data

        data = self.activate(data)
        return data


class ResNetSelf(BaseModel):
    def __init__(self, in_channel, out_channel, block, block_nums=(1, 2, 6, 3)):
        super(ResNetSelf, self).__init__()
