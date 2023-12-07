import math

from torch import nn

from base.model.base_model import BaseModel
from models.modules.base.network.cnn_pattern import ConvBlock


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, norm_fn=nn.LayerNorm):
        super(EncoderBlock, self).__init__()

        self.is_same = stride == 1 and padding == (kernel_size - stride) // 2

        self.conv = ConvBlock(in_ch, out_ch, kernel_size, stride, padding, bias)

        self.norm1 = norm_fn(in_ch)
        self.norm2 = norm_fn(out_ch)

        self.ff = MLP(in_features=out_ch, out_features=out_ch)

    def forward(self, x):
        if self.is_same:
            x = x + self.conv(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            x = x + self.ff(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        else:
            x = self.conv(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            x = self.ff(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x


class CNNEncoder(BaseModel):
    def __init__(self, input_channel, output_channel, layers=(1, 2, 5, 3)):
        super(CNNEncoder, self).__init__()

        outs = [int(output_channel // math.pow(2, i)) for i in range(len(layers) + 1)][::-1]
        ks = [3] * (len(layers) + 1)

        modules = nn.ModuleList()

        block = EncoderBlock(input_channel, outs[0],
                             kernel_size=7, stride=2, padding=3, bias=False)
        modules.append(block)

        for i in range(len(layers)):
            layer_num = layers[i]

            in_c = outs[i]
            out_c = outs[i + 1]
            stride = 2

            block = EncoderBlock(in_c, out_c,
                                 kernel_size=ks[i], stride=stride, padding=1, bias=False)
            modules.append(block)

            for i in range(1, layer_num):
                block = EncoderBlock(out_c, out_c,
                                     kernel_size=ks[i], stride=1, padding=(ks[i] - 1) // 2, bias=False)
                modules.append(block)

        self.module_list = modules

    def forward(self, x):
        y = x
        for item in self.module_list:
            y = item(y)
        return y
