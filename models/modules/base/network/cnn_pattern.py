import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def to_tuple(data):
    if isinstance(data, int):
        return (data, data)
    else:
        return data


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super(ConvBlock, self).__init__()
        self.kernel_size = to_tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.bias = bias

        weight_num = self.kernel_size[0] * self.kernel_size[1] * in_ch

        self.softmax = nn.Softmax(dim=-1)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.value = nn.Linear(weight_num, out_ch, bias=False)

        self.ln = nn.LayerNorm(out_ch)

    def forward(self, image):
        weight = self.ln(self.conv.weight.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
        out = F.conv2d(image, weight, stride=self.stride, padding=self.padding)
        att = out.permute(0, 2, 3, 1)
        att = self.softmax(att)

        oc, ic, kh, kw = weight.shape
        value = self.value(weight.view(oc, -1))

        y = att @ value  # b,h,w,hc @ hc,oc = b,h,w,oc
        y = y.permute(0, 3, 1, 2)
        return y


if __name__ == '__main__':
    conv_q = ConvBlock(4, 8, 3)
    # conv_q = conv_q.cuda()
    image = np.random.randn(1, 4, 12, 12).astype(np.float32)
    image = torch.Tensor(image)
    # image = image.cuda()
    out = conv_q(image)
