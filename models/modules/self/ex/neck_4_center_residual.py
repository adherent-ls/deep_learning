import math

import torch
from torch import nn

from base.model.base_model import BaseModel


class CenterResidual(BaseModel):
    def __init__(self, in_channel, out_channel, max_length, layer_num=4, **kwargs):
        super(CenterResidual, self).__init__(**kwargs)
        self.max_length = max_length
        self.w_o_covariance = nn.Linear(in_channel, max_length)
        self.softmax = nn.Softmax(dim=-1)

        self.residual_layers = nn.ModuleList()
        for i in range(layer_num):
            self.residual_layers.append(nn.Linear(in_channel, max_length))

        self.pred = nn.Linear(out_channel, out_channel)

    def w_o_covariance_func(self, x):
        out = self.w_o_covariance(x)
        out = out.permute(0, 2, 1)
        out = self.softmax(out)

        b, l, w = out.shape
        pos = torch.arange(0, w)[None, None, :].repeat((b, l, 1)).to(x.device)  # w -> 1*1*w -> b*l*w
        center_pos = torch.sum(out * pos, dim=-1)[..., None]  # b*l -> b*l*1

        return center_pos

    def o_o_covariance_func(self, x, layer):
        out = layer(x)
        out = self.softmax(out)

        b, o, o = out.shape
        pos = torch.arange(0, o)[None, None, :].repeat((b, o, 1)).to(x.device)  # w -> 1*1*w -> b*l*w
        center_pos = torch.sum(out * pos, dim=-1)[..., None]  # b*l -> b*l*1

        return center_pos

    def residual(self, x, center, layer):
        b, o, c = x.shape
        pos = torch.arange(0, o)[None, None, :].repeat((b, 1, 1)).to(x.device)  # w -> 1*1*w
        v = -0.5 * 2 * math.log(2) * (pos - center) ** 2
        mask = torch.exp(v)
        mask = mask / (torch.sum(mask, dim=-1)[:, :, None] + 1e-6)

        out = mask @ x

        new_center = self.o_o_covariance_func(out, layer)
        new_center = new_center + center
        return new_center

    def forward(self, x):
        b, w, c = x.shape
        center = self.w_o_covariance_func(x)

        for layer in self.residual_layers:
            center = self.residual(x, center, layer)

        pos = torch.arange(0, w)[None, None, :].repeat((b, 1, 1)).to(x.device)  # w -> 1*1*w
        v = -0.5 * 2 * math.log(2) * (pos - center) ** 2
        mask = torch.exp(v)
        mask = mask / (torch.sum(mask, dim=-1)[:, :, None] + 1e-6)

        out = mask @ x
        pred = self.pred(out)
        return pred
