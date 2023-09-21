import math

import torch
from torch import nn

from base.model.base_model import BaseModel


class Center(BaseModel):
    def __init__(self, in_channel, out_channel, max_length, hidden_channel=None, **kwargs):
        super(Center, self).__init__(**kwargs)
        if hidden_channel is None:
            hidden_channel = out_channel
        self.max_length = max_length
        self.mask_pred = nn.Linear(in_channel, max_length)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.feedword = nn.Linear(in_channel, hidden_channel)
        self.pred = nn.Linear(hidden_channel, out_channel)

    def build_mask(self, x):
        out = self.mask_pred(x)
        out = out.permute(0, 2, 1)
        out = self.softmax(out)

        b, l, w = out.shape
        pos = torch.arange(0, w)[None, None, :].repeat((b, l, 1)).to(x.device)  # w -> 1*1*w -> b*l*w
        center_pos = torch.sum(out * pos, dim=-1)[..., None]  # b*l -> b*l*1

        v = -0.5 * 2 * math.log(2) * (pos - center_pos) ** 2
        mask = torch.exp(v)

        mask = mask / (torch.sum(mask, dim=-1)[:, :, None] + 1e-6)
        return mask

    def forward(self, x):
        mask = self.build_mask(x)
        feedward = self.feedword(x)
        out = mask @ feedward
        pred = self.pred(out)
        return pred
