# basic block layer normal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = shortcut + x
        return x


class FeatureDown(nn.Module):
    def __init__(self, dim):
        super(FeatureDown, self).__init__()
        self.stage = nn.ModuleList()
        self.down_feature = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2),
                                          Block(dim=dim),
                                          Block(dim=dim))

    def forward(self, x):
        x = self.down_feature(x)
        return x


class FeatureUp(nn.Module):
    def __init__(self, dim):
        super(FeatureUp, self).__init__()

        self.up_feature = nn.Sequential(
            Block(dim=dim),
            Block(dim=dim),
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x = self.up_feature(x)
        return x


