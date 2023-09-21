
"""
因为存在multi scale的情况，所以需要需要多次循环进行grid_sample，因此速度较慢，原文发布cuda和python两个版本计算方式
"""
from torch import nn


class MSDeformAttention(nn.Module):
    def __init__(self):
        super(MSDeformAttention, self).__init__()

