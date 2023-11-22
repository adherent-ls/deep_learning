import torch
from torch import nn

from utils.build_param import build_param


class BaseModel(nn.Module):
    def __init__(self, ink='images', ouk=None):
        super().__init__()
        self.ink = ink
        if ouk is None:
            self.ouk = ouk

    @staticmethod
    def initialization(cls, **kwargs):
        param = build_param(cls, kwargs)
        obj = cls(**param)

        super_param = build_param(super(cls, obj), kwargs)
        super(cls, obj).__init__(**super_param)
        return obj

    def dict_forward(self, data):
        if isinstance(self.in_key, str):
            in_key = [self.in_key]
        else:
            in_key = self.in_key
        xs = []
        for item in in_key:
            xs.append(data[item])
        y = super(BaseModel, self).__call__(*xs)  # 调用forward函数
        if isinstance(self.out_key, list):
            for key, data_item in zip(self.out_key, y):
                data[key] = data_item
        else:
            data[self.out_key] = y
        return data

    def instance_forward(self, data):
        if isinstance(data, torch.Tensor):
            y = super(BaseModel, self).__call__(data)
        else:
            y = super(BaseModel, self).__call__(*data)
        return y

    def __call__(self, data):
        return self.dict_forward(data)
