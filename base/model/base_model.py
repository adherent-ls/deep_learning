import torch
from torch import nn

from utils.build_param import build_param


class BaseModel(nn.Module):
    def __init__(self, ink='images', ouk=None):
        super().__init__()
        self.ink = ink
        print(ouk, '-' * 80)
        if ouk is not None:
            self.ouk = ouk
        else:
            self.ouk = ink

    def dict_forward(self, data):
        if isinstance(self.ink, str):
            ink = [self.ink]
        else:
            ink = self.ink
        xs = []
        for item in ink:
            xs.append(data[item])
        y = super(BaseModel, self).__call__(*xs)  # 调用forward函数
        if isinstance(self.ouk, list):
            for key, data_item in zip(self.ouk, y):
                data[key] = data_item
        else:
            data[self.ouk] = y
        return data

    def instance_forward(self, data):
        if isinstance(data, torch.Tensor):
            y = super(BaseModel, self).__call__(data)
        else:
            y = super(BaseModel, self).__call__(*data)
        return y

    def __call__(self, data):
        return self.dict_forward(data)
