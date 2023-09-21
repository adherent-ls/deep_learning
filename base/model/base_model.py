import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, in_key='images', out_key='images'):
        super(BaseModel, self).__init__()
        self.in_key = in_key
        self.out_key = out_key

    def dict_forward(self, data):
        if isinstance(self.in_key, str):
            in_key = [self.in_key]
        else:
            in_key = self.in_key
        xs = []
        for item in in_key:
            xs.append(data[item])
        y = super(BaseModel, self).__call__(*xs) # 调用forward函数
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
