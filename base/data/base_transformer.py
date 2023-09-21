import numpy as np


class BaseTransformer(object):
    def __init__(self, in_key='images', out_key='images'):
        self.in_key = in_key
        self.out_key = out_key

    def __call__(self, data):
        if len(self.in_key) != 0:
            if isinstance(self.in_key, str):
                in_key = [self.in_key]
            else:
                in_key = self.in_key
            xs = []
            for item in in_key:
                xs.append(data[item])
        else:
            xs = []

        y = self.forward(*xs)

        if isinstance(self.out_key, list):
            for key, data_item in zip(self.out_key, y):
                data[key] = data_item
        else:
            data[self.out_key] = y
        return data

    def forward(self, *args):
        return args
