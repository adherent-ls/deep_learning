import numpy as np

from utils.build_param import build_param


class BaseTransformer(object):
    def __init__(self, ink='images', ouk=None):
        self.ink = ink
        if ouk is not None:
            self.ouk = ouk
        else:
            self.ouk = ink

    @staticmethod
    def initialization(cls, **kwargs):
        param = build_param(cls, kwargs)
        obj = cls(**param)

        super_param = build_param(super(cls, obj), kwargs)
        super(cls, obj).__init__(**super_param)
        return obj

    def __call__(self, data):
        if len(self.ink) != 0:
            if isinstance(self.ink, str):
                ink = [self.ink]
            else:
                ink = self.ink
            xs = []
            for item in ink:
                xs.append(data[item])
        else:
            xs = []

        y = self.forward(*xs)
        if isinstance(self.ouk, list) or isinstance(self.ouk, tuple):
            for key, data_item in zip(self.ouk, y):
                data[key] = data_item
        else:
            data[self.ouk] = y
        return data

    def forward(self, *args):
        return args


class TestTrans(BaseTransformer):
    def __init__(self, *t, **kwargs):
        super().__init__()
        self.t = 0
        self.t = kwargs['ink']
        self.t = 'ink1'


if __name__ == '__main__':
    t = TestTrans.init(TestTrans, t='i', ink='i')
    print(t)
