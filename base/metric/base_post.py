import numpy as np

from utils.build_param import build_param


class BaseMetric(object):
    @staticmethod
    def initialization(cls, **kwargs):
        param = build_param(cls, kwargs)
        obj = cls(**param)

        super_param = build_param(super(cls, obj), kwargs)
        super(cls, obj).__init__(**super_param)
        return obj

    def __call__(self, *args):
        return args
