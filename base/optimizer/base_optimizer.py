from torch.optim import Optimizer

from utils.build_param import build_param


class BaseOptimizer(Optimizer):
    def __init__(self, parameter, defaults):
        super(BaseOptimizer, self).__init__(parameter, defaults)

    @staticmethod
    def initialization(cls, **kwargs):
        param = build_param(cls, kwargs)
        obj = cls(**param)

        super_param = build_param(super(cls, obj), kwargs)
        super(cls, obj).__init__(**super_param)
        return obj