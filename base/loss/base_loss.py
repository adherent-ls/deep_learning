from torch.nn.modules.loss import _Loss

from utils.build_param import build_param


class BaseLoss(_Loss):
    def __init__(self):
        super(BaseLoss, self).__init__()

    @staticmethod
    def initialization(cls, **kwargs):
        param = build_param(cls, kwargs)
        obj = cls(**param)

        super_param = build_param(super(cls, obj), kwargs)
        super(cls, obj).__init__(**super_param)
        return obj
