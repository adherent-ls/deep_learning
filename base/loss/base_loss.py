from torch.nn.modules.loss import _Loss

from base.base.base_instance_call import BaseInstanceCall


class BaseLoss(_Loss, BaseInstanceCall):
    def __init__(self):
        super(BaseLoss, self).__init__()
