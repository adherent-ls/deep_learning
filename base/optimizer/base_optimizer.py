from torch.optim import Optimizer

from base.base.base_instance_call import BaseInstanceCall
from utils.build_param import build_param


class BaseOptimizer(Optimizer, BaseInstanceCall):
    def __init__(self, parameter, defaults):
        super(BaseOptimizer, self).__init__(parameter, defaults)
