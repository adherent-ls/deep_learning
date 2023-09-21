from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    def __init__(self, parameter, defaults):
        super(BaseOptimizer, self).__init__(parameter, defaults)
