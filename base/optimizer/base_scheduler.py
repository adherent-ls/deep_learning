import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from base.base.base_instance_call import BaseInstanceCall
from utils.build_param import build_param


class BaseScheduler(StepLR.__base__, BaseInstanceCall):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        return self._last_lr

    def get_lr(self):
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))

    def step(self, epoch=None):
        pass


if __name__ == '__main__':
    model = torch.nn.Conv2d(3, 3, 3)
    adam = torch.optim.Adam(model.parameters(), lr=0.02)
    base = BaseScheduler.initialization(BaseScheduler, **{'optimizer': adam})
    print(base)
