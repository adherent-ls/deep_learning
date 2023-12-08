import torch


class OptimizerWithScheduler(object):
    def __init__(self, optimizer, scheduler):
        super(OptimizerWithScheduler, self).__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.base_lr = [group['lr'] for group in self.optimizer.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        self.optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
        else:
            lr = self.base_lr
        return lr

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none)
