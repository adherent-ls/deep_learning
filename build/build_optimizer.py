import torch

from utils.register import register


class OptimizerWithScheduler(object):
    def __init__(self, model: torch.nn.Module, config):
        super(OptimizerWithScheduler, self).__init__()
        self.optimizer = self.build_optimizer(model.parameters(), config['Optimizer'])
        self.scheduler = self.build_scheduler(self.optimizer, config['Scheduler'])
        self.base_lr = [group['lr'] for group in self.optimizer.param_groups]

    def build_optimizer(self, parameters, config):
        name = config['name']
        args = config['args']
        args['params'] = parameters
        optim = register.build_from_config(name, args, 'optim')
        return optim

    def build_scheduler(self, optim, config):
        name = config['name']
        args = config['args']
        args['optimizer'] = optim
        scheduler = register.build_from_config(name, args, 'scheduler')
        return scheduler

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
