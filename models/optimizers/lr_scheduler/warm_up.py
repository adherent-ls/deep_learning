import math

import numpy as np


def cosine_decay(global_step, max_steps, initial_value, end_value):
    global_step = min(global_step, max_steps)
    cosine_decay_value = 0.5 * (1 + math.cos(math.pi * global_step / max_steps))
    return (initial_value - end_value) * cosine_decay_value + end_value


class Warmup(object):
    def __init__(self, optimizer, warm=0, step=0, min_lr=1e-6, max_step=1e7) -> None:
        super(Warmup, self).__init__()
        self.optimizer = optimizer
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.warm = warm
        self.n = step
        self.min_lr = min_lr
        self.lr = self.base_lr
        self.max_step = max_step

    def decay(self, lr) -> float:
        if self.n <= self.warm:
            lr = lr * np.sin((np.pi / 2) * (self.n / self.warm))
        else:
            # lr = lr * (1 / np.sqrt(self.n / self.warm))
            lr = cosine_decay(
                self.n - self.warm, self.max_step - self.warm, lr, self.min_lr
            )
        return lr

    def get_last_lr(self):
        return self.lr

    def step(self, epoch=None) -> None:
        if epoch is None:
            self.n += 1
        else:
            self.n = epoch
        self.lr = [self.decay(lr) for lr in self.base_lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.lr[i]
