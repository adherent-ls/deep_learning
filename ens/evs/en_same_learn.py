import cv2
import numpy as np
import torch.optim
from torch import nn
from torch.nn import functional as F

from ens.en_trainer import Trainer


def construct_tensor(m, n):
    m = m - 1
    if m > 0:
        d0 = construct_tensor(m, n)
        d1 = np.eye(n)
        l = n // (2 ** m)
        idx = np.arange(0 + l, n + l) % n
        d1 = d1[idx]
        d0 = d0.reshape((-1, n, n))
        data = np.concatenate([d0 + d1, d0 * -1 + d1], axis=0)
        data = data.reshape((-1, n))
    else:
        data = np.eye(n)
        data = np.concatenate([data, data * -1], axis=0)
    return data


def main():
    # data = np.array([
    #     [1, 1, 1, 0],
    #     [0, 1, 1, 1],
    # ])
    data = np.eye(10)[:8]
    data[:, 8:] = 1

    n, m = data.shape
    d = torch.Tensor(data)
    trainer = Trainer(m, n)
    for i in range(100000):
        label = torch.arange(0, n).long()
        y1, loss, = trainer.do_batch(d, label)
        print(loss)
        if loss < 0.01 and i > 100:
            p1 = F.softmax(y1, dim=-1)
            break

    ori = d
    step_l = 0.5
    for j in range(n):
        x1 = ori[j:j + 1]
        x1 = torch.Tensor(x1)
        g = 0
        loss = 0
        for i in range(1000):
            x2 = x1 + g
            x2.requires_grad = True

            y1, loss = trainer.do_single_wo_update(x2, torch.Tensor([j]).long())

            grad = torch.sign(x2.grad) * step_l * 0.1
            g = g + grad + torch.rand(grad.shape) * step_l * 0.01
            g = torch.clip(g, -step_l, step_l)
        pred = y1[0].detach().numpy().argmax()
        if pred != j:
            t = 0
        print(loss, j, pred)
    t = 0


if __name__ == '__main__':
    main()
