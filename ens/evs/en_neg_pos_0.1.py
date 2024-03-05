import cv2
import numpy as np
import torch.optim
from torch import nn

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
    #     [1] * 1 + [0.1] * 9 + [0] * 10,
    #     [0] * 10 + [0.1] * 9 + [1] * 1,
    #     # [0.1] + [0] * 0 + [0.1],
    #     # [1] + [0] * 0 + [1],
    # ])
    mid = 1
    data1 = np.tri(100, k=0) - np.tri(100, k=-mid)
    data2 = np.tri(100, k=-mid) * 0.1 - np.tri(100, k=-10) * 0.1
    data = data1 + data2
    data = data.T

    # noise = (np.random.random(data.shape) - 0.5) * 0.01
    # data = noise + data

    n, m = data.shape
    data = torch.Tensor(data)
    d1 = nn.Parameter(data)
    trainer = Trainer(m, n, [])
    # trainer.model.weight = nn.Parameter(torch.ones(trainer.model.weight.shape) * 0.1)
    # trainer.optim.add_param_group({'params': trainer.model.weight})
    for i in range(10000):
        label = torch.arange(0, n).long()
        w1 = trainer.model.weight + 0
        # if i % 100 == 0:
        #     trainer.model.weight = nn.Parameter(w1 / torch.norm(w1))
        #     trainer.optim.add_param_group({'params': trainer.model.weight})
        y1, loss = trainer.do_batch(d1, label)
        print(i, loss, trainer.model.weight.var())
        if loss < 0.01 and i > 100:
            break
    # w = trainer.model.weight.detach().numpy()
    # for i in range(len(data)):
    #     w[i, i + 1:i + 10] = w[i, i + 1:i + 10][::-1]
    ori = data.detach().numpy()
    step_l = 0.1
    for j in range(n):
        x1 = ori[j:j + 1]
        x1 = torch.Tensor(x1)
        g = 0
        loss = 0
        for i in range(1000):
            x2 = torch.clip(x1 + g, 0)
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
