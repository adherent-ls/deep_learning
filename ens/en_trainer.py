import numpy as np
import torch
from torch import nn


class Trainer(object):

    def __init__(self, m, n, param=[]):
        self.model = nn.Linear(m, n, bias=False)

        self.optim1 = torch.optim.Adam(lr=0.1, params=list(self.model.parameters()) + param)
        self.optim2 = torch.optim.SGD(lr=1, params=list(self.model.parameters()) + param)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = self.optim1

    def train(self, ori):
        for i in range(100000):
            idx = np.arange(0, len(ori))
            x = ori[idx]
            x = torch.Tensor(x)
            w1 = self.model.weight.detach().numpy() + 0

            y = self.model(x)

            y = torch.log(torch.exp(y) / torch.sum(torch.exp(y), dim=1, keepdim=True))
            loss = self.loss_fn(y, torch.Tensor(idx).long())
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            weight = self.model.weight.detach().numpy()
            print(weight.sum(), weight.mean(), loss)
            if loss < 0.01 and i > 0:
                break

    def do_batch(self, image, label):
        y1 = self.model(image)
        # y1 = y1 - torch.logsumexp(y1, dim=1, keepdim=True)
        loss = self.loss_fn(y1, label)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return y1, loss

    def do_single_wo_update(self, image, label):
        y1 = self.model(image)
        y1 = y1 - torch.logsumexp(y1, dim=1, keepdim=True)
        loss = self.loss_fn(y1, label)
        loss.backward()
        return y1, loss
