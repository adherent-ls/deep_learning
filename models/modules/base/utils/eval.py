import torch
from torch import nn


class Evaluation(nn.Module):
    def __init__(self):
        super(Evaluation, self).__init__()
        self.n1 = torch.randn(5, 4)
        self.n2 = torch.randn(7, 4)
        self.n4 = torch.randn(3, 4)

    def forward(self):
        t1 = ((self.n2 @ self.n1.T) @ self.n1) @ self.n4.T
        t2 = ((self.n4 @ self.n1.T) @ self.n1) @ self.n2.T
        print(t1)
        print(t2.T)


if __name__ == '__main__':
    e = Evaluation()
    e()
