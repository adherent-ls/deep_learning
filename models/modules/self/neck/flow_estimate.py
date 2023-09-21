import numpy as np
import torch.nn

from base.model.base_model import BaseModel


class FlowEstimate(BaseModel):
    def __init__(self, in_c, out_c, max_length, **kwargs):
        super(FlowEstimate, self).__init__(**kwargs)
        beta = torch.rand((2, max_length))

        self.estimate = torch.nn.Parameter(beta)

    def forward(self, x):
        b, l, c = x.shape
        self.normal = torch.arange(0, l).to(x.device)

        mean = (l / 2) * self.estimate[0]
        std = 1 * self.estimate[1]


        return x


if __name__ == '__main__':
    # inp = torch.rand(1, 12)  # b, l=rand
    # model = FlowEstimate(32, 64, 5)
    # y = model({'feature': inp})
    # print(y)
    x0 = np.random.random((1,))[0] * 100
    x = np.arange(0, 100).astype(np.float64)
    log_y = (x - x0) ** 2
    a = np.abs(x - np.sqrt(log_y))
    b = a.max()
    print((x - b) ** 2)
    print((x - x0) ** 2)
    print(a, x0, b)
