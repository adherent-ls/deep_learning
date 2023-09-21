import torch
from torch import nn

from base.model.base_model import BaseModel


class ProbCumNormal(BaseModel):
    def __init__(self, in_channel, out_channel, max_length, **kwargs):
        super(ProbCumNormal, self).__init__(**kwargs)

        self.max_length = max_length

        self.center = (2 * torch.arange(0, max_length) + 1) / 2

        self.word_prob = nn.Linear(in_channel, 1)
        self.sigmoid = nn.Sigmoid()

        self.feed_word = nn.Linear(in_channel, out_channel)
        self.pred = nn.Linear(out_channel, out_channel)

    def build_mask(self, x):
        out = self.word_prob(x).squeeze(-1)  # (b,w,1) -> (b,w)
        out = self.sigmoid(out)
        # out = x.squeeze(-1)

        prob = torch.clamp(out * 1.25 - 0.125, 0, 1)
        prob[:, 1:] = torch.clamp(prob[:, 1:] - prob[:, :-1], 0)
        prob_sum = torch.cumsum(prob, dim=-1)

        if self.center.device != x.device:
            self.center = self.center.to(x.device)

        # [(b,w)->(b,1,w)] - [(o,)->(1,o,1)] -> (b,o,w)

        poor = ((prob_sum[:, None, :] - self.center[None, :, None])) * 2.5
        attention = torch.exp(-0.5 * 2 * poor ** 2)
        attention = attention / (torch.sum(attention, dim=-1)[:, :, None] + 1e-6)
        return attention

    def forward(self, x):
        mask = self.build_mask(x)
        feed_word = self.feed_word(x)
        out = mask @ feed_word
        pred = self.pred(out)
        return pred


if __name__ == '__main__':
    m = ProbCumNormal(32, 32, 10)
    x = torch.rand((1, 18, 32))
    # x = torch.zeros((1, 18, 1))
    # cs = [5, 10, 15]
    # position = torch.arange(0, 18)
    # for i in range(3):
    #     x = x + torch.exp(-0.5 * 1 * (position[None, :, None] - cs[i]) ** 2)
    y = m.forward(x)
    print(y.shape)
