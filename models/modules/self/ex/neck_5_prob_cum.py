import torch
from torch import nn

from base.model.base_model import BaseModel


class ProbCum(BaseModel):
    def __init__(self, in_channel, out_channel, max_length, **kwargs):
        super(ProbCum, self).__init__(**kwargs)

        self.center = torch.arange(0, max_length) * 2 + 1

        self.word_prob = nn.Linear(in_channel, 1)
        self.sigmoid = nn.Sigmoid()

        self.feed_word = nn.Linear(in_channel, out_channel)

    def build_mask(self, x):
        out = self.word_prob(x).squeeze(-1)  # (b,w,1) -> (b,w)

        prob = self.sigmoid(out)
        prob_sum = torch.cumsum(prob, dim=-1)  # 因为是两边，所以整体的和为2

        if self.center.device != x.device:
            self.center = self.center.to(x.device)

        # [(b,w)->(b,1,w)] - [(o,)->(1,o,1)] -> (b,o,w)
        attention = torch.exp(-0.5 * 2 * (prob_sum[:, None, :] - self.center[None, :, None]))
        return attention

    def forward(self, x):
        mask = self.build_mask(x)
        feedward = self.feedword(x)
        out = mask @ feedward
        pred = self.pred(out)
        return pred
