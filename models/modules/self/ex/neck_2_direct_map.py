from torch import nn

from base.model.base_model import BaseModel


class DirectMap(BaseModel):
    def __init__(self, in_channel, out_channel, max_length, hidden_channel=None, **kwargs):
        super(DirectMap, self).__init__(**kwargs)
        if hidden_channel is None:
            hidden_channel = out_channel
        self.max_length = max_length
        self.mask_pred = nn.Linear(in_channel, max_length)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.feedword = nn.Linear(in_channel, hidden_channel)
        self.pred = nn.Linear(hidden_channel, out_channel)

    def build_mask(self, x):
        out = self.mask_pred(x)
        out = out.permute(0, 2, 1)
        out = self.softmax(out)
        return out

    def forward(self, x):
        mask = self.build_mask(x)
        feedward = self.feedword(x)
        out = mask @ feedward
        pred = self.pred(out)
        return pred
