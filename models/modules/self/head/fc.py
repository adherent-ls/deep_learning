from torch.nn import functional as F
from torch import nn

from base.model.base_model import BaseModel


class FCPrediction(BaseModel):
    def __init__(self, in_channel, n_class, **kwargs):
        super(FCPrediction, self).__init__(**kwargs)
        self.linear = nn.Linear(in_channel, n_class)

    def forward(self, x):
        pred = self.linear(x)
        if not self.training:
            pred = F.softmax(pred, dim=-1)
        return pred
