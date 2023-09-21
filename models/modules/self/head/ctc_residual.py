from torch.nn import functional as F
from torch import nn

from base.model.base_model import BaseModel


class CTCResidual(BaseModel):
    def __init__(self, in_channel, n_class, **kwargs):
        super(CTCResidual, self).__init__(**kwargs)

        n_class = n_class if isinstance(n_class, int) else eval(n_class)

        self.fc = nn.Linear(in_channel, n_class)

    def forward(self, x):
        pred = self.fc(x)
        if not self.training:
            pred = F.softmax(pred, dim=-1)
        return pred
