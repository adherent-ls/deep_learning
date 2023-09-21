from torch.nn import functional as F
from base.model.base_model import BaseModel


class BatchSimilarity(BaseModel):
    def __init__(self, **kwargs):
        super(BatchSimilarity, self).__init__(**kwargs)

    def forward(self, x, y):
        out = x @ y.t()
        if not self.training:
            out = F.softmax(out, dim=-1)
        return out
