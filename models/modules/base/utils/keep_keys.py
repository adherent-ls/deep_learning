from base.model.base_model import BaseModel


class KeepKeys(BaseModel):
    def __init__(self, keys):
        super(KeepKeys, self).__init__()
        self.keys = keys

    def forward(self, data):
        res = []
        for key in self.keys:
            res.append(data[key])
        return res
