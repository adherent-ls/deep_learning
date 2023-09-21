from base.model.base_model import BaseModel


class LoadKeys(BaseModel):
    def __init__(self, key_pairs, **kwargs):
        super(LoadKeys, self).__init__(**kwargs)
        self.key_pairs = key_pairs

    def forward(self, input_data):
        data = {}
        for input_key, key in self.key_pairs.items():
            item = input_data[input_key]
            data[key] = item
        return data
