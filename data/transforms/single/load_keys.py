from base.data.base_transformer import BaseTransformer


class LoadKeys(BaseTransformer):
    def __init__(self, keys, **kwargs):
        super(LoadKeys, self).__init__(**kwargs)
        self.keys = keys

    def __call__(self, datas):
        data = {}
        for key, data_item in zip(self.keys, datas):
            data[key] = data_item
        return data
