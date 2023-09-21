from base.data.base_transformer import BaseTransformer


class LoadKeys(BaseTransformer):
    def __init__(self, keys, **kwargs):
        super(LoadKeys, self).__init__(**kwargs)
        self.keys = keys

    def __call__(self, args):
        data = {}
        for key in self.keys:
            data[key] = []
        for item in args:
            for key, data_item in zip(self.keys, item):
                data[key].append(data_item)
        return data
