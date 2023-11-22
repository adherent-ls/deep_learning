from base.data.base_transformer import BaseTransformer


class LoadBatchKeys(BaseTransformer):
    def __init__(self, keys, **kwargs):
        super(LoadBatchKeys, self).__init__(**kwargs)
        self.keys = keys

    def __call__(self, datas):
        data = {}
        for key in self.keys:
            data[key] = []
        for item in datas:
            if item is None:
                continue
            for key in self.keys:
                data[key].append(item[key])
        return data
