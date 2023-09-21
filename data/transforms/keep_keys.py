import torch

from base.data.base_transformer import BaseTransformer


class KeepKeyTensor(BaseTransformer):
    def __init__(self, keep_data_keys, keep_label_keys, device='cpu', **kwargs):
        super(KeepKeyTensor, self).__init__(**kwargs)
        self.keep_data_keys = keep_data_keys
        self.keep_label_keys = keep_label_keys
        self.device = device

    def convert(self, data_item, type_name):
        data_item = torch.Tensor(data_item)
        if type_name == 'float32':
            data_item = data_item.float()
        if type_name == 'long':
            data_item = data_item.long()
        return data_item

    def __call__(self, data):
        input_data = {}
        for key, type_name in self.keep_data_keys.items():
            input_data[key] = self.convert(data[key], type_name)

        labels = {}
        for key, type_name in self.keep_label_keys.items():
            labels[key] = self.convert(data[key], type_name)
        return input_data, labels
