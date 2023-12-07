import os

from torch.utils.data import Dataset

from base.base.base_dict_call import BaseDictCall


class BaseDataset(Dataset, BaseDictCall):
    def __init__(self, transforms=None, is_cache=True, recache=False, cache_file=None):
        super(BaseDataset, self).__init__()
        self.is_cache = is_cache
        self.recache = recache
        self.cache_file = cache_file
        self.transforms = transforms

    def save_cache(self, data_index):
        raise NotImplemented

    def get_cache(self):
        raise NotImplemented
