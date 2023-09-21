from torch import nn
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, is_cache=True, recache=False, cache_path=None):
        super(BaseDataset, self).__init__()
        self.is_cache = is_cache
        self.recache = recache
        self.cache_path = cache_path
