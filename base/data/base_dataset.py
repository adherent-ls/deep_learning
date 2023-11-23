import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.build_param import build_param


class BaseDataset(Dataset):
    def __init__(self, transforms=None, is_cache=True, recache=False, cache_file=None):
        super(BaseDataset, self).__init__()
        self.is_cache = is_cache
        self.recache = recache
        self.cache_file = cache_file
        self.transforms = transforms

    @staticmethod
    def initialization(cls, **kwargs):
        param = build_param(cls, kwargs)
        obj = cls(**param)

        super_param = build_param(super(cls, obj), kwargs)
        super(cls, obj).__init__(**super_param)
        return obj
