import os

import numpy as np
from tqdm import tqdm

from base.data.base_filter import BaseFilter


class TextDataFilter(BaseFilter):
    def __init__(self, filter, is_cache=True, cache_file=None, recache=False):
        super().__init__()
        self.filter = filter
        self.is_cache = is_cache
        self.cache_file = cache_file
        self.recache = recache

    def forward(self, root, ori_data):
        data = []
        if self.cache_file is None:
            cache_path = os.path.join(os.path.dirname(root), 'cache.npy')
        else:
            cache_path = os.path.join(os.path.dirname(root), self.cache_file)
        if self.is_cache and os.path.exists(cache_path) and not self.recache:
            data = np.load(cache_path)
        else:
            if self.filter is not None:
                bar = tqdm(ori_data)
                for i, item in enumerate(bar):
                    image, label = item
                    is_valid = self.filter(image, label)
                    if not is_valid:
                        data.append(i)
            else:
                data = np.arange(0, len(ori_data)).tolist()
            if self.is_cache:
                np.save(cache_path, np.array(data))
        return data
