import os

import numpy as np
from tqdm import tqdm


class BuildFilter(object):
    def __init__(self, root, filters, is_cache=True, cache_file=None, recache=False):
        super().__init__()
        self.root = root
        self.filters = filters
        self.is_cache = is_cache
        self.cache_file = cache_file
        self.recache = recache

    def __call__(self, ori_data):
        data = []
        if self.cache_file is None:
            cache_path = os.path.join(self.root, 'cache.npy')
        else:
            cache_path = os.path.join(self.root, self.cache_file)

        if self.is_cache and os.path.exists(cache_path) and not self.recache:
            data = np.load(cache_path)
        else:
            if self.filters is not None:
                bar = tqdm(ori_data)
                for i, item in enumerate(bar):
                    image, label = item
                    is_valid = True
                    for item in self.filters:
                        is_valid = item(image, label)
                        if not is_valid:
                            break
                    if is_valid:
                        data.append(i)
            else:
                data = np.arange(0, len(ori_data)).tolist()
            if self.is_cache:
                np.save(cache_path, np.array(data))
        return data
