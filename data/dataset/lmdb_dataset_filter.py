import os.path
import sys

import lmdb
import numpy as np
import six
from PIL import Image
import tqdm

from base.data.base_dataset import BaseDataset


class LmdbDatasetFilter(BaseDataset):

    def __init__(self, root):
        super(LmdbDatasetFilter, self).__init__()
        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)
        self.txn = self.env.begin(write=False)

        nSamples = int(self.txn.get('num-samples'.encode()))
        if self.cache_path is None:
            cache_path = os.path.join(root, 'cache.npy')
        else:
            cache_path = self.cache_path
        if self.is_cache and os.path.exists(cache_path) and not self.recache:
            indices = np.load(cache_path)
        else:
            indices = []
            bar = tqdm.tqdm(range(nSamples))
            for index in bar:
                index = str(index).zfill(9)
                label_key = f'label-{index}'.encode()
                img_key = f'image-{index}'.encode()
                imgbuf = self.txn.get(img_key)
                label = self.txn.get(label_key)
                is_valid = self.filter(imgbuf, label)
                if not is_valid:
                    continue
                indices.append(index)
            if self.is_cache:
                np.save(cache_path, np.array(indices))
        self.nSamples = len(indices)
        self.indices = indices
        print(self.nSamples)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        in_indices = self.indices[index]
        label_key = f'label-{in_indices}'.encode()
        label = self.txn.get(label_key)

        img_key = f'image-{in_indices}'.encode()
        imgbuf = self.txn.get(img_key)

        image, label = self.transforms(imgbuf, label)
        return image, label
