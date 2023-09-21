import os.path
import sys

import lmdb
import numpy as np
import six
from PIL import Image
import tqdm
from torch.utils.data import Dataset


class LmdbDatasetFilter(Dataset):

    def __init__(self, root, filter=None, is_cache=True, recache=False, cache_path=None):

        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)
        self.txn = self.env.begin(write=False)

        nSamples = int(self.txn.get('num-samples'.encode()))
        if cache_path is None:
            cache_path = os.path.join(root, 'cache.npy')
        else:
            cache_path = os.path.join(root, cache_path)
        if is_cache and os.path.exists(cache_path) and not recache:
            indices = np.load(cache_path)
        else:
            indices = []
            bar = tqdm.tqdm(range(nSamples))
            for index in bar:
                index = str(index).zfill(9)
                label_key = f'label-{index}'.encode()
                if self.txn.get(label_key) is None:
                    continue
                img_key = f'image-{index}'.encode()
                imgbuf = self.txn.get(img_key)
                if imgbuf is None:
                    continue
                label = self.txn.get(label_key).decode('utf-8')
                is_valid = True
                for item_filter in filter:
                    item_valid = item_filter(imgbuf, label)
                    is_valid = is_valid and item_valid
                if not is_valid:
                    continue
                indices.append(index)
            if is_cache:
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
        label = self.txn.get(label_key).decode('utf-8')

        img_key = f'image-{in_indices}'.encode()
        imgbuf = self.txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')  # for color image
        image = np.array(image)
        buf.close()
        label = list(label) + ['end']
        return image, label
