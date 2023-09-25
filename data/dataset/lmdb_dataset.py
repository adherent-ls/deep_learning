import sys

import lmdb
import numpy as np
import six
from PIL import Image

from base.data.base_dataset import BaseDataset


class LmdbDataset(BaseDataset):

    def __init__(self, root):
        super(LmdbDataset, self).__init__()

        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % root)
            sys.exit(0)
        self.txn = self.env.begin(write=False)

        n_samples = int(self.txn.get('num-samples'.encode()))
        self.n_samples = n_samples
        print(self.n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        label_key = f'label-{str(index).zfill(9)}'.encode()
        img_key = f'image-{str(index).zfill(9)}'.encode()

        # if self.txn.get(label_key) is None:
        #     return None
        print(label_key, img_key)

        label = self.txn.get(label_key).decode('utf-8')
        imgbuf = self.txn.get(img_key)
        # if imgbuf is None:
        #     return None
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            image = Image.open(buf).convert('RGB')  # for color image
            image = np.array(image)
            buf.close()
        except IOError:
            buf.close()
            return None
        return image, label
