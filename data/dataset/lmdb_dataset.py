import sys

import lmdb

from base.data.base_dataset import BaseDataset


class LmdbDataset(BaseDataset):

    def __init__(self, root):
        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            indices = [i for i in range(nSamples)]
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = str(self.indices[index]).zfill(9)
        with self.env.begin(write=False) as txn:
            label_key = f'label-{index}'.encode()
            image_key = f'image-{index}'.encode()

            label = txn.get(label_key)
            image = txn.get(image_key)

        return image, label
