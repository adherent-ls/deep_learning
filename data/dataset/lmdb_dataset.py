import re
import sys

import lmdb
import six
import tqdm
from PIL import Image
from torch.utils.data import Dataset


class LmdbDataset(Dataset):

    def __init__(self, root, filters, transforms):

        self.root = root
        self.transforms = transforms

        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            bar = tqdm.tqdm(range(nSamples))
            data = []
            data_indices = []
            for index in bar:
                index = str(index).zfill(9)
                label_key = f'label-{index}'.encode()
                img_key = f'image-{index}'.encode()
                imgbuf = txn.get(img_key)
                label = txn.get(label_key)
                data.append([imgbuf, label])
                data_indices.append(index)
            if filters is not None:
                indices = filters(root, data)
            else:
                indices = [i for i in range(nSamples)]
            self.indices = indices
            self.data_indices = data_indices
        print(len(self))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.data_indices[self.indices[index]]
        with self.env.begin(write=False) as txn:
            label_key = f'label-{index}'.encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = f'image-{index}'.encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')  # for color image
        return self.transforms(img, label)
