from typing import Union

import math
import os.path

import cv2
import lmdb
import numpy as np
import torch
import tqdm

from data.dataset.lmdb_dataset import LmdbDataset
from data.transforms.image.lmdb_image_decode import LmdbImageDecode
from data.transforms.label.lmdb_stream_decode import LmdbStreamDecode


def check():
    root = '/home/data/data_old/lmdb/SynthText/train'

    env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    os.makedirs(os.path.join(root, 'check'), exist_ok=True)

    image_decode = LmdbImageDecode()
    label_decode = LmdbStreamDecode()
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        indices = np.random.randint(0, nSamples, (100))
        for i, index in enumerate(indices):
            index = str(index).zfill(9)
            label_key = f'label-{index}'.encode()
            image_key = f'image-{index}'.encode()
            label = txn.get(label_key)
            image = txn.get(image_key)

            image = np.array(image_decode.forward(image))
            label = label_decode.forward(label)
            print(label)
            cv2.imwrite(os.path.join(root, 'check', f'{i}_{label}.jpg'), image)


def load():
    path = r'/home/data/workspace/training_models/deep_learning/deep_learning_check_v2'
    param = torch.load(os.path.join(path, 'latest.pth'), map_location='cpu')
    print()
    bar = tqdm.tqdm(range(0, 9))
    bar = iter(bar)
    while True:
        item = next(bar)
        print(item)


def te():
    x = torch.rand((5, 16))
    x.requires_grad = True

    y = []
    for i in range(5):
        y.append(x[i] * (i + 1))

    y = torch.stack(y, dim=0)
    l = torch.sum(y, dim=1).sum()
    l.backward()

    print(x.grad)


def tt(data: Union[str, int] = None):
    print(data)
    return 0


if __name__ == '__main__':
# te()
# import torch
#
# x1 = torch.randn(3, 4)
# x2 = torch.randn(3, 4)
# similarity = torch.cosine_similarity(x1, x2, dim=1)
# print(similarity)
    tt()
