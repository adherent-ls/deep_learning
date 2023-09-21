import re
import sys

import cv2
import lmdb
import torch
import numpy as np
import six
from PIL import Image
from torch.utils.data import Dataset


class TextLabelBase(object):
    def __init__(self, character):
        self.character = character
        labels = {}
        for i, line in enumerate(character):
            labels[line] = i
        self.character_map = labels

    def once_encode(self, text):
        label = []
        for char_item in text:
            if char_item not in self.character_map:
                return None
            index = self.character_map[char_item]
            label.append(index)
        label.append(self.character_map[self.character[1]])
        label = np.array(label)
        return label

    def encode(self, text):
        labels = []
        for item in text:
            label = []
            for char_item in item:
                index = self.character_map[char_item]
                label.append(index)
            label.append(self.character_map[self.character[1]])
            labels.append(label)
        labels = np.array(labels)
        return labels

    def decode(self, index):
        texts = []
        for item in index:
            text = ''
            for single in item:
                chars = self.character[int(single)]
                if chars == 'end':
                    break
                if chars != 'blank':
                    text += self.character[int(single)]
            texts.append(text)
        return texts


class LmdbDataset(Dataset):

    def __init__(self, root, characters, imgH=32, max_length=50, padding=8):
        self.label_client = TextLabelBase(characters)

        self.imgH = imgH
        self.max_length = max_length
        self.padding = padding
        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)
        self.txn = self.env.begin(write=False)

        nSamples = int(self.txn.get('num-samples'.encode()))
        self.nSamples = nSamples
        print(self.nSamples)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        label_key = f'label-{str(index).zfill(9)}'.encode()
        label = self.txn.get(label_key).decode('utf-8')

        if label == '/Name' or label == '/Father Name':
            return None

        img_key = f'image-{str(index).zfill(9)}'.encode()
        imgbuf = self.txn.get(img_key)

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

    def collate_fn(self, datas):
        max_radio = 0
        b = 0
        max_c = 0
        valid_data = []
        for item in datas:
            if item is None:
                continue
            image, label = item
            label = self.label_client.once_encode(label)
            if label is None:
                continue
            b += 1
            valid_data.append(item)
            h, w, c = image.shape
            max_radio = max(max_radio, w / h)
            max_c = max(max_c, c)
        if b == 0:
            return None, None, None
        max_w = int(max_radio * self.imgH) + 1 + self.padding
        max_h = self.imgH
        images = np.zeros((b, max_h, max_w, 1))
        labels = np.zeros((b, self.max_length))
        lengths = np.zeros((b,))
        for i, item in enumerate(valid_data):
            image, label = item
            h, w, c = image.shape
            w1 = int(w / h * self.imgH)
            label = label.strip('\n')
            label = self.label_client.once_encode(label)
            length = len(label)

            image = cv2.resize(image, (w1, max_h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]
            image = image / 255 * 2 - 1  # (0-255) -> (0-1) -> (0-2) -> (-1-1)

            images[i, :max_h, :w1, :c] = image
            labels[i, :length] = label
            lengths[i] = length
        images = torch.Tensor(images).permute(0, 3, 1, 2).float()
        labels = torch.Tensor(labels).long()
        lengths = torch.Tensor(lengths).long()
        return images, labels, lengths

    def decode(self, index):
        return self.label_client.decode(index)
