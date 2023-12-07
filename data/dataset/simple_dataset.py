import os.path

import cv2
import numpy as np
from tqdm import tqdm

from base.data.base_dataset import BaseDataset


class SimpleDataset(BaseDataset):
    def __init__(self, image_root, label_file, filter=None, transforms=None):
        super().__init__()
        self.image_root = image_root
        self.label_file = label_file
        self.transforms = transforms

        lines = open(self.label_file, encoding='utf-8').readlines()
        data = [line.strip('\n').split('\t') for line in lines]
        self.data = filter(self.label_file, data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_name, label = self.data[index]
        image = cv2.imread(os.path.join(os.path.join(self.image_root, image_name)))
        return self.transforms((image, label))
