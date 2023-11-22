import os.path

import cv2
import numpy as np

from base.data.base_dataset import BaseDataset


class SimpleDataset(BaseDataset):
    def __init__(self, image_root, label_file):
        super().__init__()
        self.image_root = image_root
        self.label_file = label_file
        lines = open(self.label_file).readlines()

        data = []
        if self.cache_path is None:
            cache_path = os.path.join(label_file, 'cache.npy')
        else:
            cache_path = self.cache_path
        if self.is_cache and os.path.exists(cache_path) and not self.recache:
            data = np.load(cache_path)
        else:
            for line in lines:
                image_name, label = line.strip('\n').split('\t')
                image = cv2.imread(os.path.join(self.image_root, image_name))
                is_valid = self.filter(image, label)
                if not is_valid:
                    data.append([image_name, label])
            if self.is_cache:
                np.save(cache_path, np.array(data))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image_name, label = self.data[index]
        image = cv2.imread(os.path.join(os.path.join(self.image_root, image_name)))
        image, label = self.transforms(image, label)
        return image, label
