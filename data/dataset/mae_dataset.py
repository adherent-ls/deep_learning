import os

import cv2

from base.data.base_dataset import BaseDataset


class MAEDataset(BaseDataset):
    def __init__(self, root):
        super(MAEDataset, self).__init__()
        self.root = root
        self.dirs = os.listdir(self.root)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, item):
        image_path = os.path.join(self.root, self.dirs[item])
        image = cv2.imread(image_path)
        return image, image
