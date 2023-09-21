import os

import cv2
from torch.utils.data import Dataset


class MAEDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.dirs = os.listdir(self.root)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, item):
        image_path = os.path.join(self.root, self.dirs[item])
        image = cv2.imread(image_path)
        return image, image
