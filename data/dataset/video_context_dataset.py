import os

from base.data.base_dataset import BaseDataset


class VideoContextDataset(BaseDataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.video_names = os.listdir(self.root)
    