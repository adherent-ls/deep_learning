import numpy as np

from base.data.base_transform import ImageTransform


class ImagePadding(ImageTransform):
    def __init__(self, pad=0):
        super().__init__()
        self.pad = pad

    def forward(self, image):
        h, w, c = image.shape
        pad_image = np.zeros((h, w + self.pad, c), dtype=image.dtype)
        pad_image[:, self.pad:, :] = image
        return pad_image
