import numpy as np

from base.data.base_transform import ImageTransform


class ImageNormal(ImageTransform):
    def __init__(self, scale=1.0 / 255.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageNormal, self).__init__()
        self.scale = scale
        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape)
        self.std = np.array(std).reshape(shape)

    def forward(self, image):
        image = (image * self.scale - self.mean) / self.std
        return image
