import numpy as np

from base.data.base_transformer import BaseTransformer


class ZeroMeanNormal(BaseTransformer):
    def __init__(self, scale, mean, std, **kwargs):
        super(ZeroMeanNormal, self).__init__(**kwargs)
        self.scale = scale
        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape)
        self.std = np.array(std).reshape(shape)

    def forward(self, image):
        image = (image * self.scale - self.mean) / self.std
        return image
