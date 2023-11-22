import numpy as np

from base.data.base_transformer import BaseTransformer


class ImageReshape(BaseTransformer):
    def __init__(self, permute_indices, **kwargs):
        super(ImageReshape, self).__init__(**kwargs)
        self.indices = permute_indices

    def forward(self, images):
        images = np.array(images).transpose(self.indices)
        return images
