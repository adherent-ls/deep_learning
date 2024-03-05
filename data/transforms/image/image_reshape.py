import numpy as np

from base.data.base_transform import ImageTransform


class ImageReshape(ImageTransform):
    def __init__(self, permute_indices):
        super(ImageReshape, self).__init__()
        self.indices = permute_indices

    def forward(self, images):
        images = images.transpose(self.indices)
        return images
