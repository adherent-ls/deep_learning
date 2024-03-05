import numpy as np

from base.data.base_transform import ImageTransform


class ImageCollate(ImageTransform):
    def __init__(self):
        super(ImageCollate, self).__init__()

    def forward(self, images):
        max_w, max_h, max_c = 0, 0, 0
        for image in images:
            h, w, c = image.shape
            max_w = max(max_w, w)
            max_h = max(max_h, h)
            max_c = max(max_c, c)

        b = len(images)

        pad_images = np.zeros((b, max_h, max_w, max_c))
        for i, image in enumerate(images):
            h, w, c = image.shape
            pad_images[i, :h, :w, :c] = image
        return pad_images
