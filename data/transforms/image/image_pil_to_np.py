import numpy as np
from PIL import Image

from base.data.base_transform import ImageTransform


class ImagePilToNP(ImageTransform):

    def __init__(self):
        super().__init__()

    def forward(self, image: Image):
        return np.array(image)
