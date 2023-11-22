import cv2

from base.data.base_transformer import BaseTransformer


class ColorConvert(BaseTransformer):
    def __init__(self, mole=cv2.COLOR_RGB2GRAY, **kwargs):
        super(ColorConvert, self).__init__(**kwargs)
        self.mole = mole

    def forward(self, image):
        image = cv2.cvtColor(image, self.mole)
        if len(image.shape) == 2:
            image = image[..., None]
        return image
