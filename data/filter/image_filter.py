import numpy as np
import six
from PIL import Image

from base.data.base_filter import ImageFilter


class LmdbImageFilter(ImageFilter):
    def __init__(self):
        super(LmdbImageFilter, self).__init__()

    def forward(self, imgbuf):
        is_valid = False
        if imgbuf is None:
            return is_valid
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        try:
            image = Image.open(buf).convert('RGB')  # for color image
            image = np.array(image)
            buf.close()
        except IOError:
            buf.close()
            return is_valid

        if image is None:
            return is_valid
        h, w, c = image.shape
        if w <= h / 4:
            return is_valid

        is_valid = True
        return is_valid
