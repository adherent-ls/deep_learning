import json

import numpy as np
import six
from PIL import Image

from base.data.base_transformer import BaseTransformer


class ImageBufferDecode(BaseTransformer):
    def __init__(self):
        super(ImageBufferDecode, self).__init__()

    def forward(self, imgbuf):
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')  # for color image
        image = np.array(image)
        buf.close()
        return image
