import six
from PIL import Image

from base.data.base_transform import ImageTransform


class LmdbImageDecode(ImageTransform):
    def __init__(self):
        super().__init__()

    def forward(self, image_buffer):
        buf = six.BytesIO()
        buf.write(image_buffer)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        return image
