
import numpy as np
import six
from PIL import Image

from base.data.base_filter import BaseFilter


class LabelVocabCheckFilter(BaseFilter):
    def __init__(self, label_length_limit=None, characters=None, **kwargs):
        super(LabelVocabCheckFilter, self).__init__(**kwargs)
        self.label_length_limit = label_length_limit
        self.characters = characters if isinstance(characters, list) else eval(characters)

    def __call__(self, image, label):
        is_valid = False
        if label is None:
            return is_valid
        if self.label_length_limit is not None:
            if len(label) >= self.label_length_limit:
                return is_valid
        if self.label_length_limit is not None:
            out_vocab = False
            for item in label:
                if item not in self.characters:
                    out_vocab = True
                    break
            if out_vocab:
                return is_valid
        is_valid = True
        return is_valid


class BoxCheckFilter(BaseFilter):
    def __init__(self, length_limit=None, **kwargs):
        super(BoxCheckFilter, self).__init__(**kwargs)
        self.length_limit = length_limit

    def __call__(self, image, label):
        is_valid = False
        if label is None:
            return is_valid
        if self.length_limit is not None:
            if len(label) >= self.length_limit:
                return is_valid
        is_valid = True
        return is_valid


class ImageCheckFilter(BaseFilter):
    def __init__(self, **kwargs):
        super(ImageCheckFilter, self).__init__(**kwargs)

    def __call__(self, imgbuf, label):
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

        l = len(label)

        if l < w / h * 2 / 3:
            return is_valid  # 图像中不含文字区域过多

        is_valid = True
        return is_valid
