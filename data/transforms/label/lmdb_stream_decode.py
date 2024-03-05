import json

import numpy as np

from base.data.base_transform import LabelTransform


class LmdbStreamDecode(LabelTransform):
    def __init__(self, encoding_type='utf-8'):
        super(LmdbStreamDecode, self).__init__()
        self.encoding_type = encoding_type

    def forward(self, label):
        label = label.decode(self.encoding_type)
        return label
