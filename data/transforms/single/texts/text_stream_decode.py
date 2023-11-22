import json

import numpy as np

from base.data.base_transformer import BaseTransformer


class TextStreamDecode(BaseTransformer):
    def __init__(self):
        super(TextStreamDecode, self).__init__()

    def forward(self, label):
        label = label.decode('utf-8')
        label = list(label) + ['end']
        return label
