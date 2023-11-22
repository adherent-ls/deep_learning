import numpy as np

from base.data.base_transformer import BaseTransformer


class TextAttnEndChar(BaseTransformer):
    def __init__(self, end='end', **kwargs):
        super(TextAttnEndChar, self).__init__(**kwargs)
        self.end = end

    def forward(self, label):
        label_index = self.once_encode(label)
        label_index.append(self.character_map['end'])
        return label_index
