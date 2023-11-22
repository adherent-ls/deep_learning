import numpy as np

from base.data.base_transformer import BaseTransformer


class TextAttnEndChar(BaseTransformer):
    def __init__(self, end='end', **kwargs):
        super(TextAttnEndChar, self).__init__(**kwargs)
        self.end = end

    def forward(self, labels):
        new_labels = []
        for i, label in enumerate(labels):
            label_index = self.once_encode(label)
            label_index.append(self.character_map['end'])
            labels[i] = np.array(label_index)
            new_labels.append(label + ['end'])
        return labels, new_labels
