import numpy as np

from base.data.base_transformer import BaseTransformer


class LabelCoder(BaseTransformer):
    def __init__(self, characters, **kwargs):
        super(LabelCoder, self).__init__(**kwargs)
        characters = characters if isinstance(characters, list) else eval(characters)
        labels = {}
        self.characters = characters
        for i, line in enumerate(self.characters):
            labels[line] = i
        self.character_map = labels

    def forward(self, labels):
        for i, label in enumerate(labels):
            label = self.once_encode(label)
            labels[i] = np.array(label)
        return labels

    def once_encode(self, text):
        label = []
        for char_item in text:
            index = self.character_map[char_item]
            label.append(index)
        return label
