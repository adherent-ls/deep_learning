import numpy as np

from base.data.base_transform import LabelTransform


class TextEncode(LabelTransform):
    def __init__(self, characters):
        super(TextEncode, self).__init__()
        labels = {}
        self.characters = characters
        for i, line in enumerate(self.characters):
            labels[line] = i
        self.character_map = labels

    def forward(self, label):
        label = self.once_encode(label)
        return label

    def once_encode(self, text):
        label = []
        for char_item in text:
            index = self.character_map[char_item]
            label.append(index)
        return label
