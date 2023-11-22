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

    def forward(self, label):
        label = self.once_encode(label)
        return label

    def once_encode(self, text):
        label = []
        for char_item in text:
            index = self.character_map[char_item]
            label.append(index)
        return label

    def decode(self, index):
        texts = []
        for item in index:
            text = ''
            for single in item:
                chars = self.characters[int(single)]
                if chars == self.characters[-1]:  # 结束符号
                    break
                if chars != self.characters[0]:  # 开始/空白符号
                    text += self.characters[int(single)]
            texts.append(text)
        return texts
