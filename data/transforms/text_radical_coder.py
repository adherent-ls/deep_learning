import json

from base.data.base_transformer import BaseTransformer


class RadicalCoder(BaseTransformer):
    def __init__(self, maps_path, characters, **kwargs):
        super(RadicalCoder, self).__init__(**kwargs)
        self.maps = json.load(open(maps_path, encoding='utf-8'))
        labels = {}
        self.characters = ['blank', 'start', 'end']
        self.characters.extend(characters)
        for i, line in enumerate(self.characters):
            labels[line] = i
        self.character_map = labels

    def forward(self, labels):
        new_labels = []
        for i, label in enumerate(labels):
            label = self.once_encode(label)
            new_labels.append(label)
        return new_labels

    def once_encode(self, text):
        label = []
        for char_item in text:
            radicals = self.maps[char_item]
            label.append(self.character_map['start'])
            for radical in radicals:
                index = self.character_map[radical]
                label.append(index)
            label.append(self.character_map['end'])  # 添加字与字之间的间隔符号
        return label
