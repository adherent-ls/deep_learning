import json

from base.data.base_transformer import BaseTransformer


class RadicalMapCoder(BaseTransformer):
    def __init__(self, maps_path, characters, **kwargs):
        super(RadicalMapCoder, self).__init__(**kwargs)
        self.maps = json.load(open(maps_path, encoding='utf-8'))
        labels = {}
        self.characters = characters
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
            radical_items = []
            if char_item in self.maps:
                radicals = self.maps[char_item]
            elif char_item in self.character_map:
                radicals = [char_item]
            else:
                assert f'{char_item} not found in radical table'
                return

            for radical in radicals:
                index = self.character_map[radical]
                radical_items.append(index)
            radical_items.append(self.character_map[self.characters[-1]])  # 添加结束符号
            label.append(radical_items)
        return label
