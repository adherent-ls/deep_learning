import numpy as np

import json

from base.data.base_transformer import BaseTransformer


class RadicalMapCoder(object):
    def __init__(self, maps_path, characters, **kwargs):
        super(RadicalMapCoder, self).__init__(**kwargs)
        self.maps = json.load(open(maps_path, encoding='utf-8'))
        labels = {}
        self.characters = characters
        for i, line in enumerate(self.characters):
            labels[line] = i
        self.character_map = labels

    def __call__(self, labels):
        label = self.once_encode(labels)
        return label

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


class RadicalMapCollate(object):
    def __init__(self, max_length, **kwargs):
        super(RadicalMapCollate, self).__init__(**kwargs)
        self.max_length = max_length

    def __call__(self, radicals):
        b = len(radicals)
        new_labels = np.zeros((b, self.max_length))
        for i, label in enumerate(radicals):
            r = len(label)
            new_labels[i, :r] = np.array(label)
        return new_labels


class TextMap(BaseTransformer):
    def __init__(self, characters, maps_path, radical_characters, max_length, **kwargs):
        super(TextMap, self).__init__(**kwargs)
        self.radical_map = RadicalMapCoder(maps_path=maps_path, characters=radical_characters)
        self.map_collate = RadicalMapCollate(max_length)

        self.characters = characters

        self.result = None
        self.old_text = []

    def forward(self, text):
        if self.result is None:
            radicals = self.radical_map(self.characters)  # 1 * n * l
            radical_collate = self.map_collate(radicals)
            self.result = radical_collate
        new_texts = []
        new_text_maps = {}
        new_radicals = []
        self.old_text = []
        for item in text:
            text_items = []
            for char_item in item:
                char_item = int(char_item)
                if char_item not in new_text_maps:
                    new_text_maps[char_item] = len(new_text_maps)
                    new_radicals.append(self.result[char_item])
                    self.old_text.append(char_item)
                text_items.append(new_text_maps[char_item])
            new_texts.append(text_items)
        return new_radicals, new_texts

    def decode(self, index):
        texts = []
        for item in index:
            items = []
            for char_item in item:
                char_str = self.old_text[char_item]
                items.append(char_str)
            texts.append(items)
        return texts


if __name__ == '__main__':
    from data.transforms.text_label_coder import LabelCoder

    radical_characters = ['blank'] + [item.strip("\n") for item in
                                      open(r"D:\Code\DeepLearning\self\deep_learning_restruct\vocab\radical\usual.txt",
                                           "r", encoding="utf-8").readlines()] + ['end']
    characters = ['blank'] + [item.strip("\n") for item in
                              open(r"D:\Code\DeepLearning\self\deep_learning_restruct\vocab\radical\word.txt", "r",
                                   encoding="utf-8").readlines()] + ['end']
    maps_path = r'D:\Code\DeepLearning\self\deep_learning_restruct\vocab\radical\usual_map.json'
    in_text = ['浙象渔', '玉渔运']
    label_coder = LabelCoder(characters, in_key='text', out_key='text')
    in_text_index = label_coder({'text': in_text})['text']
    t = TextMap(characters, maps_path, radical_characters, 32, in_key='text', out_key=['text_map', 'text'])

    result = t({'text': in_text})
    print(t.decode(result['text']))
    print(result)
