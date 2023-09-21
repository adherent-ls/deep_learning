import numpy as np

from base.data.base_transformer import BaseTransformer


class ImageCollate(BaseTransformer):
    def __init__(self, padding=0, img_w_cell=16, **kwargs):
        super(ImageCollate, self).__init__(**kwargs)
        self.padding = padding
        self.img_w_cell = img_w_cell

    def forward(self, images):
        max_w, max_h, max_c = 0, 0, 0
        for image in images:
            h, w, c = image.shape
            max_w = max(max_w, w)
            max_h = max(max_h, h)
            max_c = max(max_c, c)
        b = len(images)
        max_w = ((max_w + self.padding) // self.img_w_cell + 1) * self.img_w_cell
        pad_images = np.zeros((b, max_h, max_w, max_c))
        for i, image in enumerate(images):
            h, w, c = image.shape
            pad_images[i, :h, :w, :c] = image
        return pad_images


class RadicalAttnCollate(BaseTransformer):
    def __init__(self, max_length, **kwargs):
        super(RadicalAttnCollate, self).__init__(**kwargs)
        self.max_length = max_length

    def forward(self, radicals, labels):
        max_l = 0
        for label in radicals:
            length = len(label)
            max_l = max(max_l, length)
        b, n = labels.shape[:2]
        new_labels = np.zeros((b, n, self.max_length))
        for i, label in enumerate(radicals):
            for j, item in enumerate(label):
                length = len(item)
                new_labels[i, j, :length] = item
        return new_labels


class RadicalCollate(BaseTransformer):
    def __init__(self, max_length, **kwargs):
        super(RadicalCollate, self).__init__(**kwargs)
        self.max_length = max_length

    def forward(self, radicals):
        b = len(radicals)
        new_labels = np.zeros((b, self.max_length))
        for i, label in enumerate(radicals):
            l = len(label)
            new_labels[i, :l] = np.array(label)
        return new_labels


class RadicalMapCollate(BaseTransformer):
    def __init__(self, max_length, word_length, **kwargs):
        super(RadicalMapCollate, self).__init__(**kwargs)
        self.max_length = max_length
        self.word_length = word_length

    def forward(self, radicals):
        b = len(radicals)
        new_labels = np.zeros((b, self.word_length, self.max_length))
        for i, label in enumerate(radicals):
            for j, radical_items in enumerate(label):
                r = len(radical_items)
                new_labels[i, j, :r] = np.array(radical_items)
        return new_labels


class LabelAttnCollate(BaseTransformer):
    def __init__(self, max_length, **kwargs):
        super(LabelAttnCollate, self).__init__(**kwargs)
        self.max_length = max_length

    def forward(self, labels):
        b = len(labels)
        new_labels = np.zeros((b, self.max_length))
        for i, label in enumerate(labels):
            length = len(label)
            new_labels[i, :length] = label
            new_labels[i, length] = 1
        return new_labels

    def decode(self, index):
        return index


class LabelCTCCollate(BaseTransformer):
    def __init__(self, max_length, **kwargs):
        super(LabelCTCCollate, self).__init__(**kwargs)
        self.max_length = max_length

    def forward(self, labels):
        b = len(labels)
        new_labels = np.zeros((b, self.max_length))
        lengths = np.zeros((b,))
        for i, label in enumerate(labels):
            length = len(label)
            new_labels[i, :length] = label
            lengths[i] = length
        return new_labels, lengths

    def decode(self, index):
        if isinstance(index, tuple):
            index = index[0]
        return index
