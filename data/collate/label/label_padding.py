import numpy as np

from base.data.base_transform import LabelTransform


class LabelCollate(LabelTransform):
    def __init__(self, max_length):
        super(LabelCollate, self).__init__()
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
