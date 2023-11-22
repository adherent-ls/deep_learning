import numpy as np

from base.data.base_transformer import BaseTransformer


class TextMask(BaseTransformer):
    def __init__(self, max_length, **kwargs):
        super(TextMask, self).__init__(**kwargs)
        self.max_length = max_length

    def forward(self, labels):
        b = len(labels)
        mask = np.zeros((b, self.max_length, self.max_length))
        labels = np.array(labels)
        for i, item in enumerate(labels):
            l = np.sum(item == 0)
            mask[i, :l, :l] = 1
            for j, sub in enumerate(item):  
                for k, sub1 in enumerate(item):
                    if j == k:
                        continue
                    if sub == sub1:  # 相同的标签不进行比较
                        mask[i, j, k] = 0
                        mask[i, k, j] = 0
        return mask


if __name__ == '__main__':
    text_mask = TextMask(32)
    labels = np.zeros((4, 32))
    for i in range(4):
        v = np.random.randint(5, 32, ())
        t = np.random.randint(0, 5, (v,))
        labels[i, :v] = t
    y = text_mask({'images': labels})
    print(y)
