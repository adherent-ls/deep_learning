from base.data.base_transformer import BaseTransformer


class TextFlatten(BaseTransformer):
    def __init__(self, **kwargs):
        super(TextFlatten, self).__init__(**kwargs)

    def forward(self, labels):  # (b, l1 or l2 or l3..., n)
        lengths = []

        texts = []
        for item in labels:
            length = len(item)
            lengths.append(length)
            texts.extend(item)

        return texts, lengths
