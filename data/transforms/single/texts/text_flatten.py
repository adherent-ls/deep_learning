from base.data.base_transformer import BaseTransformer


class TextFlatten(BaseTransformer):
    def __init__(self, **kwargs):
        super(TextFlatten, self).__init__(**kwargs)

    def forward(self, text):  # (b, l1 or l2 or l3..., n)
        length = len(text)
        return text, length
