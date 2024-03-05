class BaseFilter(object):
    pass


class ImageFilter(BaseFilter):
    def __init__(self):
        super().__init__()

    def __call__(self, image, label):
        return self.forward(image), label

    def forward(self, image):
        raise NotImplemented


class LabelFilter(BaseFilter):
    def __init__(self):
        super().__init__()

    def __call__(self, image, label):
        return image, self.forward(label)

    def forward(self, label):
        raise NotImplemented
