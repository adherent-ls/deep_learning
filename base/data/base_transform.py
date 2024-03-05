class BaseTransform(object):
    def __init__(self):
        super().__init__()

    def __call__(self, *data):
        return self.forward(*data)

    def forward(self, *data):
        raise NotImplemented


class ImageTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, image, label):
        return self.forward(image), label

    def forward(self, image):
        raise NotImplemented


class LabelTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def __call__(self, image, label):
        return image, self.forward(label)

    def forward(self, label):
        raise NotImplemented
