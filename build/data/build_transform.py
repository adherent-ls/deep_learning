class BuildTransform(object):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, data):
        y = data
        for item in self.transforms:
            y = item(*y)
        return y
