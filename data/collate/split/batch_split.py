from base.data.base_transform import BaseTransform


class BatchSplit(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, *datas):
        images = []
        labels = []
        for item in datas:
            if item is None:
                continue
            if len(item) == 2:
                image = item[0]
                label = item[1]
            else:
                image = item[0]
                label = item[1:]
            images.append(image)
            labels.append(label)
        return images, labels
