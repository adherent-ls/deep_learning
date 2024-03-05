import torch

from base.data.base_transform import BaseTransform


class NPToTensor(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, image, label):
        image = torch.Tensor(image).float()
        text = torch.Tensor(label[0]).long()
        length = torch.Tensor(label[1]).long()
        return image, [text, length]
