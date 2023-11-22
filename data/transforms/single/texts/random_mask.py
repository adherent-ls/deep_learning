import numpy as np
import torch
from base.data.base_transformer import BaseTransformer


class RandomMaskingGenerator(BaseTransformer):
    def __init__(self, input_size, mask_ratio, **kwargs):
        super(RandomMaskingGenerator, self).__init__(**kwargs)
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def forward(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196]


if __name__ == '__main__':
    t = RandomMaskingGenerator(16, 1)
    data = t({})['mask']
    x = np.random.random((1, 256, 10))
    x = torch.Tensor(x)
    data = torch.Tensor(data).long()
    y = x * data[None, :, None]
    print(y)
