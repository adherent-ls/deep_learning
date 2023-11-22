import numpy as np

from einops import rearrange

from base.data.base_transformer import BaseTransformer


class LabelResize(BaseTransformer):
    def __init__(self, patch_size=(640, 640), **kwargs):
        super(LabelResize, self).__init__(**kwargs)
        self.patch_size = patch_size

    def forward(self, images):
        images = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                           p1=self.patch_size[0],
                           p2=self.patch_size[1])
        return np.array(images)
