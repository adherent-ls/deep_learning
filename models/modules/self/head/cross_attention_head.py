import numpy as np
from torch import nn

from base.model.base_model import BaseModel


class CrossAttentionHead(BaseModel):
    def __init__(self, max_length, **kwargs):
        super(CrossAttentionHead, self).__init__(**kwargs)
        self.scale = 1 / np.sqrt(max_length)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_embedding, text_embedding):
#         b, l, c = image_embedding.shape
#         image_embedding = image_embedding.reshape(b * l, c)
#         text_embedding = text_embedding.reshape(b * l, c)
        cross_attention = image_embedding @ text_embedding.permute(0, 2, 1)
        cross_attention = cross_attention * self.scale
        cross_attention = self.softmax(cross_attention)
        return cross_attention
