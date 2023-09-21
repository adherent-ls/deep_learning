import math

import torch
from torch import nn

from base.model.base_model import BaseModel
from models.modules.base.network.patch_embedding import get_sinusoid_encoding_table
from models.modules.base.network.vit_block import CrossAttention


def gen_sine_embed_for_position(pos_tensor, dim):
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    embed = pos_tensor * scale
    pos = embed[:, None] / dim_t
    pos = torch.stack((pos[:, 0::2].sin(), pos[:, 1::2].cos()), dim=1).flatten(1)
    return pos


class DETRResidual(BaseModel):
    def __init__(self, max_query_num, query_dim, layer_num=4, max_w=10000, **kwargs):
        super(DETRResidual, self).__init__(**kwargs)
        self.query_embedding = nn.Parameter(torch.arange(0, max_query_num) / 50)

        self.attention_layers = nn.ModuleList()
        for i in range(layer_num):
            self.attention_layers.append(CrossAttention(query_dim))

        self.k_linear = nn.Linear(query_dim, query_dim)

        self.query_map = nn.Linear(query_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.feed_word = nn.Linear(query_dim, query_dim)
        self.lamuda = nn.Parameter(torch.rand(1, ))
        self.table = get_sinusoid_encoding_table(max_w, query_dim)

    def build_mask(self, feature):
        b, w, c = feature.shape

        query = self.query_embedding[None]

        for layer in self.attention_layers:
            query_embedding = gen_sine_embed_for_position(self.query_embedding, c)[None].to(feature.device)
            
            key = feature + self.table[:, :w].to(feature.device)

            value = layer(query_embedding, key, key)
            deta = self.query_map(value).squeeze(-1)
            query = query * self.lamuda + self.sigmoid(deta) * (1 - self.lamuda)

        center = w * query[:, :, None]  # (q_n,) -> (1,q_n,1)
        length = torch.arange(0, w)[None, None, :].repeat((b, 1, 1)).to(feature.device)  # w -> 1*1*w -> b*1*w

        v = -0.5 * 4 * math.log(2) * (length - center) ** 2  # b*q_n*w

        mask = torch.exp(v)
        mask = mask / (torch.sum(mask, dim=-1)[:, :, None] + 1e-6)
        return mask

    def forward(self, x):
        mask = self.build_mask(x)
        out = mask @ x
        feedward = self.feed_word(out)
        return feedward


if __name__ == '__main__':
    m = DETRResidual(10, 32)

    x = torch.rand(2, 18, 32)

    y = m.forward(x)
    print(y.shape)
