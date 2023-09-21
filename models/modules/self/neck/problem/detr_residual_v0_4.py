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


class DETRResidualV04(BaseModel):
    def __init__(self, max_query_num, query_dim, layer_num=4, max_w=1000, **kwargs):
        super(DETRResidualV04, self).__init__(**kwargs)
        self.layer_num = layer_num

        self.query_embedding = nn.Embedding(max_query_num, query_dim)
        self.cross_attention = CrossAttention(query_dim)
        self.query_map = nn.Linear(query_dim, 1)

        self.word_map = nn.Linear(query_dim, 1)
        self.word_restruct_map = nn.Linear(query_dim, 1)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

        self.feed_word = nn.Linear(query_dim, query_dim)

        self.table = get_sinusoid_encoding_table(max_w, query_dim)

    def build_mask(self, feature):
        b, w, c = feature.shape
        key = feature + self.table[:, :w].to(feature.device)
        query_w = self.query_embedding.weight[None]

        value = self.cross_attention(query_w, key, key)
        query = self.query_map(value)
        # query = torch.cumsum(self.query_embedding[None, :, None], 1)

        key_map = self.word_map(feature).permute(0, 2, 1)  # (b,w,c)->(b,w,1)->(b,1,w)
        key_map = self.sigmoid(key_map)

        length = torch.arange(0, w).to(feature.device)[None, None, :]  # w

        for i in range(self.layer_num):
            query_map = torch.exp(-0.5 * 2 * math.log(2) * (length - query) ** 2)

            # 计算两者的近余弦距离，不进行归一化，一个多峰分布，一个单峰分布，当一个单峰分布接近多峰分布中的一个峰时较大
            cosine = query_map * key_map  # (1,q,w)*(b,1,w)=(b,q,w)
            cosine = cosine / ((torch.sum(cosine, dim=-1)[:, :, None] + 1e-6))

            key = cosine @ feature  # (b,q,w) @ (b,w,c) -> (b,q,c)
            restruct_map = self.word_restruct_map(key)  # (b,q,c)->(b,q,1)

            # restruct_map = self.sigmoid(restruct_map)
            # restruct_query = torch.cat([query[:, :1], query[:, 1:] - query[:, :-1]], dim=1)
            # restruct_query = restruct_query + restruct_map
            # query = torch.cumsum(restruct_query, 1)
            query = query + restruct_map

        query_map = torch.exp(-0.5 * 2 * math.log(2) * (length - query) ** 2)
        query_map = query_map / (torch.sum(query_map, dim=-1)[:, :, None] + 1e-6)
        return query_map

    def forward(self, x):
        mask = self.build_mask(x)
        out = mask @ x
        feedward = self.feed_word(out)
        return feedward


if __name__ == '__main__':
    m = DETRResidualV03(10, 32)

    x = torch.rand(2, 18, 32)

    y = m.forward(x)
    print(y.shape)
