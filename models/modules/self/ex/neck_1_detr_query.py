import torch
from torch import nn

from base.model.base_model import BaseModel
from models.modules.base.network.patch_embedding import get_sinusoid_encoding_table
from models.modules.base.network.vit_block import CrossAttention


class DETRQuery(BaseModel):
    def __init__(self, max_query_num, query_dim, max_w=10000, **kwargs):
        super(DETRQuery, self).__init__(**kwargs)
        self.query_embedding = nn.Embedding(max_query_num, query_dim)

        self.attention_layers = CrossAttention(query_dim)

        self.k_linear = nn.Linear(query_dim, query_dim)

        self.query_map = nn.Linear(query_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.feed_word = nn.Linear(query_dim, query_dim)
        self.lamuda = nn.Parameter(torch.FloatTensor((0.7,)))

        self.table = get_sinusoid_encoding_table(max_w, query_dim)

    def build_mask(self, feature):
        b, w, c = feature.shape

        query_embedding = self.query_embedding.weight[None]

        key = feature + self.table[:, :w].to(feature.device)
        value = self.attention_layers(query_embedding, key, key)

        return value

    def forward(self, x):
        mask = self.build_mask(x)
        out = mask @ x
        feedward = self.feed_word(out)
        return feedward
