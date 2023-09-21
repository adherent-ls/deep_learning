import math

from torch import nn

from base.model.base_model import BaseModel


class RadicalTransformer(BaseModel):
    def __init__(self, in_channel, word_length=50, query_num=357, num_head=32, attn_drop=0.0, prop_drop=0.0, **kwargs):
        super(RadicalTransformer, self).__init__(**kwargs)
        self.num_head = num_head
        sim_channel = in_channel // num_head

        self.scale = 1 / math.sqrt(sim_channel)

        self.query_embedding = nn.Embedding(query_num, sim_channel)

        self.q = nn.Linear(sim_channel, sim_channel)
        self.k = nn.Linear(sim_channel, sim_channel)
        self.v = nn.Linear(sim_channel, sim_channel)
        self.softmax = nn.Softmax(-1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pred = nn.Linear(sim_channel, query_num)
        self.prop_drop = nn.Dropout(prop_drop)

        self.bn = nn.BatchNorm1d(word_length)

    def build_mask(self, x):
        query_embedding = self.query_embedding.weight[None, None]
        q = self.q(x)
        k = self.k(query_embedding)
        v = self.v(query_embedding)
        prop = q @ k.permute(0, 1, 3, 2)
        prop = prop * self.scale
        attention = self.softmax(prop)
        attention = self.attn_drop(attention)

        max_sim, max_index = attention.max(-1)

        out = (attention @ v) * max_sim[:, :, :, None]

        return out, prop

    def forward(self, x):
        b, l, c = x.shape
        r_x = x.reshape(b, l, self.num_head, c // self.num_head)
        out, prop = self.build_mask(r_x)
        y = out.reshape(b, l, -1)
        out = self.bn(y + x)
        if self.training:
            prop = self.prop_drop(prop)
            return out, prop
        else:
            return out, None
