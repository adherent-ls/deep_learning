from base.model.base_model import BaseModel
from models.modules.base.network.vit_block import SelfAttention


class ViT(BaseModel):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., **kwargs):
        super(ViT, self).__init__(**kwargs)
        self.attn = SelfAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

