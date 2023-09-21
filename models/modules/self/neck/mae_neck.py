import torch
from torch import nn

from models.modules.base.network.patch_embedding import get_sinusoid_encoding_table
# from utils.truncated_normal import truncated_normal


class MAENeck(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, num_patches):
        super(MAENeck, self).__init__()
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)
        truncated_normal(self.mask_token, std=.02)

    def forward(self, x_vis, mask):
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        B, N, C = x_vis.shape

        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x_vis).to(x_vis.device).clone().detach()
        pos_emd_vis = (expand_pos_embed * mask[None, :, None]).reshape(B, -1, C)
        pos_emd_mask = (expand_pos_embed * mask[None, :, None]).reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        return x_full, pos_emd_mask.shape[1]
