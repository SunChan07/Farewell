import torch
import torch.nn as nn
from hp_layer import HPLinear, HPConfig
from hp_attention import HPMultiHeadAttention

class HPTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=512, dropout=0.1, cfg: HPConfig = None):
        super().__init__()
        self.self_attn = HPMultiHeadAttention(embed_dim, num_heads, dropout=dropout, cfg=cfg)
        self.cross_attn = HPMultiHeadAttention(embed_dim, num_heads, dropout=dropout, cfg=cfg)
        self.linear1 = HPLinear(embed_dim, dim_feedforward, cfg=cfg)
        self.linear2 = HPLinear(dim_feedforward, embed_dim, cfg=cfg)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        self_attn_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))
        cross_attn_out = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))
        tgt_shape = tgt.shape
        ff_out = self.linear2(self.activation(self.linear1(tgt.reshape(-1, tgt_shape[-1])))).reshape(tgt_shape)
        tgt = self.norm3(tgt + self.dropout3(ff_out))
        return tgt
