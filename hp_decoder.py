import torch
import torch.nn as nn
from hp_layer import HPLinear, HPConfig
from hp_attention import HPMultiHeadAttention

class HPTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, dropout=0.1, cfg: HPConfig = None):
        super().__init__()
        self.self_attn = HPMultiHeadAttention(embed_dim, num_heads, dropout=dropout, cfg=cfg)
        self.cross_attn = HPMultiHeadAttention(embed_dim, num_heads, dropout=dropout, cfg=cfg)
        self.linear1 = HPLinear(embed_dim, dim_feedforward, cfg=cfg)
        self.linear_gate = HPLinear(embed_dim, dim_feedforward, cfg=cfg)
        self.linear2 = HPLinear(dim_feedforward, embed_dim, cfg=cfg)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU() # Защита от "зануления" градиентов ReLU

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, layer_past=None):
        # 1. Self-Attention
        norm_tgt = self.norm1(tgt)
        if layer_past is not None:
            self_attn_out, next_kv = self.self_attn.forward_kv(norm_tgt, layer_past=layer_past, attn_mask=tgt_mask)
        else:
            self_attn_out = self.self_attn(norm_tgt, norm_tgt, norm_tgt, attn_mask=tgt_mask)
            next_kv = None
        tgt = tgt + self.dropout1(self_attn_out)

        # 2. Cross-Attention
        norm_tgt = self.norm2(tgt)
        cross_attn_out = self.cross_attn(norm_tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(cross_attn_out)

        # 3. Gated Feed-Forward (GeGLU)
        norm_tgt = self.norm3(tgt)
        tgt_shape = norm_tgt.shape
        flat_tgt = norm_tgt.reshape(-1, tgt_shape[-1])

        # Механизм вентиляции
        gate = self.activation(self.linear1(flat_tgt))
        ff_hidden = gate * self.linear_gate(flat_tgt)
        ff_out = self.linear2(ff_hidden).reshape(tgt_shape)

        tgt = tgt + self.dropout3(ff_out)

        if layer_past is not None:
            return tgt, next_kv
        return tgt
