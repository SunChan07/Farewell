import math
import torch
import torch.nn as nn
from hp_layer import HPLinear, HPConfig

class HPMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, cfg: HPConfig = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.q_proj = HPLinear(embed_dim, embed_dim, cfg=cfg)
        self.k_proj = HPLinear(embed_dim, embed_dim, cfg=cfg)
        self.v_proj = HPLinear(embed_dim, embed_dim, cfg=cfg)
        self.out_proj = HPLinear(embed_dim, embed_dim, cfg=cfg)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        q = self.q_proj(query.reshape(-1, embed_dim)).reshape(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = self.k_proj(key.reshape(-1, embed_dim)).reshape(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = self.v_proj(value.reshape(-1, embed_dim)).reshape(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        
        scaling = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
            else:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                attn_weights = attn_weights + attn_mask
            
        attn_probs = self.softmax(attn_weights)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).transpose(0, 1).contiguous().reshape(tgt_len, bsz, embed_dim)
        output = self.out_proj(attn_output.reshape(-1, embed_dim)).reshape(tgt_len, bsz, embed_dim)
        return output
