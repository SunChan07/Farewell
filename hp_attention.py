import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, query, key, value, attn_mask=None):
        # Этот метод используется только во время ОБУЧЕНИЯ
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        q = self.q_proj(query.reshape(-1, embed_dim)).reshape(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = self.k_proj(key.reshape(-1, embed_dim)).reshape(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = self.v_proj(value.reshape(-1, embed_dim)).reshape(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        scaling = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling

        # Бродкастинг маски
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

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).transpose(0, 1).contiguous().reshape(tgt_len, bsz, embed_dim)
        return self.out_proj(attn_output.reshape(-1, embed_dim)).reshape(tgt_len, bsz, embed_dim)

    def forward_kv(self, query, layer_past=None, attn_mask=None):
        tgt_len, bsz, embed_dim = query.size()

        q_flat = query.transpose(0, 1).reshape(-1, embed_dim)

        q = self.q_proj(q_flat).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(q_flat).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(q_flat).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present_kv = (k, v)

        scaling = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
            attn_weights = attn_weights + attn_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_probs), v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz * tgt_len, embed_dim)
        output = self.out_proj(attn_output).view(bsz, tgt_len, embed_dim).transpose(0, 1)

        return output, present_kv