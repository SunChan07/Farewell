import torch

def generate_causal_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask

def generate_padding_mask(seq, pad_id=0):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

def generate_combined_mask(tgt_seq, pad_id=0):
    sz = tgt_seq.size(1)
    device = tgt_seq.device
    
    causal = torch.triu(torch.ones(sz, sz, device=device), diagonal=1) == 1
    
    pad_mask = tgt_seq == pad_id
    pad_mask_expanded = pad_mask.unsqueeze(1).expand(-1, sz, -1)
    
    combined = causal.unsqueeze(0) | pad_mask_expanded
    
    mask = torch.zeros_like(combined, dtype=torch.float32)
    mask = mask.masked_fill(combined, float("-inf"))
    return mask
