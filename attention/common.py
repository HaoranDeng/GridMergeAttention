import torch
import torch.nn.functional as F


def video_to_blocks(x, height, width):
    """[B, F*H*W, heads, dim] -> [B, outer, row, col, heads, dim]."""
    batch, seq_len, heads, dim = x.shape
    frames = seq_len // (height * width)
    return x.reshape(batch, frames, height, width, heads, dim)


def blocks_to_video(x):
    """[B, outer, row, col, heads, dim] -> [B, F*H*W, heads, dim]."""
    batch, outer, row, col, heads, dim = x.shape
    return x.reshape(batch, outer * row * col, heads, dim)


def local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows):
    """Local-with-ith seed rows for hard C4 blocks.

    For key row k, seed 0 is query row k. The other three seeds are the nearest
    query rows around k, matching the optimized Triton hard_c4 path.
    """
    device = query_outer.device
    key_row = torch.arange(key_rows, device=device).view(1, 1, key_rows, 1)
    query_col = torch.arange(query_columns, device=device).view(1, 1, 1, query_columns)
    ith_seed = key_row.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns)

    candidate = torch.arange(query_rows, device=device).view(1, 1, 1, 1, query_rows)
    distance = (candidate - ith_seed.unsqueeze(-1)).abs()
    score = -distance * (query_rows + 1) - candidate
    score = score.masked_fill(candidate == ith_seed.unsqueeze(-1), torch.iinfo(score.dtype).min)
    extras = score.topk(3, dim=-1).indices.long()
    return torch.cat((ith_seed.unsqueeze(-1).long(), extras), dim=-1)


def gather_seed_queries(q, seed_rows):
    # q: [B, A, I, J, H, D], seed_rows: [A, F, K, J, 4].
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(q.size(3), device=q.device).view(1, 1, 1, -1, 1)
    return q[:, query_outer, seed_rows, query_col]


def biased_sdpa(query, key, value, bias, scale=1.0):
    """Attention over compressed tokens with a per-key additive bias.

    query: [G, Q, D], key/value: [G, K, D], bias: [G, K].
    The attention logits are scale * query @ key.T - bias.
    """
    query_for_attn = query.to(key.dtype) if query.dtype != key.dtype else query
    out = F.scaled_dot_product_attention(
        query_for_attn.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        attn_mask=(-bias.float()).to(query_for_attn.dtype).unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(1).to(query.dtype)
