import torch
import torch.nn.functional as F


def validate_block_seed_mode(seed_mode, context):
    if seed_mode != "local_with_ith":
        raise ValueError(f"{context} keeps only seed_mode='local_with_ith'.")
    return True


def video_to_blocks(x, f_tied, h_reduce, w_reduce, height, width):
    """[B, F*H*W, heads, dim] -> [B, outer, row, col, heads, dim]."""
    if (int(f_tied), int(h_reduce), int(w_reduce)) != (1, 1, 1):
        raise ValueError("hard C4 keeps only f_tied=h_reduce=w_reduce=1.")
    batch, seq_len, heads, dim = x.shape
    if seq_len % (height * width):
        raise ValueError("sequence length must be a multiple of height * width.")
    frames = seq_len // (height * width)
    return x.reshape(batch, frames, height, width, heads, dim)


def blocks_to_video(x, f_tied, h_reduce, w_reduce):
    """[B, outer, row, col, heads, dim] -> [B, F*H*W, heads, dim]."""
    if (int(f_tied), int(h_reduce), int(w_reduce)) != (1, 1, 1):
        raise ValueError("hard C4 keeps only f_tied=h_reduce=w_reduce=1.")
    batch, outer, row, col, heads, dim = x.shape
    return x.reshape(batch, outer * row * col, heads, dim)


def local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows):
    """Local-with-ith seed rows for hard C4 blocks.

    For key row k, seed 0 is query row k. The other three seeds are the nearest
    query rows around k, matching the optimized Triton hard_c4 path.
    """
    if query_rows < 4:
        raise ValueError("hard C4 requires at least four query rows.")

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


def seed_rows_by_mode(
    query_outer,
    key_outer,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
    seed_mode,
):
    del seed
    validate_block_seed_mode(seed_mode, "hard C4")
    if int(num_seeds) != 4:
        raise ValueError("hard C4 keeps only num_merge_tokens=4.")
    return local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows)


def gather_seed_queries(q, seed_rows):
    # q: [B, A, I, J, H, D], seed_rows: [A, F, K, J, 4].
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(q.size(3), device=q.device).view(1, 1, 1, -1, 1)
    return q[:, query_outer, seed_rows, query_col]


def pad_video_tokens_to_multiples(x, height, width, h_reduce, w_reduce):
    """Pad [B, F*H*W, heads, dim] so H/W are divisible by reduce factors."""
    height = int(height)
    width = int(width)
    h_reduce = int(h_reduce)
    w_reduce = int(w_reduce)
    batch, seq_len, heads, dim = x.shape
    if seq_len % (height * width):
        raise ValueError("sequence length must be a multiple of height * width.")

    frames = seq_len // (height * width)
    padded_height = ((height + h_reduce - 1) // h_reduce) * h_reduce
    padded_width = ((width + w_reduce - 1) // w_reduce) * w_reduce
    if padded_height == height and padded_width == width:
        return x, height, width, None

    padded = x.new_zeros(batch, frames, padded_height, padded_width, heads, dim)
    padded[:, :, :height, :width] = x.reshape(batch, frames, height, width, heads, dim)
    return padded.reshape(batch, frames * padded_height * padded_width, heads, dim), padded_height, padded_width, None


def crop_video_tokens_from_multiples(x, height, width, padded_height, padded_width):
    """Crop padded [B, F*Hp*Wp, heads, dim] output back to [B, F*H*W, heads, dim]."""
    height = int(height)
    width = int(width)
    padded_height = int(padded_height)
    padded_width = int(padded_width)
    if padded_height == height and padded_width == width:
        return x

    batch, seq_len, heads, dim = x.shape
    if seq_len % (padded_height * padded_width):
        raise ValueError("sequence length must be a multiple of padded_height * padded_width.")

    frames = seq_len // (padded_height * padded_width)
    x = x.reshape(batch, frames, padded_height, padded_width, heads, dim)
    return x[:, :, :height, :width].contiguous().reshape(batch, frames * height * width, heads, dim)


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
