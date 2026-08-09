import torch
import torch.nn.functional as F
from einops import rearrange


HASH_MOD = 2147483647
BLOCK_SEED_MODES = {
    "random",
    "random_with_ith",
    "fast_random_with_ith",
    "local_with_ith",
    "strided_with_ith",
}

_Q_INIT_ALIASES = {
    "identity": "ith",
    "i": "ith",
    "uniform": "mean",
    "avg": "mean",
    "average": "mean",
    "first": "1st",
}


def normalize_q_init(q_init):
    q_init = (q_init or "ith").lower()
    return _Q_INIT_ALIASES.get(q_init, q_init)


def scale_qk(q, k, scale):
    scale_sqrt = scale**0.5
    return q * scale_sqrt, k * scale_sqrt


def validate_block_seed_mode(seed_mode, context):
    if seed_mode not in BLOCK_SEED_MODES:
        raise ValueError(
            f"{context} currently supports seed_mode='random', "
            "'random_with_ith', 'fast_random_with_ith', "
            "'local_with_ith', or 'strided_with_ith'."
        )
    return seed_mode != "random"


def merge_query_row(q, key_rows, q_init):
    """Choose query rows used to merge each key row's columns."""
    q_init = normalize_q_init(q_init)

    if q_init == "ith":
        if q.size(2) != key_rows:
            raise ValueError("q_init=ith needs query_row and key_row to have the same size.")
        return q

    if q_init == "mean":
        return q.mean(dim=2, keepdim=True).expand(-1, -1, key_rows, -1, -1, -1)

    if q_init == "1st":
        return q[:, :, :1].expand(-1, -1, key_rows, -1, -1, -1)

    raise ValueError(f"unsupported q_init={q_init!r}; expected mean, 1st, or ith.")


def merge_query_column(q, key_columns, q_init):
    """Choose query columns used to merge each key column's rows."""
    q_init = normalize_q_init(q_init)

    if q_init == "ith":
        if q.size(3) != key_columns:
            raise ValueError("q_init=ith needs query_column and key_column to have the same size.")
        return q

    if q_init == "mean":
        return q.mean(dim=3, keepdim=True).expand(-1, -1, -1, key_columns, -1, -1)

    if q_init == "1st":
        return q[:, :, :, :1].expand(-1, -1, -1, key_columns, -1, -1)

    raise ValueError(f"unsupported q_init={q_init!r}; expected mean, 1st, or ith.")


def video_to_blocks(x, f_tied, h_reduce, w_reduce, height, width):
    """[B, F*H*W, heads, dim] -> [B, outer, row, col, heads, dim]."""
    batch, seq_len, heads, dim = x.shape
    if height % h_reduce or width % w_reduce:
        raise ValueError("height/width must be divisible by h_reduce/w_reduce.")
    if seq_len % (height * width):
        raise ValueError("sequence length must be a multiple of height * width.")

    frames = seq_len // (height * width)
    if frames % f_tied:
        raise ValueError("number of frames must be divisible by f_tied.")

    if f_tied == 1 and h_reduce == 1 and w_reduce == 1:
        return x.reshape(batch, frames, height, width, heads, dim)

    return rearrange(
        x,
        "b (outer ft hr row wr col) head dim -> b (outer hr wr) (ft row) col head dim",
        ft=f_tied,
        hr=h_reduce,
        row=height // h_reduce,
        wr=w_reduce,
        col=width // w_reduce,
    ).contiguous()


def blocks_to_video(x, f_tied, h_reduce, w_reduce):
    """[B, outer, row, col, heads, dim] -> [B, F*H*W, heads, dim]."""
    if f_tied == 1 and h_reduce == 1 and w_reduce == 1:
        batch, outer, row, col, heads, dim = x.shape
        return x.reshape(batch, outer * row * col, heads, dim)

    return rearrange(
        x,
        "b (outer hr wr) (ft row) col head dim -> b (outer ft hr row wr col) head dim",
        ft=f_tied,
        hr=h_reduce,
        wr=w_reduce,
    ).contiguous()


def hashed_candidate_scores(base, num_choices):
    """Deterministic scores for choosing pseudo-random seed indices."""
    candidate = torch.arange(num_choices, device=base.device).view(
        *((1,) * base.ndim),
        num_choices,
    )
    scores = (
        base.unsqueeze(-1).remainder(HASH_MOD)
        + candidate * 374761393
    ).remainder(HASH_MOD)
    scores = (scores * (scores + 1274126177)).remainder(HASH_MOD)
    scores = (scores * (scores + 60493)).remainder(HASH_MOD)
    return scores * num_choices + candidate, candidate


def seed_indices_from_base(base, num_choices, num_seeds, include_seed=None):
    """Map a per-block hash base to deterministic seed indices.

    The num_seeds==2 path preserves the original two-seed sampling rule.
    """
    if include_seed is not None:
        if num_seeds == 1:
            return include_seed.unsqueeze(-1).long()

        scores, candidate = hashed_candidate_scores(base, num_choices)
        scores = scores.masked_fill(candidate == include_seed.unsqueeze(-1), -1)
        random_seeds = scores.topk(num_seeds - 1, dim=-1).indices.long()
        return torch.cat((include_seed.unsqueeze(-1).long(), random_seeds), dim=-1)

    if num_seeds == 2:
        seed0 = base.remainder(num_choices)
        step = (base.div(num_choices, rounding_mode="floor").remainder(num_choices - 1)) + 1
        seed1 = (seed0 + step).remainder(num_choices)
        return torch.stack((seed0, seed1), dim=-1).long()

    scores, _ = hashed_candidate_scores(base, num_choices)
    return scores.topk(num_seeds, dim=-1).indices.long()


def fast_seed_indices_with_include(base, num_choices, num_seeds, include_seed):
    """Choose distinct deterministic seeds without scoring every candidate."""
    if num_seeds == 1:
        return include_seed.unsqueeze(-1).long()
    if num_seeds > num_choices:
        raise ValueError("num_seeds cannot exceed num_choices.")

    pool_size = num_choices - 1
    extra_count = num_seeds - 1
    offsets = torch.arange(extra_count, device=base.device).view(
        *((1,) * base.ndim),
        extra_count,
    )
    extra = (base.unsqueeze(-1).remainder(pool_size) + offsets).remainder(pool_size)
    extra = extra + (extra >= include_seed.unsqueeze(-1)).long()
    return torch.cat((include_seed.unsqueeze(-1).long(), extra.long()), dim=-1)


def local_seed_indices_with_include(num_choices, num_seeds, include_seed):
    """Pick the nearest rows/columns around the matched ith seed."""
    if num_seeds == 1:
        return include_seed.unsqueeze(-1).long()
    if num_seeds > num_choices:
        raise ValueError("num_seeds cannot exceed num_choices.")

    candidate = torch.arange(num_choices, device=include_seed.device).view(
        *((1,) * include_seed.ndim),
        num_choices,
    )
    distance = (candidate - include_seed.unsqueeze(-1)).abs()
    score = -distance * (num_choices + 1) - candidate
    score = score.masked_fill(candidate == include_seed.unsqueeze(-1), torch.iinfo(score.dtype).min)
    extras = score.topk(num_seeds - 1, dim=-1).indices.long()
    return torch.cat((include_seed.unsqueeze(-1).long(), extras), dim=-1)


def strided_seed_indices_with_include(num_choices, num_seeds, include_seed):
    """Pick ith plus evenly spaced rows/columns to cover the whole block."""
    if num_seeds == 1:
        return include_seed.unsqueeze(-1).long()
    if num_seeds > num_choices:
        raise ValueError("num_seeds cannot exceed num_choices.")

    step = max(1, num_choices // num_seeds)
    offsets = torch.arange(num_seeds, device=include_seed.device).view(
        *((1,) * include_seed.ndim),
        num_seeds,
    )
    return (include_seed.unsqueeze(-1).long() + offsets.long() * step).remainder(num_choices)


def seed_indices_for_mode(base, num_choices, num_seeds, include_seed, seed_mode):
    """Choose per-block seed indices for row or column hard-route merging."""
    if seed_mode == "random":
        return seed_indices_from_base(base, num_choices, num_seeds)
    if seed_mode == "random_with_ith":
        return seed_indices_from_base(base, num_choices, num_seeds, include_seed=include_seed)
    if seed_mode == "fast_random_with_ith":
        return fast_seed_indices_with_include(base, num_choices, num_seeds, include_seed)
    if seed_mode == "local_with_ith":
        return local_seed_indices_with_include(num_choices, num_seeds, include_seed)
    if seed_mode == "strided_with_ith":
        return strided_seed_indices_with_include(num_choices, num_seeds, include_seed)
    raise ValueError(f"unsupported seed_mode={seed_mode!r}.")


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
    """Deterministic query-row seeds for each hard-route block."""
    if num_seeds > query_rows:
        raise ValueError("num_merge_tokens cannot exceed query rows per block.")

    device = query_outer.device
    key_row = torch.arange(key_rows, device=device).view(1, 1, key_rows, 1)
    query_col = torch.arange(query_columns, device=device).view(1, 1, 1, query_columns)
    base = (
        query_outer.view(-1, 1, 1, 1) * 73856093
        + key_outer.view(1, -1, 1, 1) * 19349663
        + key_row * 83492791
        + query_col * 2654435761
        + int(seed)
    )
    ith_seed = key_row.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns)
    return seed_indices_for_mode(base, query_rows, num_seeds, ith_seed, seed_mode)


def seed_columns_by_mode(
    query_outer,
    key_outer,
    query_rows,
    key_columns,
    query_columns,
    num_seeds,
    seed,
    seed_mode,
):
    """Deterministic query-column seeds for each transposed hard-route block."""
    if num_seeds > query_columns:
        raise ValueError("num_merge_tokens cannot exceed query columns per block.")

    device = query_outer.device
    query_row = torch.arange(query_rows, device=device).view(1, 1, query_rows, 1)
    key_col = torch.arange(key_columns, device=device).view(1, 1, 1, key_columns)
    base = (
        query_outer.view(-1, 1, 1, 1) * 73856093
        + key_outer.view(1, -1, 1, 1) * 19349663
        + query_row * 83492791
        + key_col * 2654435761
        + int(seed)
    )
    ith_seed = key_col.expand(query_outer.numel(), key_outer.numel(), query_rows, key_columns)
    return seed_indices_for_mode(base, query_columns, num_seeds, ith_seed, seed_mode)


def gather_seed_queries(q, seed_rows):
    # q: [B, A, I, J, H, D], seed_rows: [A, F, K, J, C].
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(q.size(3), device=q.device).view(1, 1, 1, -1, 1)
    return q[:, query_outer, seed_rows, query_col]


def gather_seed_queries_transpose(q, seed_cols):
    # q: [B, A, I, J, H, D], seed_cols: [A, F, I, L, C].
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_row = torch.arange(q.size(2), device=q.device).view(1, 1, -1, 1, 1)
    return q[:, query_outer, query_row, seed_cols]


def soft_merge_weights(merge_logits, dtype):
    """Softmax all keys for each seed and keep the corresponding log-probs."""
    merge_logprob = torch.log_softmax(merge_logits.float(), dim=-1)
    merge_weights = merge_logprob.exp().to(dtype)
    return merge_weights, merge_logprob


def masked_soft_merge_weights(merge_logits, valid, dtype):
    """Softmax valid keys only; all-invalid rows produce zero merge weights."""
    valid = valid.to(torch.bool)
    merge_logits = merge_logits.float().masked_fill(~valid, torch.finfo(torch.float32).min)
    has_valid = valid.any(dim=-1, keepdim=True)
    merge_logits = torch.where(has_valid, merge_logits, torch.zeros_like(merge_logits))

    merge_logprob = torch.log_softmax(merge_logits, dim=-1).masked_fill(~valid, 0.0)
    merge_weights = merge_logprob.exp().masked_fill(~valid, 0.0)
    merge_logprob = torch.where(has_valid, merge_logprob, torch.zeros_like(merge_logprob))
    return merge_weights.to(dtype), merge_logprob


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

    valid = torch.zeros(
        1,
        frames,
        padded_height,
        padded_width,
        1,
        1,
        dtype=torch.bool,
        device=x.device,
    )
    valid[:, :, :height, :width] = True
    return (
        padded.reshape(batch, frames * padded_height * padded_width, heads, dim),
        padded_height,
        padded_width,
        valid.reshape(1, frames * padded_height * padded_width, 1, 1),
    )


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


def biased_sdpa(query, key, value, bias, valid=None, scale=1.0):
    """Attention over compressed tokens with a per-key additive bias.

    query: [G, Q, D], key/value: [G, K, D], bias: [G, K].
    The attention logits are scale * query @ key.T - bias.
    """
    attn_bias = -bias.float()
    if valid is not None:
        attn_bias = attn_bias.masked_fill(~valid, torch.finfo(attn_bias.dtype).min)

    query_for_attn = query.to(key.dtype) if query.dtype != key.dtype else query
    out = F.scaled_dot_product_attention(
        query_for_attn.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        attn_mask=attn_bias.to(query_for_attn.dtype).unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(1).to(query.dtype)
