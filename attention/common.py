import torch
import torch.nn.functional as F
from einops import rearrange


HASH_MOD = 2147483647
BLOCK_SEED_MODES = {"random", "random_with_ith", "fast_random_with_ith"}

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
            "'random_with_ith', or 'fast_random_with_ith'."
        )
    return seed_mode in {"random_with_ith", "fast_random_with_ith"}


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


def random_seed_rows(
    query_outer,
    key_outer,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
    include_ith=False,
    fast_with_include=False,
):
    """Deterministic pseudo-random query-row seeds for each block."""
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
    ith_seed = None
    if include_ith:
        ith_seed = key_row.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns)
        if fast_with_include:
            return fast_seed_indices_with_include(base, query_rows, num_seeds, ith_seed)
    return seed_indices_from_base(base, query_rows, num_seeds, include_seed=ith_seed)


def random_seed_columns(
    query_outer,
    key_outer,
    query_rows,
    key_columns,
    query_columns,
    num_seeds,
    seed,
    include_ith=False,
    fast_with_include=False,
):
    """Deterministic pseudo-random query-column seeds for transposed blocks."""
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
    ith_seed = None
    if include_ith:
        ith_seed = key_col.expand(query_outer.numel(), key_outer.numel(), query_rows, key_columns)
        if fast_with_include:
            return fast_seed_indices_with_include(base, query_columns, num_seeds, ith_seed)
    return seed_indices_from_base(base, query_columns, num_seeds, include_seed=ith_seed)


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
    merge_logits = merge_logits.float()
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_weights = torch.softmax(merge_logits, dim=-1).to(dtype)
    merge_logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    return merge_weights, merge_logprob


def biased_sdpa(query, key, value, bias, valid=None, scale=1.0):
    """Attention over compressed tokens with a per-key additive bias.

    query: [G, Q, D], key/value: [G, K, D], bias: [G, K].
    The attention logits are scale * query @ key.T - bias.
    """
    attn_bias = -bias.float()
    if valid is not None:
        attn_bias = attn_bias.masked_fill(~valid, torch.finfo(attn_bias.dtype).min)

    out = F.scaled_dot_product_attention(
        query.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        attn_mask=attn_bias.to(query.dtype).unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(1)
