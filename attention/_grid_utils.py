"""Grid reshaping and deterministic seed helpers used outside Wan hard C4.

Upstream Wan now maintains a deliberately narrow hard-C4 implementation. The
causal Self-Forcing one-token path and Hunyuan joint attention still need the
general grid operations below, so they live in this private compatibility
module instead of restoring the removed experimental attention stack.
"""

from __future__ import annotations

import torch
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


def scale_qk(query, key, scale):
    scale_sqrt = float(scale) ** 0.5
    return query * scale_sqrt, key * scale_sqrt


def validate_block_seed_mode(seed_mode, context):
    if seed_mode not in BLOCK_SEED_MODES:
        raise ValueError(
            f"{context} currently supports seed_mode='random', "
            "'random_with_ith', or 'fast_random_with_ith'."
        )
    return seed_mode in {"random_with_ith", "fast_random_with_ith"}


def merge_query_row(query, key_rows, q_init):
    """Choose the query rows used to merge each key row's columns."""

    q_init = normalize_q_init(q_init)
    if q_init == "ith":
        if query.size(2) != key_rows:
            raise ValueError(
                "q_init=ith needs query_row and key_row to have the same size."
            )
        return query
    if q_init == "mean":
        return query.mean(dim=2, keepdim=True).expand(
            -1, -1, key_rows, -1, -1, -1
        )
    if q_init == "1st":
        return query[:, :, :1].expand(-1, -1, key_rows, -1, -1, -1)
    raise ValueError(f"unsupported q_init={q_init!r}; expected mean, 1st, or ith.")


def video_to_blocks(x, f_tied, h_reduce, w_reduce, height, width):
    """Convert ``[B, F*H*W, heads, dim]`` to general GridMerge blocks."""

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
        "b (outer ft hr row wr col) head dim -> "
        "b (outer hr wr) (ft row) col head dim",
        ft=f_tied,
        hr=h_reduce,
        row=height // h_reduce,
        wr=w_reduce,
        col=width // w_reduce,
    ).contiguous()


def blocks_to_video(x, f_tied, h_reduce, w_reduce):
    """Invert :func:`video_to_blocks`."""

    if f_tied == 1 and h_reduce == 1 and w_reduce == 1:
        batch, outer, row, col, heads, dim = x.shape
        return x.reshape(batch, outer * row * col, heads, dim)
    return rearrange(
        x,
        "b (outer hr wr) (ft row) col head dim -> "
        "b (outer ft hr row wr col) head dim",
        ft=f_tied,
        hr=h_reduce,
        wr=w_reduce,
    ).contiguous()


def _hashed_candidate_scores(base, num_choices):
    candidate = torch.arange(num_choices, device=base.device).view(
        *((1,) * base.ndim), num_choices
    )
    scores = (
        base.unsqueeze(-1).remainder(HASH_MOD) + candidate * 374761393
    ).remainder(HASH_MOD)
    scores = (scores * (scores + 1274126177)).remainder(HASH_MOD)
    scores = (scores * (scores + 60493)).remainder(HASH_MOD)
    return scores * num_choices + candidate, candidate


def _seed_indices_from_base(base, num_choices, num_seeds, include_seed=None):
    if include_seed is not None:
        if num_seeds == 1:
            return include_seed.unsqueeze(-1).long()
        scores, candidate = _hashed_candidate_scores(base, num_choices)
        scores = scores.masked_fill(candidate == include_seed.unsqueeze(-1), -1)
        random_seeds = scores.topk(num_seeds - 1, dim=-1).indices.long()
        return torch.cat((include_seed.unsqueeze(-1).long(), random_seeds), dim=-1)
    if num_seeds == 2:
        seed0 = base.remainder(num_choices)
        step = (
            base.div(num_choices, rounding_mode="floor").remainder(num_choices - 1)
            + 1
        )
        return torch.stack((seed0, (seed0 + step).remainder(num_choices)), dim=-1).long()
    scores, _ = _hashed_candidate_scores(base, num_choices)
    return scores.topk(num_seeds, dim=-1).indices.long()


def _fast_seed_indices_with_include(base, num_choices, num_seeds, include_seed):
    if num_seeds == 1:
        return include_seed.unsqueeze(-1).long()
    pool_size = num_choices - 1
    offsets = torch.arange(num_seeds - 1, device=base.device).view(
        *((1,) * base.ndim), num_seeds - 1
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
    """Choose deterministic pseudo-random query-row seeds for each block."""

    if num_seeds > query_rows:
        raise ValueError("num_merge_tokens cannot exceed query rows per block.")
    key_row = torch.arange(key_rows, device=query_outer.device).view(1, 1, key_rows, 1)
    query_col = torch.arange(query_columns, device=query_outer.device).view(
        1, 1, 1, query_columns
    )
    base = (
        query_outer.view(-1, 1, 1, 1) * 73856093
        + key_outer.view(1, -1, 1, 1) * 19349663
        + key_row * 83492791
        + query_col * 2654435761
        + int(seed)
    )
    ith_seed = None
    if include_ith:
        ith_seed = key_row.expand(
            query_outer.numel(), key_outer.numel(), key_rows, query_columns
        )
        if fast_with_include:
            return _fast_seed_indices_with_include(
                base, query_rows, num_seeds, ith_seed
            )
    return _seed_indices_from_base(
        base, query_rows, num_seeds, include_seed=ith_seed
    )


def gather_seed_queries(query, seed_rows):
    """Gather ``[A,F,K,J,C]`` seed rows from block queries."""

    query_outer = torch.arange(query.size(1), device=query.device).view(
        -1, 1, 1, 1, 1
    )
    query_col = torch.arange(query.size(3), device=query.device).view(
        1, 1, 1, -1, 1
    )
    return query[:, query_outer, seed_rows, query_col]


def soft_merge_weights(merge_logits, dtype):
    merge_logits = merge_logits.float()
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    weights = torch.softmax(merge_logits, dim=-1).to(dtype)
    logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    return weights, logprob


def masked_soft_merge_weights(merge_logits, valid, dtype):
    """Softmax valid keys only; all-invalid rows produce zero weights."""

    valid = valid.to(torch.bool)
    merge_logits = merge_logits.float().masked_fill(
        ~valid, torch.finfo(torch.float32).min
    )
    has_valid = valid.any(dim=-1, keepdim=True)
    merge_logits = torch.where(has_valid, merge_logits, torch.zeros_like(merge_logits))
    weights = torch.softmax(merge_logits, dim=-1) * valid.to(merge_logits.dtype)
    logprob = merge_logits - torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    logprob = logprob.masked_fill(~valid, 0.0)
    logprob = torch.where(has_valid, logprob, torch.zeros_like(logprob))
    return weights.to(dtype), logprob


def pad_video_tokens_to_multiples(x, height, width, h_reduce, w_reduce):
    """Spatially pad video tokens to reduction-factor multiples."""

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
    padded[:, :, :height, :width] = x.reshape(
        batch, frames, height, width, heads, dim
    )
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


def crop_video_tokens_from_multiples(
    x, height, width, padded_height, padded_width
):
    """Crop spatially padded video tokens back to their requested geometry."""

    height = int(height)
    width = int(width)
    padded_height = int(padded_height)
    padded_width = int(padded_width)
    if padded_height == height and padded_width == width:
        return x
    batch, seq_len, heads, dim = x.shape
    if seq_len % (padded_height * padded_width):
        raise ValueError(
            "sequence length must be a multiple of padded_height * padded_width."
        )
    frames = seq_len // (padded_height * padded_width)
    x = x.reshape(batch, frames, padded_height, padded_width, heads, dim)
    return (
        x[:, :, :height, :width]
        .contiguous()
        .reshape(batch, frames * height * width, heads, dim)
    )


__all__ = [
    "blocks_to_video",
    "crop_video_tokens_from_multiples",
    "gather_seed_queries",
    "masked_soft_merge_weights",
    "merge_query_row",
    "pad_video_tokens_to_multiples",
    "random_seed_rows",
    "scale_qk",
    "soft_merge_weights",
    "validate_block_seed_mode",
    "video_to_blocks",
]
