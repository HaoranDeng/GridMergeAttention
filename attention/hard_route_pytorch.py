"""Plain PyTorch reference for hard-route GridMerge.

This file is the algorithmic source of truth. It intentionally favors the
short math-like implementation over kernel-level tricks:

    seed/key logits -> hard assignment -> masked softmax -> merge K/V + bias

The Triton kernels should match these tensor equations.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


FLOAT_NEG_INF = torch.finfo(torch.float32).min

# Shape letters used below:
#   B: batch, H: attention head, D: head dim
#   A: query outer block, F: key outer block
#   I: query rows, J: query columns
#   K: key rows, L: key columns
#   C: merged tokens / seed queries per block. Hard C4 keeps C=4.


_SEED_ROWS_CACHE = {}


def local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows):
    """Choose the four local-with-ith seed query rows used by hard C4.

    For key row k, seed 0 is query row k. The other seeds are nearby query
    rows in the same order as the Triton kernel.
    """
    device = query_outer.device
    key_row = torch.arange(key_rows, device=device).view(1, 1, key_rows, 1)
    seed0 = key_row
    seed1 = torch.where(key_row > 0, key_row - 1, torch.ones_like(key_row))
    seed2 = torch.where(
        key_row == 0,
        torch.full_like(key_row, 2),
        torch.where(key_row == query_rows - 1, torch.full_like(key_row, query_rows - 3), key_row + 1),
    )
    seed3 = torch.where(
        key_row <= 1,
        torch.full_like(key_row, 3),
        torch.where(key_row >= query_rows - 2, torch.full_like(key_row, query_rows - 4), key_row - 2),
    )
    seed_rows = torch.stack((seed0, seed1, seed2, seed3), dim=-1).long()
    return seed_rows.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns, 4)


def cached_local_c4_seed_rows(
    device,
    query_outer_offset,
    query_outer_count,
    key_outer_count,
    key_rows,
    query_columns,
    query_rows,
):
    cache_key = (
        str(device),
        int(query_outer_offset),
        int(query_outer_count),
        int(key_outer_count),
        int(key_rows),
        int(query_columns),
        int(query_rows),
    )
    seed_rows = _SEED_ROWS_CACHE.get(cache_key)
    if seed_rows is not None:
        return seed_rows

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + query_outer_count,
        device=device,
    )
    key_outer = torch.arange(key_outer_count, device=device)
    seed_rows = local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows)
    _SEED_ROWS_CACHE[cache_key] = seed_rows
    return seed_rows


def gather_seed_queries(query, seed_rows):
    """Gather seed queries for every (query outer, key outer, key row, query column) block."""
    query_outer = torch.arange(query.size(1), device=query.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(query.size(3), device=query.device).view(1, 1, 1, -1, 1)
    return query[:, query_outer, seed_rows, query_col]


def biased_sdpa(query, key, value, bias, scale=1.0):
    """Full compressed attention with additive entropy/routing bias."""
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


def hard_route_merge_kv_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for hard C4 merge.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]

    A fixed (A, F, J, K) block has query rows I=30 and key columns L=52.
    The C seed queries compress those L keys into C merged K/V tokens.

    Returns:
      merged_key/value: [B, A, F, J, K, C, H, D]
      routing_bias:     [B, H, A, F, K, J, C]
    """
    seed_key_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    seed_key_logits = seed_key_logits * float(scale)

    # Hard-assign each key column L to the seed C with the largest seed-key logit.
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    best_seed = seed_key_logits.argmax(dim=-2, keepdim=True)
    assignment.scatter_(dim=-2, index=best_seed, value=True)

    logits = seed_key_logits.masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(logits, dim=-1).masked_fill(~assignment, 0.0)
    weights_f32 = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights_f32 * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(~assignment.any(dim=-1), float("inf"))
    merge_weights = weights_f32.to(key.dtype)

    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_key, merged_value, routing_bias


def hard_route_attention_pytorch(q, k, v, scale, query_outer_offset=0):
    """Readable end-to-end hard C4 attention reference.

    q:   [B, A, I, J, H, D]
    k/v: [B, F, K, L, H, D]

    A fixed (a, f, j, k) block has I query rows x L key columns. This PyTorch
    path computes the same C4 merge and full compressed attention as Triton.
    """
    seed_rows = cached_local_c4_seed_rows(
        q.device,
        query_outer_offset,
        q.size(1),
        k.size(1),
        k.size(2),
        q.size(3),
        q.size(2),
    )
    seed_queries = gather_seed_queries(q, seed_rows)
    merged_keys, merged_values, routing_bias = hard_route_merge_kv_pytorch(
        seed_queries,
        k,
        v,
        scale,
    )

    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    out = biased_sdpa(query, key, value, bias, scale=scale)
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


__all__ = [
    "biased_sdpa",
    "cached_local_c4_seed_rows",
    "gather_seed_queries",
    "hard_route_attention_pytorch",
    "hard_route_merge_kv_pytorch",
    "local_c4_seed_rows",
]
