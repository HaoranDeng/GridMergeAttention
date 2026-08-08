"""Readable PyTorch reference for hard-route token merging.

This file is the algorithmic source of truth. It is intentionally written like
the math, not like the fastest implementation:

  seed/key logits -> hard assignment -> masked softmax -> merge K/V + bias

The Triton kernels in ``hard_route_triton.py`` should match this tensor math.
"""

import torch


FLOAT_NEG_INF = torch.finfo(torch.float32).min


def assign_keys_to_seeds(seed_key_logits):
    """Assign each key to exactly one seed.

    seed_key_logits: [..., C, L]
      C: number of seed queries in one block
      L: number of keys in one block

    The output has the same shape. A True entry means this key belongs to that
    seed's cluster. Empty clusters are given key 0 so the number of compressed
    tokens stays fixed.
    """
    best_seed = seed_key_logits.argmax(dim=-2, keepdim=True)
    assigned = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assigned.scatter_(dim=-2, index=best_seed, value=True)

    empty_seed = ~assigned.any(dim=-1, keepdim=True)
    assigned[..., :1] |= empty_seed
    return assigned


def assigned_key_log_softmax(seed_key_logits, assigned):
    """Run log-softmax only over the keys assigned to each seed."""
    masked_logits = seed_key_logits.float().masked_fill(~assigned, FLOAT_NEG_INF)
    log_weights = masked_logits - torch.logsumexp(masked_logits, dim=-1, keepdim=True)
    return log_weights.masked_fill(~assigned, 0.0)


def hard_route_merge_weights(seed_key_logits, dtype):
    """Build merge weights and the entropy-style routing bias.

    seed_key_logits: [..., C, L]
    merge_weights:  [..., C, L], zero outside each seed cluster
    routing_bias:   [..., C], equal to sum_l alpha_l log(alpha_l)
    """
    assigned = assign_keys_to_seeds(seed_key_logits)
    log_weights = assigned_key_log_softmax(seed_key_logits, assigned)
    merge_weights = log_weights.exp().masked_fill(~assigned, 0.0)
    routing_bias = (merge_weights * log_weights).sum(dim=-1)
    return merge_weights.to(dtype), routing_bias


# Backward-compatible names used by older analysis scripts.
masked_log_softmax = assigned_key_log_softmax
hard_cluster_weights = hard_route_merge_weights


def score_regular_blocks(seed_queries, key, scale):
    """Seed-query/key logits for regular 30-query x 52-key blocks."""
    # seed_queries: [B, A, F, K, J, C, H, D]
    # key:          [B, F, K, L, H, D]
    # logits:       [B, H, A, F, K, J, C, L]
    return torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float() * float(scale)


def score_transpose_blocks(seed_queries, key, scale):
    """Seed-query/key logits for transposed 52-query x 30-key blocks."""
    # seed_queries: [B, A, F, I, L, C, H, D]
    # key:          [B, F, K, L, H, D]
    # logits:       [B, H, A, F, I, L, C, K]
    return torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, key).float() * float(scale)


def merge_regular_blocks(merge_weights, key, value):
    """Apply per-cluster weights to K/V for regular blocks."""
    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_key, merged_value


def merge_transpose_blocks(merge_weights, key, value):
    """Apply per-cluster weights to K/V for transposed blocks."""
    merged_key = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, key)
    merged_value = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, value)
    return merged_key, merged_value


def hard_route_merge_kv_pytorch(seed_queries, key, value, scale):
    """Reference merge for the regular layout.

    A fixed (query_outer, key_outer, query_column, key_row) block contains
    30 query tokens x 52 key tokens in the default Wan 480p grid.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]
    merged K/V:   [B, A, F, J, K, C, H, D]
    bias:         [B, H, A, F, K, J, C]
    """
    seed_key_logits = score_regular_blocks(seed_queries, key, scale)
    merge_weights, routing_bias = hard_route_merge_weights(seed_key_logits, key.dtype)
    merged_key, merged_value = merge_regular_blocks(merge_weights, key, value)
    return merged_key, merged_value, routing_bias


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Reference merge for the transposed layout.

    The regular block is 30 queries x 52 keys. The transposed block is
    52 queries x 30 keys: seed query columns compress key rows.
    """
    seed_key_logits = score_transpose_blocks(seed_queries, key, scale)
    merge_weights, routing_bias = hard_route_merge_weights(seed_key_logits, key.dtype)
    merged_key, merged_value = merge_transpose_blocks(merge_weights, key, value)
    return merged_key, merged_value, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "assigned_key_log_softmax",
    "hard_cluster_weights",
    "hard_route_merge_weights",
    "masked_log_softmax",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
    "merge_regular_blocks",
    "merge_transpose_blocks",
    "score_regular_blocks",
    "score_transpose_blocks",
]
