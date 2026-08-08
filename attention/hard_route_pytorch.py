"""Readable PyTorch reference for hard-route token merging.

This file is intentionally simple rather than fast. The Triton kernels should
match the tensor math here:
  1. seed queries score every key inside one grid block;
  2. every key is assigned to the seed with the largest score;
  3. each seed runs a softmax over only its assigned keys and merges K/V.
"""

import torch


NEG_INF = torch.finfo(torch.float32).min


def assign_keys_to_seeds(seed_key_logits):
    """Return a mask assigning every key to its best seed query.

    seed_key_logits: [..., C, L]
      C = number of seed queries in this block
      L = number of keys in this block

    Empty seed clusters are kept alive by assigning key 0 to them. This keeps
    the compressed token count fixed and matches the Triton path.
    """
    best_seed = seed_key_logits.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)

    empty_seed = ~assignment.any(dim=-1, keepdim=True)
    assignment[..., :1] |= empty_seed
    return assignment


def masked_log_softmax(logits, mask):
    """Softmax over assigned keys only; masked-out keys get log-probability 0."""
    logits = logits.float().masked_fill(~mask, NEG_INF)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return log_probs.masked_fill(~mask, 0.0)


def hard_cluster_weights(seed_key_logits, dtype):
    """Convert seed/key logits into merge weights and routing bias.

    This is the reference implementation of the first stage:
      1. hard-assign each key to one seed;
      2. softmax only inside that seed's assigned key cluster;
      3. compute routing_bias = sum_l alpha_l log(alpha_l).

    seed_key_logits: [..., C, L]
    weights:         [..., C, L], zeros outside each assigned cluster
    routing_bias:    [..., C]
    """
    assignment = assign_keys_to_seeds(seed_key_logits)
    log_weights = masked_log_softmax(seed_key_logits, assignment)
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    return weights.to(dtype), routing_bias


def hard_route_merge_kv_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for one regular hard-route merge.

    A fixed (query_outer, key_outer, query_column, key_row) block has
    I queries x L keys. This function compresses those L keys into C merged
    KV tokens. The later attention step still attends to all compressed tokens.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]
    merged K/V:   [B, A, F, J, K, C, H, D]
    bias:         [B, H, A, F, K, J, C]
    """
    # logits: [B, H, A, F, K, J, C, L]
    seed_key_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key)
    seed_key_logits = seed_key_logits.float() * float(scale)
    weights, routing_bias = hard_cluster_weights(seed_key_logits, key.dtype)

    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", weights, key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", weights, value)
    return merged_key, merged_value, routing_bias


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for the transposed hard-route merge.

    The regular block is I queries x L keys. The transposed block is
    J queries x K keys, so seed query columns compress key rows.
    """
    # logits: [B, H, A, F, I, L, C, K]
    seed_key_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, key)
    seed_key_logits = seed_key_logits.float() * float(scale)
    weights, routing_bias = hard_cluster_weights(seed_key_logits, key.dtype)

    merged_key = torch.einsum("bhafilck,bfklhd->bafilchd", weights, key)
    merged_value = torch.einsum("bhafilck,bfklhe->bafilche", weights, value)
    return merged_key, merged_value, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "hard_cluster_weights",
    "masked_log_softmax",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
]
