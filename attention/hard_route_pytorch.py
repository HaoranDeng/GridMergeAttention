"""Plain PyTorch reference for hard-route GridMerge.

This file is the algorithmic source of truth. It intentionally favors the
short math-like implementation over kernel-level tricks:

    seed/key logits -> hard assignment -> masked softmax -> merge K/V + bias

The Triton kernels should match these tensor equations.
"""

import torch


FLOAT_NEG_INF = torch.finfo(torch.float32).min

# Shape letters used below:
#   B: batch, H: attention head, D: head dim
#   A: query outer block, F: key outer block
#   I: query rows, J: query columns
#   K: key rows, L: key columns
#   C: merged tokens / seed queries per block


def assign_keys_to_seeds(seed_key_logits):
    """Assign each key to the seed query with the largest dot product.

    seed_key_logits: [..., C, L]
      C: seed queries inside one block
      L: keys inside one block

    Returns: bool mask [..., C, L]. If a seed receives no key, key 0 is assigned
    to it so every block still produces exactly C compressed tokens.
    """
    best_seed = seed_key_logits.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)

    empty_seed = ~assignment.any(dim=-1, keepdim=True)
    assignment[..., :1] |= empty_seed
    return assignment


def masked_cluster_log_softmax(seed_key_logits, assignment):
    """Softmax over keys in each assigned seed cluster."""
    logits = seed_key_logits.float().masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return log_weights.masked_fill(~assignment, 0.0)


def hard_route_merge_weights(seed_key_logits, dtype):
    """Turn seed-key logits into merge weights and entropy routing bias.

    merge_weights[..., c, l] is nonzero only when key l is assigned to seed c.
    routing_bias[..., c] = sum_l alpha_l log(alpha_l). The second attention
    stage subtracts this value, which preserves the entropy term in the
    softmax variational objective.
    """
    assignment = assign_keys_to_seeds(seed_key_logits)
    log_weights = masked_cluster_log_softmax(seed_key_logits, assignment)
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    return weights.to(dtype), routing_bias


def hard_route_merge_kv_pytorch(seed_queries, key, value, scale):
    """Merge regular 30-query x 52-key blocks.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]

    A fixed (A, F, J, K) block has query rows I=30 and key columns L=52.
    The C seed queries compress those L keys into C merged K/V tokens.

    Returns:
      merged_key/value: [B, A, F, J, K, C, H, D]
      routing_bias:     [B, H, A, F, K, J, C]
    """
    # seed_key_logits: [B, H, A, F, K, J, C, L]
    seed_key_logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl",
        seed_queries,
        key,
    ).float() * float(scale)
    merge_weights, routing_bias = hard_route_merge_weights(seed_key_logits, key.dtype)

    # merge_weights softmax-compresses the L keys inside each 30q x 52k block.
    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_key, merged_value, routing_bias


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Merge transposed 52-query x 30-key blocks.

    seed_queries: [B, A, F, I, L, C, H, D]
    key/value:    [B, F, K, L, H, D]

    A fixed (A, F, I, L) block has query columns J=52 and key rows K=30.
    The C seed queries compress those K keys into C merged K/V tokens.

    Returns:
      merged_key/value: [B, A, F, I, L, C, H, D]
      routing_bias:     [B, H, A, F, I, L, C]
    """
    # seed_key_logits: [B, H, A, F, I, L, C, K]
    seed_key_logits = torch.einsum(
        "bafilchd,bfklhd->bhafilck",
        seed_queries,
        key,
    ).float() * float(scale)
    merge_weights, routing_bias = hard_route_merge_weights(seed_key_logits, key.dtype)
    merged_key = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, key)
    merged_value = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, value)
    return merged_key, merged_value, routing_bias


# Legacy analysis scripts import this older name.
assigned_key_log_softmax = masked_cluster_log_softmax


__all__ = [
    "assign_keys_to_seeds",
    "assigned_key_log_softmax",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
    "hard_route_merge_weights",
    "masked_cluster_log_softmax",
]
