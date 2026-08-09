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

# Each tuple is: (seed-key logits, merged K, merged V).
REGULAR_MERGE_EQUATIONS = (
    "bafkjchd,bfklhd->bhafkjcl",
    "bhafkjcl,bfklhd->bafjkchd",
    "bhafkjcl,bfklhe->bafjkche",
)
TRANSPOSE_MERGE_EQUATIONS = (
    "bafilchd,bfklhd->bhafilck",
    "bhafilck,bfklhd->bafilchd",
    "bhafilck,bfklhe->bafilche",
)


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
    log_weights = torch.log_softmax(logits, dim=-1)
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
    weights_f32 = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights_f32 * log_weights).sum(dim=-1)
    return weights_f32.to(dtype), routing_bias


def _merge_kv_from_logits(seed_queries, key, value, scale, equations):
    """Shared regular/transpose merge; equations keep the layout explicit."""
    logit_eq, key_eq, value_eq = equations
    seed_key_logits = torch.einsum(logit_eq, seed_queries, key).float() * float(scale)
    merge_weights, routing_bias = hard_route_merge_weights(seed_key_logits, key.dtype)
    merged_key = torch.einsum(key_eq, merge_weights, key)
    merged_value = torch.einsum(value_eq, merge_weights, value)
    return merged_key, merged_value, routing_bias


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
    return _merge_kv_from_logits(
        seed_queries,
        key,
        value,
        scale,
        REGULAR_MERGE_EQUATIONS,
    )


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
    return _merge_kv_from_logits(
        seed_queries,
        key,
        value,
        scale,
        TRANSPOSE_MERGE_EQUATIONS,
    )


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
