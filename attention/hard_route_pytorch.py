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
#   C: merged tokens / seed queries per block. Hard C4 keeps C=4.


def assign_keys_to_seeds(seed_key_logits):
    """Assign each key to the seed query with the largest dot product.

    seed_key_logits: [..., C, L]
      C: seed queries inside one block
      L: keys inside one block

    Returns: bool mask [..., C, L]. Hard C4 also pins key c to seed c for
    c=0..3, matching the optimized Triton path and guaranteeing that each seed
    produces one compressed token.
    """
    best_seed = seed_key_logits.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    num_seeds = seed_key_logits.size(-2)
    forced_key = torch.arange(num_seeds, device=seed_key_logits.device)
    assignment[..., :, :num_seeds] = False
    assignment[..., forced_key, forced_key] = True
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
    if seed_queries.size(5) != 4:
        raise ValueError("hard C4 PyTorch reference expects exactly four seed queries.")
    seed_key_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    seed_key_logits = seed_key_logits * float(scale)
    merge_weights, routing_bias = hard_route_merge_weights(seed_key_logits, key.dtype)
    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_key, merged_value, routing_bias


# Legacy analysis scripts import this older name.
assigned_key_log_softmax = masked_cluster_log_softmax


__all__ = [
    "assign_keys_to_seeds",
    "assigned_key_log_softmax",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_weights",
    "masked_cluster_log_softmax",
]
