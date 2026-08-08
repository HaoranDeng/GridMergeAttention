import torch


def seed_key_logits(seed_queries, key, scale):
    """Score every seed query against every key in the same block.

    seed_queries: [B, A, F, K, J, C, H, D]
    key:          [B, F, K, L, H, D]
    return:       [B, H, A, F, K, J, C, L]

    C is the number of seed queries. L is the number of keys being compressed.
    """
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key)
    return logits.float() * float(scale)


def seed_key_logits_transpose(seed_queries, key, scale):
    """Transposed layout version of seed_key_logits.

    seed_queries: [B, A, F, I, L, C, H, D]
    key:          [B, F, K, L, H, D]
    return:       [B, H, A, F, I, L, C, K]

    C is the number of seed query columns. K is the number of keys compressed.
    """
    logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, key)
    return logits.float() * float(scale)


def assign_keys_to_seeds(seed_logits):
    """Hard-route each key to the seed with the largest seed/key score.

    seed_logits: [..., C, L]
      C = number of seed queries inside the block
      L = number of keys inside the block

    return: [..., C, L] boolean mask. Exactly one seed owns each key, except
    empty seed clusters are kept alive by assigning key 0 to that seed too.
    """
    best_seed = seed_logits.argmax(dim=-2, keepdim=True)
    assigned = torch.zeros_like(seed_logits, dtype=torch.bool)
    assigned.scatter_(dim=-2, index=best_seed, value=True)

    # Keep empty clusters numerically well-defined by forcing them to own key 0.
    fallback_to_key0 = torch.zeros_like(assigned)
    fallback_to_key0[..., 0] = True
    empty_cluster = ~assigned.any(dim=-1, keepdim=True)
    return assigned | (empty_cluster & fallback_to_key0)


def mask_unassigned_keys(logits, assigned):
    """Keep only the keys owned by each seed cluster."""
    logits = logits.float()
    return logits.masked_fill(~assigned, torch.finfo(logits.dtype).min)


def _assigned_key_softmax(seed_logits, assigned):
    """Softmax over each seed's assigned keys only.

    This returns fp32 weights/log-probs. Callers cast weights back to K/V dtype
    before the KV merge, matching the original reference implementation.
    """
    active_logits = mask_unassigned_keys(seed_logits, assigned)
    log_denom = torch.logsumexp(active_logits, dim=-1, keepdim=True)
    logprob = active_logits - log_denom
    weights = logprob.exp().masked_fill(~assigned, 0.0)
    logprob = logprob.masked_fill(~assigned, 0.0)
    return weights, logprob, log_denom.squeeze(-1)


def masked_log_softmax_over_keys(logits, key_mask):
    """Log-softmax over keys, ignoring keys that are not assigned to a seed."""
    _, logprob, _ = _assigned_key_softmax(logits, key_mask)
    return logprob


def hard_route_merge_weights(seed_logits, dtype):
    """Convert seed/key logits into per-seed merge weights.

    Returns:
      merge_weights:    [..., C, L]
      merge_logprob:    [..., C, L]
      cluster_has_keys: [..., C]
    """
    assigned = assign_keys_to_seeds(seed_logits)
    weights, logprob, _ = _assigned_key_softmax(seed_logits, assigned)
    return weights.to(dtype), logprob, assigned.any(dim=-1)


def hard_route_merge_weights_and_bias(seed_logits, dtype):
    """Core hard-route merge math used by the readable PyTorch reference.

    This is the operation the Triton kernels accelerate:
      1. assign each key to the seed query with the largest seed/key score,
      2. run a softmax over only the keys owned by that seed,
      3. reuse those weights both for KV merging and for the routing bias.
    """
    assigned = assign_keys_to_seeds(seed_logits)
    logits = seed_logits.float()
    weights, _, log_denom = _assigned_key_softmax(logits, assigned)

    merge_weights = weights.to(dtype)
    weights_for_bias = merge_weights.float()
    expected_logit = (weights_for_bias * logits).sum(dim=-1)
    routing_bias = expected_logit - log_denom * weights_for_bias.sum(dim=-1)
    return merge_weights, routing_bias, assigned.any(dim=-1)


def hard_route_merge_stats(seed_logits, dtype):
    """Backward-compatible alias for old analysis scripts."""
    return hard_route_merge_weights_and_bias(seed_logits, dtype)


def merge_kv_from_weights(merge_weights, key, value):
    """Apply regular-layout merge weights to K and V."""
    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_keys, merged_values


def merge_kv_from_weights_transpose(merge_weights, key, value):
    """Apply transpose-layout merge weights to K and V."""
    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, key)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, value)
    return merged_keys, merged_values


def hard_route_merge_kv_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for one regular hard-route merge.

    A fixed (query_outer, key_outer, query_column, key_row) block contains
    I queries x L keys. This function compresses those L keys into C hard
    routed KV tokens. The later attention step still attends to all compressed
    tokens from all blocks.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]
    merged K/V:   [B, A, F, J, K, C, H, D]
    bias:         [B, H, A, F, K, J, C]
    """
    logits = seed_key_logits(seed_queries, key, scale)
    weights, routing_bias, cluster_has_keys = hard_route_merge_weights_and_bias(logits, key.dtype)
    merged_keys, merged_values = merge_kv_from_weights(weights, key, value)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for the transposed hard-route merge.

    The regular block is I queries x L keys. The transposed block is
    J queries x K keys, so seed query columns compress key rows.
    """
    logits = seed_key_logits_transpose(seed_queries, key, scale)
    weights, routing_bias, _ = hard_route_merge_weights_and_bias(logits, key.dtype)
    merged_keys, merged_values = merge_kv_from_weights_transpose(weights, key, value)
    return merged_keys, merged_values, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_merge_weights",
    "hard_route_merge_weights_and_bias",
    "hard_route_merge_stats",
    "mask_unassigned_keys",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
    "masked_log_softmax_over_keys",
    "merge_kv_from_weights",
    "seed_key_logits",
    "seed_key_logits_transpose",
]
