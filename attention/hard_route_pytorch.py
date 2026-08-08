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
    num_seeds = seed_logits.size(-2)
    num_keys = seed_logits.size(-1)

    best_seed = seed_logits.argmax(dim=-2)
    assigned = torch.zeros(
        *seed_logits.shape[:-2],
        num_seeds,
        num_keys,
        dtype=torch.bool,
        device=seed_logits.device,
    )
    assigned.scatter_(dim=-2, index=best_seed.unsqueeze(-2), value=True)

    cluster_has_key = assigned.any(dim=-1)
    key0 = torch.arange(num_keys, device=seed_logits.device) == 0
    key0 = key0.view(*((1,) * cluster_has_key.ndim), num_keys)
    return assigned | ((~cluster_has_key).unsqueeze(-1) & key0)


def masked_log_softmax_over_keys(logits, key_mask):
    """Log-softmax over keys, ignoring keys that are not assigned to a seed."""
    logits = logits.float().masked_fill(~key_mask, torch.finfo(torch.float32).min)
    logprob = torch.log_softmax(logits, dim=-1)
    return logprob.masked_fill(~key_mask, 0.0)


def hard_route_merge_weights(seed_logits, dtype):
    """Convert seed/key logits into per-seed merge weights.

    Returns:
      merge_weights:    [..., C, L]
      merge_logprob:    [..., C, L]
      cluster_has_keys: [..., C]
    """
    assigned = assign_keys_to_seeds(seed_logits)
    merge_logprob = masked_log_softmax_over_keys(seed_logits, assigned)
    merge_weights = merge_logprob.exp().masked_fill(~assigned, 0.0).to(dtype)
    return merge_weights, merge_logprob, assigned.any(dim=-1)


def hard_route_merge_stats(seed_logits, dtype):
    """Return merge weights and routing bias without materializing log-probs."""
    assigned = assign_keys_to_seeds(seed_logits)
    logits = seed_logits.float()
    active_logits = logits.masked_fill(~assigned, torch.finfo(torch.float32).min)

    log_denom = torch.logsumexp(active_logits, dim=-1, keepdim=True)
    weights = torch.exp(logits - log_denom).masked_fill(~assigned, 0.0).to(dtype)

    weights_for_bias = weights.float()
    expected_logit = (weights_for_bias * logits.masked_fill(~assigned, 0.0)).sum(dim=-1)
    weight_sum = weights_for_bias.sum(dim=-1)
    routing_bias = expected_logit - log_denom.squeeze(-1) * weight_sum
    return weights, routing_bias, assigned.any(dim=-1)


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
    weights, routing_bias, cluster_has_keys = hard_route_merge_stats(logits, key.dtype)
    merged_keys, merged_values = merge_kv_from_weights(weights, key, value)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for the transposed hard-route merge.

    The regular block is I queries x L keys. The transposed block is
    J queries x K keys, so seed query columns compress key rows.
    """
    logits = seed_key_logits_transpose(seed_queries, key, scale)
    weights, routing_bias, _ = hard_route_merge_stats(logits, key.dtype)
    merged_keys, merged_values = merge_kv_from_weights_transpose(weights, key, value)
    return merged_keys, merged_values, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_merge_weights",
    "hard_route_merge_stats",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
    "masked_log_softmax_over_keys",
    "merge_kv_from_weights",
    "seed_key_logits",
    "seed_key_logits_transpose",
]
