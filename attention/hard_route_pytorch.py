import torch


FP32_MIN = torch.finfo(torch.float32).min


def seed_key_logits(seed_queries, key, scale):
    """Score each seed query against each key in the same regular block.

    seed_queries: [B, A, F, K, J, C, H, D]
    key:          [B, F, K, L, H, D]
    return:       [B, H, A, F, K, J, C, L]
    """
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key)
    return logits.float() * float(scale)


def seed_key_logits_transpose(seed_queries, key, scale):
    """Score each seed query against each key in the transposed block.

    seed_queries: [B, A, F, I, L, C, H, D]
    key:          [B, F, K, L, H, D]
    return:       [B, H, A, F, I, L, C, K]
    """
    logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, key)
    return logits.float() * float(scale)


def assign_keys_to_seeds(seed_logits):
    """Assign every key to its best seed query.

    seed_logits: [..., C, L]
      C = number of seed queries inside the block
      L = number of keys inside the block

    return: [..., C, L] boolean mask. Empty seed clusters are kept alive by
    assigning key 0 to them, matching the Triton implementation.
    """
    best_seed = seed_logits.argmax(dim=-2, keepdim=True)
    assigned = torch.zeros_like(seed_logits, dtype=torch.bool)
    assigned.scatter_(dim=-2, index=best_seed, value=True)

    key0 = torch.zeros_like(assigned)
    key0[..., 0] = True
    empty_cluster = ~assigned.any(dim=-1, keepdim=True)
    return assigned | (empty_cluster & key0)


def _masked_logits(seed_logits, assigned):
    return seed_logits.float().masked_fill(~assigned, FP32_MIN)


def _assigned_key_softmax(seed_logits, assigned):
    """Softmax over only the keys assigned to each seed."""
    active_logits = _masked_logits(seed_logits, assigned)
    log_denom = torch.logsumexp(active_logits, dim=-1, keepdim=True)
    weights = (active_logits - log_denom).exp().masked_fill(~assigned, 0.0)
    return weights, log_denom.squeeze(-1)


def hard_route_merge_weights_and_bias(seed_logits, dtype):
    """Build merge weights and routing bias from seed/key logits.

    This is the core operation accelerated by Triton:
      1. assign keys to seed queries,
      2. softmax over each seed's assigned keys,
      3. reuse the same weights for KV merge and entropy/routing bias.
    """
    assigned = assign_keys_to_seeds(seed_logits)
    logits = seed_logits.float()
    weights, log_denom = _assigned_key_softmax(logits, assigned)

    merge_weights = weights.to(dtype)
    weights_fp32 = merge_weights.float()
    expected_logit = (weights_fp32 * logits).sum(dim=-1)
    routing_bias = expected_logit - log_denom * weights_fp32.sum(dim=-1)
    return merge_weights, routing_bias, assigned.any(dim=-1)


def _merge_regular(merge_weights, key, value):
    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_keys, merged_values


def _merge_transpose(merge_weights, key, value):
    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, key)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, value)
    return merged_keys, merged_values


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
    logits = seed_key_logits(seed_queries, key, scale)
    weights, routing_bias, cluster_has_keys = hard_route_merge_weights_and_bias(logits, key.dtype)
    merged_keys, merged_values = _merge_regular(weights, key, value)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for the transposed hard-route merge.

    The regular block is I queries x L keys. The transposed block is
    J queries x K keys, so seed query columns compress key rows.
    """
    logits = seed_key_logits_transpose(seed_queries, key, scale)
    weights, routing_bias, _ = hard_route_merge_weights_and_bias(logits, key.dtype)
    merged_keys, merged_values = _merge_transpose(weights, key, value)
    return merged_keys, merged_values, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_merge_weights_and_bias",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
    "seed_key_logits",
    "seed_key_logits_transpose",
]
