import torch


NEG_INF = torch.finfo(torch.float32).min


def _regular_seed_logits(seed_queries, key, scale):
    """Score seed queries against keys in the regular 30-query x 52-key block.

    seed_queries: [B, A, F, K, J, C, H, D]
    key:          [B, F, K, L, H, D]
    return:       [B, H, A, F, K, J, C, L]
    """
    return torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float() * float(scale)


def _transpose_seed_logits(seed_queries, key, scale):
    """Score seed queries against keys in the transposed 52-query x 30-key block.

    seed_queries: [B, A, F, I, L, C, H, D]
    key:          [B, F, K, L, H, D]
    return:       [B, H, A, F, I, L, C, K]
    """
    return torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, key).float() * float(scale)


def assign_keys_to_seeds(seed_logits):
    """Assign every key to the seed query with the largest dot product.

    seed_logits: [..., C, L]
      C = number of seed queries inside the block
      L = number of keys inside the block

    return: [..., C, L] boolean mask.

    Empty seed clusters are kept alive by assigning key 0 to them. This matches
    the Triton fast path and keeps the number of compressed tokens fixed.
    """
    best_seed = seed_logits.argmax(dim=-2, keepdim=True)
    assigned = torch.zeros_like(seed_logits, dtype=torch.bool)
    assigned.scatter_(dim=-2, index=best_seed, value=True)

    empty_cluster = ~assigned.any(dim=-1, keepdim=True)
    assigned[..., :1] = assigned[..., :1] | empty_cluster
    return assigned


def _cluster_softmax_and_bias(seed_logits, assigned, dtype):
    """Return per-cluster KV weights and the entropy routing correction.

    For one seed cluster with logits z and softmax weights alpha:
        routing_bias = sum_l alpha_l log(alpha_l)
                     = E_alpha[z] - logsumexp(z)
    """
    active_logits = seed_logits.float().masked_fill(~assigned, NEG_INF)
    log_denom = torch.logsumexp(active_logits, dim=-1, keepdim=True)

    merge_weights = (active_logits - log_denom).exp().masked_fill(~assigned, 0.0)
    weights_for_bias = merge_weights.float()
    expected_logit = (weights_for_bias * seed_logits.float()).sum(dim=-1)
    normalizer_term = log_denom.squeeze(-1) * weights_for_bias.sum(dim=-1)
    routing_bias = expected_logit - normalizer_term
    return merge_weights.to(dtype), routing_bias


def _merge_regular_blocks(merge_weights, key, value):
    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_keys, merged_values


def _merge_transpose_blocks(merge_weights, key, value):
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
    seed_logits = _regular_seed_logits(seed_queries, key, scale)
    assigned = assign_keys_to_seeds(seed_logits)
    weights, routing_bias = _cluster_softmax_and_bias(seed_logits, assigned, key.dtype)
    merged_keys, merged_values = _merge_regular_blocks(weights, key, value)
    return merged_keys, merged_values, routing_bias


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for the transposed hard-route merge.

    The regular block is I queries x L keys. The transposed block is
    J queries x K keys, so seed query columns compress key rows.
    """
    seed_logits = _transpose_seed_logits(seed_queries, key, scale)
    assigned = assign_keys_to_seeds(seed_logits)
    weights, routing_bias = _cluster_softmax_and_bias(seed_logits, assigned, key.dtype)
    merged_keys, merged_values = _merge_transpose_blocks(weights, key, value)
    return merged_keys, merged_values, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
]
