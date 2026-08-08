import torch


NEG_INF = torch.finfo(torch.float32).min


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
    assigned[..., 0:1] |= empty_cluster
    return assigned


def hard_route_merge_weights(seed_logits, dtype):
    """Build hard-cluster softmax weights and entropy routing bias.

    The algorithm is:
      1. assign each key to its best seed query;
      2. run a softmax only inside each assigned seed cluster;
      3. compute routing_bias = sum_l alpha_l log(alpha_l).

    seed_logits: [..., C, L]
    weights:     [..., C, L]
    bias:        [..., C]
    """
    assigned = assign_keys_to_seeds(seed_logits)
    active_logits = seed_logits.float().masked_fill(~assigned, NEG_INF)
    log_denom = torch.logsumexp(active_logits, dim=-1, keepdim=True)

    weights = (active_logits - log_denom).exp().masked_fill(~assigned, 0.0)
    weights_fp32 = weights.float()
    expected_logit = (weights_fp32 * seed_logits.float()).sum(dim=-1)
    routing_bias = expected_logit - log_denom.squeeze(-1) * weights_fp32.sum(dim=-1)
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
    seed_logits = (
        torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
        * float(scale)
    )
    weights, routing_bias = hard_route_merge_weights(seed_logits, key.dtype)
    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", weights, key)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", weights, value)
    return merged_keys, merged_values, routing_bias


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for the transposed hard-route merge.

    The regular block is I queries x L keys. The transposed block is
    J queries x K keys, so seed query columns compress key rows.
    """
    seed_logits = (
        torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, key).float()
        * float(scale)
    )
    weights, routing_bias = hard_route_merge_weights(seed_logits, key.dtype)
    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", weights, key)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", weights, value)
    return merged_keys, merged_values, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_merge_weights",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
]
