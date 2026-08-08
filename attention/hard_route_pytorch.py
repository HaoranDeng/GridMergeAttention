import torch


def assign_keys_to_seeds(seed_logits):
    """Hard-assign every key to the seed query with the largest score.

    seed_logits: [B, H, A, F, K, J, C, L]
      C = number of seed queries inside the block
      L = number of keys inside the block
    """
    num_seeds = seed_logits.size(-2)
    num_keys = seed_logits.size(-1)

    best_seed = seed_logits.argmax(dim=-2)
    assigned = torch.nn.functional.one_hot(best_seed, num_classes=num_seeds)
    assigned = assigned.movedim(-1, -2).bool()

    has_key = assigned.any(dim=-1)
    key_idx = torch.arange(num_keys, device=seed_logits.device)
    key_idx = key_idx.view(*((1,) * has_key.ndim), num_keys)

    # Keep empty seed clusters alive by assigning key 0 to them. This is a
    # fallback only; normal blocks route every key to exactly one seed.
    force_key0 = (~has_key).unsqueeze(-1) & (key_idx == 0)
    return assigned | force_key0


def _masked_log_softmax(logits, mask):
    logits = logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    logprob = torch.log_softmax(logits, dim=-1)
    return logprob.masked_fill(~mask, 0.0)


def hard_route_merge_weights(seed_logits, dtype):
    """Return per-seed softmax weights over hard-assigned keys."""
    assigned = assign_keys_to_seeds(seed_logits)
    merge_logprob = _masked_log_softmax(seed_logits, assigned)
    merge_weights = merge_logprob.exp().masked_fill(~assigned, 0.0).to(dtype)
    cluster_has_keys = assigned.any(dim=-1)
    return merge_weights, merge_logprob, cluster_has_keys


def hard_route_merge_kv_pytorch(seed_queries, key, value, scale):
    """Readable reference implementation for regular hard-route blocks.

    Block meaning:
      fixed (query_outer, key_outer, query_column, key_row)
      I queries attend to L keys after those L keys are compressed into C tokens.

    Shapes:
      seed_queries: [B, A, F, K, J, C, H, D]
      key/value:    [B, F, K, L, H, D]
      output K/V:   [B, A, F, J, K, C, H, D]
      bias:         [B, H, A, F, K, J, C]
    """
    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key)
    seed_logits = seed_logits.float() * float(scale)
    merge_weights, merge_logprob, cluster_has_keys = hard_route_merge_weights(
        seed_logits,
        key.dtype,
    )

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def hard_route_merge_kv_transpose_pytorch(seed_queries, key, value, scale):
    """Readable reference implementation for transposed hard-route blocks.

    The regular block is I queries x L keys. The transposed block is
    J queries x K keys, so seed query columns compress key rows.
    """
    seed_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, key)
    seed_logits = seed_logits.float() * float(scale)
    merge_weights, merge_logprob, _ = hard_route_merge_weights(seed_logits, key.dtype)

    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, key)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, value)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    return merged_keys, merged_values, routing_bias


__all__ = [
    "assign_keys_to_seeds",
    "hard_route_merge_weights",
    "hard_route_merge_kv_pytorch",
    "hard_route_merge_kv_transpose_pytorch",
]
