import torch
from einops import rearrange

from .common import (
    biased_sdpa,
    gather_seed_queries,
    random_seed_rows,
    scale_qk,
    validate_block_seed_mode,
)
from .hard_route_triton import triton_hard_route_merge


def assign_keys_to_seeds(seed_logits):
    # seed_logits: [B, H, A, F, K, J, C, L].
    num_seeds = seed_logits.size(-2)
    num_keys = seed_logits.size(-1)

    assignments = seed_logits.argmax(dim=-2)
    assignment_mask = torch.nn.functional.one_hot(
        assignments,
        num_classes=num_seeds,
    ).movedim(-1, -2).bool()

    has_seed = assignment_mask.any(dim=-1)
    best_key_per_seed = seed_logits.argmax(dim=-1)
    key_idx = torch.arange(num_keys, device=seed_logits.device)
    key_idx = key_idx.view(*((1,) * best_key_per_seed.ndim), num_keys)
    force_mask = (~has_seed).unsqueeze(-1) & (best_key_per_seed.unsqueeze(-1) == key_idx)

    force_scores = seed_logits.masked_fill(~force_mask, torch.finfo(seed_logits.dtype).min)
    forced_assignments = force_scores.argmax(dim=-2)
    has_forced_owner = force_mask.any(dim=-2)
    assignments = torch.where(has_forced_owner, forced_assignments, assignments)

    return torch.nn.functional.one_hot(assignments, num_classes=num_seeds).movedim(-1, -2).bool()


def hard_route_merge_weights(seed_logits, dtype):
    """Assign each key to one seed, then softmax within each routed cluster."""
    assignment_mask = assign_keys_to_seeds(seed_logits)
    cluster_has_keys = assignment_mask.any(dim=-1)

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logprob = torch.log_softmax(masked_logits, dim=-1)
    merge_weights = merge_logprob.exp().masked_fill(~assignment_mask, 0).to(dtype)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)
    return merge_weights, merge_logprob, cluster_has_keys


def _torch_merge(seed_queries, k, v):
    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k).float()
    merge_weights, merge_logprob, cluster_has_keys = hard_route_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def _merge(seed_queries, k, v, backend):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return _torch_merge(seed_queries, k, v)
    if backend == "triton":
        return triton_hard_route_merge(seed_queries, k, v)
    raise ValueError(f"unsupported hard_route_backend={backend!r}; expected 'sdpa' or 'triton'.")


def hard_route_attention(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
    hard_route_backend="sdpa",
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A fixed (a, f, j, k) block has I queries x L keys. This path merges
    # the L keys into several hard-assigned merged tokens instead of one.
    include_ith = validate_block_seed_mode(seed_mode, "multi-token merging")
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    q, k = scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_rows = random_seed_rows(
        query_outer,
        key_outer,
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
    )
    seed_queries = gather_seed_queries(q, seed_rows)

    merged_keys, merged_values, routing_bias, cluster_has_keys = _merge(
        seed_queries,
        k,
        v,
        hard_route_backend,
    )

    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    del cluster_has_keys
    out = biased_sdpa(query, key, value, bias)
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


__all__ = ["assign_keys_to_seeds", "hard_route_attention", "hard_route_merge_weights"]
