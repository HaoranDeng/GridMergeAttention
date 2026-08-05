import torch
from einops import rearrange

from .common import (
    biased_sdpa,
    gather_seed_queries,
    hard_route_merge_weights,
    random_seed_rows,
    scale_qk,
    validate_block_seed_mode,
)


def hard_route_attention(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
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

    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k).float()
    merge_weights, merge_logprob, cluster_has_keys = hard_route_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)

    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    valid = rearrange(cluster_has_keys, "b h a f k j c -> (b h a j) (f k c)")

    out = biased_sdpa(query, key, value, bias, valid=valid)
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


__all__ = ["hard_route_attention"]
