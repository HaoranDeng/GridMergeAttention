import torch
from einops import rearrange

from .common import (
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
    routing_bias = rearrange(routing_bias, "b h a f k j c -> b h a f j k c 1")

    routing_logits = torch.einsum("bafjkchd,baijhd->bhafjkci", merged_keys, q)
    routing_logits = routing_logits - routing_bias
    cluster_has_keys = rearrange(cluster_has_keys, "b h a f k j c -> b h a f j k c 1")
    routing_logits = routing_logits.masked_fill(
        ~cluster_has_keys,
        torch.finfo(routing_logits.dtype).min,
    )

    routing_weights = rearrange(routing_logits, "b h a f j k c i -> b h a j i (f k c)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k c) -> b h a f j k c i",
        f=k.size(1),
        k=k.size(2),
        c=num_merge_tokens,
    )

    return torch.einsum("bhafjkci,bafjkche->baijhe", routing_weights, merged_values)


__all__ = ["hard_route_attention"]
