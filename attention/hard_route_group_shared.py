import torch
from einops import rearrange

from .common import seed_rows_by_mode, validate_block_seed_mode
from .hard_route import hard_route_compressed_attention, _merge


def _group_seed_queries(q, seed_rows, anchor_rows):
    # q: [B, A, I, J, H, D], seed_rows: [G, F, K, J, C].
    query_col = torch.arange(q.size(3), device=q.device).view(1, 1, 1, -1, 1)
    anchor_rows = anchor_rows.view(-1, 1, 1, 1, 1)
    return q[:, anchor_rows, seed_rows, query_col]


def hard_route_group_shared_attention(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
    query_outer_group_size,
    hard_route_backend="sdpa",
    hard_route_routing_backend="sdpa",
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # Original hard_route compresses KV once per query outer. This variant
    # compresses once per query-outer group and reuses that compressed KV within
    # the group, giving a speed/accuracy middle point between hard_route and
    # hard_route_shared.
    if query_outer_offset:
        raise ValueError("hard_route_group_shared expects the full query outer dimension.")

    validate_block_seed_mode(seed_mode, "group-shared multi-token merging")
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    group_size = int(query_outer_group_size or q.size(1))
    if group_size < 1:
        raise ValueError("query_outer_group_size must be >= 1.")
    if q.size(1) % group_size:
        raise ValueError("query_outer must be divisible by query_outer_group_size.")

    groups = q.size(1) // group_size
    local_anchor = torch.arange(groups, device=q.device) * group_size + group_size // 2
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_rows = seed_rows_by_mode(
        local_anchor,
        key_outer,
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        seed_mode,
    )
    seed_queries = _group_seed_queries(q, seed_rows, local_anchor)

    merged_keys, merged_values, routing_bias, cluster_has_keys = _merge(
        seed_queries,
        k,
        v,
        hard_route_backend,
        scale,
    )

    q_groups = rearrange(q, "b (g a) i j h d -> b g a i j h d", g=groups)
    query = rearrange(q_groups, "b g a i j h d -> (b h g j) (a i) d")
    key = rearrange(merged_keys, "b g f j k c h d -> (b h g j) (f k c) d")
    value = rearrange(merged_values, "b g f j k c h d -> (b h g j) (f k c) d")
    bias = rearrange(routing_bias, "b h g f k j c -> (b h g j) (f k c)")
    del cluster_has_keys
    out = hard_route_compressed_attention(
        query,
        key,
        value,
        bias,
        hard_route_routing_backend,
        scale,
    )
    return rearrange(
        out,
        "(b h g j) (a i) d -> b (g a) i j h d",
        b=q.size(0),
        h=q.size(4),
        g=groups,
        j=q.size(3),
        a=group_size,
        i=q.size(2),
    )


__all__ = ["hard_route_group_shared_attention"]
