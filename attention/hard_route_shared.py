import torch
from einops import rearrange

from .common import (
    fast_seed_indices_with_include,
    seed_indices_from_base,
    validate_block_seed_mode,
)
from .hard_route import (
    hard_route_compressed_attention,
    hard_route_grid_compressed_attention,
    hard_route_merge_weights,
)
from .hard_route_triton import (
    triton_hard_route_merge,
    triton_hard_route_merge_c4,
    triton_hard_route_shared_merge_c4,
    triton_hard_route_shared_merge_c4_route_layout,
)


_SHARED_SEED_ROWS_CACHE = {}


def shared_seed_rows(
    key_outer,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
    include_ith=False,
    fast_with_include=False,
):
    """Choose seed query rows once per key outer, key row, and query column."""
    if num_seeds > query_rows:
        raise ValueError("num_merge_tokens cannot exceed query rows per block.")

    key_row = torch.arange(key_rows, device=key_outer.device).view(1, key_rows, 1)
    query_col = torch.arange(query_columns, device=key_outer.device).view(1, 1, query_columns)
    key_outer = key_outer.view(-1, 1, 1)
    base = (
        key_outer * 73856093
        + key_outer * 19349663
        + key_row * 83492791
        + query_col * 2654435761
        + int(seed)
    )
    ith_seed = None
    if include_ith:
        ith_seed = key_row.expand(key_outer.numel(), key_rows, query_columns)
        if fast_with_include:
            return fast_seed_indices_with_include(base, query_rows, num_seeds, ith_seed)
    return seed_indices_from_base(base, query_rows, num_seeds, include_seed=ith_seed)


def cached_shared_seed_rows(
    device,
    key_outers,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
    include_ith=False,
    fast_with_include=False,
):
    """Reuse deterministic shared seed rows across layers and denoise steps."""
    cache_key = (
        str(device),
        int(key_outers),
        int(key_rows),
        int(query_columns),
        int(query_rows),
        int(num_seeds),
        int(seed),
        bool(include_ith),
        bool(fast_with_include),
    )
    seed_rows = _SHARED_SEED_ROWS_CACHE.get(cache_key)
    if seed_rows is None:
        key_outer = torch.arange(key_outers, device=device)
        seed_rows = shared_seed_rows(
            key_outer,
            key_rows,
            query_columns,
            query_rows,
            num_seeds,
            seed,
            include_ith=include_ith,
            fast_with_include=fast_with_include,
        )
        _SHARED_SEED_ROWS_CACHE[cache_key] = seed_rows
    return seed_rows


def gather_shared_seed_queries(q, seed_rows):
    # q: [B, A, I, J, H, D], seed_rows: [F, K, J, C].
    if q.size(1) < seed_rows.size(0):
        raise ValueError("shared hard-route needs the full query outer dimension.")

    key_outer = torch.arange(seed_rows.size(0), device=q.device).view(-1, 1, 1, 1)
    query_col = torch.arange(q.size(3), device=q.device).view(1, 1, -1, 1)
    seed_queries = q[:, key_outer, seed_rows, query_col]
    return rearrange(seed_queries, "b f k j c h d -> b 1 f k j c h d")


def _torch_merge(seed_queries, k, v, scale):
    seed_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k).float()
    seed_logits = seed_logits * float(scale)
    merge_weights, merge_logprob, cluster_has_keys = hard_route_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    return merged_keys, merged_values, routing_bias, cluster_has_keys


def _merge(seed_queries, k, v, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return _torch_merge(seed_queries, k, v, scale)
    if backend == "triton":
        return triton_hard_route_merge(seed_queries, k, v, scale=scale)
    if backend == "triton_c4":
        return triton_hard_route_merge_c4(seed_queries, k, v, scale=scale)
    if backend == "triton_c4_j16":
        return triton_hard_route_merge_c4(seed_queries, k, v, scale=scale, block_j_target=16)
    raise ValueError(
        f"unsupported hard_route_backend={backend!r}; expected 'sdpa', 'triton', "
        "'triton_c4', 'triton_c4_j16', 'triton_c4_direct', 'triton_c4_direct_j16', "
        "'triton_c4_direct_route', 'triton_c4_direct_route_j16', "
        "'triton_c4_direct_route_j32_w2', 'triton_c4_direct_route_j32_w2_fastdot', "
        "'triton_c4_direct_route_j32_w2_nobias', "
        "or 'triton_c4_direct_route_j64'."
    )


def _merge_shared(q, seed_rows, k, v, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "triton_c4_direct":
        return triton_hard_route_shared_merge_c4(q, seed_rows, k, v, scale=scale)
    if backend == "triton_c4_direct_j16":
        return triton_hard_route_shared_merge_c4(q, seed_rows, k, v, scale=scale, block_j_target=16)
    if backend == "triton_c4_direct_route":
        return triton_hard_route_shared_merge_c4_route_layout(q, seed_rows, k, v, scale=scale)
    if backend == "triton_c4_direct_route_j16":
        return triton_hard_route_shared_merge_c4_route_layout(q, seed_rows, k, v, scale=scale, block_j_target=16)
    if backend == "triton_c4_direct_route_j32_w2":
        return triton_hard_route_shared_merge_c4_route_layout(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
        )
    if backend == "triton_c4_direct_route_j32_w2_fastdot":
        return triton_hard_route_shared_merge_c4_route_layout(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            fast_dot=True,
        )
    if backend == "triton_c4_direct_route_j32_w2_nobias":
        return triton_hard_route_shared_merge_c4_route_layout(
            q,
            seed_rows,
            k,
            v,
            scale=scale,
            block_j_target=32,
            num_warps=2,
            write_bias=False,
        )
    if backend == "triton_c4_direct_route_j64":
        return triton_hard_route_shared_merge_c4_route_layout(q, seed_rows, k, v, scale=scale, block_j_target=64)

    seed_queries = gather_shared_seed_queries(q, seed_rows)
    return _merge(seed_queries, k, v, backend, scale)


def hard_route_shared_attention(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
    hard_route_backend="sdpa",
    hard_route_routing_backend="sdpa",
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # Compress each key outer once, then share the compressed KV across all
    # query outers. This keeps the C-seed hard-route rule but removes the
    # repeated per-query-outer KV compression used by hard_route_attention.
    del query_outer_offset
    include_ith = validate_block_seed_mode(seed_mode, "shared multi-token merging")
    fast_with_include = seed_mode == "fast_random_with_ith"
    if q.size(1) != k.size(1):
        raise ValueError("shared hard-route expects query_outer and key_outer to match.")
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    seed_rows = cached_shared_seed_rows(
        q.device,
        k.size(1),
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
        fast_with_include=fast_with_include,
    )
    merged_keys, merged_values, routing_bias, cluster_has_keys = _merge_shared(
        q,
        seed_rows,
        k,
        v,
        hard_route_backend,
        scale,
    )

    route_layout_backend = (hard_route_backend or "").lower() in {
        "triton_c4_direct_route",
        "triton_c4_direct_route_j16",
        "triton_c4_direct_route_j32_w2",
        "triton_c4_direct_route_j32_w2_fastdot",
        "triton_c4_direct_route_j32_w2_nobias",
        "triton_c4_direct_route_j64",
    }
    routing_backend = (hard_route_routing_backend or "").lower()
    if route_layout_backend and routing_backend.startswith("triton_grid_"):
        del cluster_has_keys
        return hard_route_grid_compressed_attention(
            q,
            merged_keys,
            merged_values,
            routing_bias,
            routing_backend,
            scale,
        )

    query = rearrange(q, "b a i j h d -> (b h j) (a i) d")
    if route_layout_backend:
        key = merged_keys
        value = merged_values
        bias = routing_bias
    else:
        key = rearrange(merged_keys[:, 0], "b f j k c h d -> (b h j) (f k c) d")
        value = rearrange(merged_values[:, 0], "b f j k c h d -> (b h j) (f k c) d")
        bias = rearrange(routing_bias[:, :, 0], "b h f k j c -> (b h j) (f k c)")
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
        "(b h j) (a i) d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        j=q.size(3),
        a=q.size(1),
        i=q.size(2),
    )


__all__ = ["hard_route_shared_attention"]
