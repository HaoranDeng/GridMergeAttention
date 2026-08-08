import torch
from einops import rearrange

from .common import (
    biased_sdpa,
    gather_seed_queries,
    seed_rows_by_mode,
    validate_block_seed_mode,
)
from .hard_route_backend import (
    direct_compressed_attention,
    direct_indirect_value_attention,
    direct_seed_key_attention,
    direct_top_key_attention,
    merge_kv_direct_c4,
    parse_direct_route_backend,
)
from .hard_route_pytorch import hard_route_merge_kv_pytorch
from .hard_route_triton import (
    triton_biased_attention,
    triton_hard_route_merge,
    triton_hard_route_merge_c4,
)


_SEED_ROWS_CACHE = {}
_DIRECT_ROUTE_ONLY_FLAGS = (
    ("use_seed_key", "seed-key"),
    ("use_indirect_top_key", "indirect top-key"),
    ("use_indirect_top_kv", "indirect top-KV"),
    ("use_indirect_top_value", "indirect top-value"),
    ("use_indirect_seed_value", "indirect seed-value"),
)


# ---------------------------------------------------------------------------
# Seed query selection.
# ---------------------------------------------------------------------------
def _cached_seed_rows(
    device,
    query_outer_offset,
    query_outer_count,
    key_outer_count,
    key_rows,
    query_columns,
    query_rows,
    num_seeds,
    seed,
    seed_mode,
):
    cache_key = (
        str(device),
        int(query_outer_offset),
        int(query_outer_count),
        int(key_outer_count),
        int(key_rows),
        int(query_columns),
        int(query_rows),
        int(num_seeds),
        int(seed),
        str(seed_mode),
    )
    seed_rows = _SEED_ROWS_CACHE.get(cache_key)
    if seed_rows is not None:
        return seed_rows

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + query_outer_count,
        device=device,
    )
    key_outer = torch.arange(key_outer_count, device=device)
    seed_rows = seed_rows_by_mode(
        query_outer,
        key_outer,
        key_rows,
        query_columns,
        query_rows,
        num_seeds,
        seed,
        seed_mode,
    )
    _SEED_ROWS_CACHE[cache_key] = seed_rows
    return seed_rows


def _build_seed_rows(
    q,
    k,
    query_outer_offset,
    num_merge_tokens,
    seed,
    seed_mode,
):
    return _cached_seed_rows(
        q.device,
        query_outer_offset,
        q.size(1),
        k.size(1),
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        seed_mode,
    )


# ---------------------------------------------------------------------------
# Backend dispatch.
# ---------------------------------------------------------------------------
def _merge_kv(seed_queries, k, v, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return hard_route_merge_kv_pytorch(seed_queries, k, v, scale)
    if backend == "triton":
        return triton_hard_route_merge(seed_queries, k, v, scale=scale)[:3]
    if backend == "triton_c4":
        return triton_hard_route_merge_c4(seed_queries, k, v, scale=scale)[:3]
    if backend == "triton_c4_j16":
        return triton_hard_route_merge_c4(seed_queries, k, v, scale=scale, block_j_target=16)[:3]
    raise ValueError(
        f"unsupported hard_route_backend={backend!r}; expected 'sdpa', 'triton', "
        "'triton_c4', 'triton_c4_j16', or a direct-route C4 backend."
    )


def hard_route_compressed_attention(query, key, value, bias, backend, scale):
    """Second hard-route stage: full attention over compressed KV tokens."""
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return biased_sdpa(query, key, value, bias, scale=scale)
    if backend == "triton":
        return triton_biased_attention(query, key, value, bias, scale=scale)
    raise ValueError(
        f"unsupported hard_route_routing_backend={backend!r}; expected 'sdpa', "
        "'triton', or a direct-route backend."
    )


def _restore_attention_blocks(out, q):
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


def _flatten_queries(q):
    return rearrange(q, "b a i j h d -> (b h a j) i d")


def _flatten_compressed_kv(merged_keys, merged_values, routing_bias):
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    return key, value, bias


def _run_reference_path(q, k, v, scale, seed_rows, hard_route_backend, hard_route_routing_backend):
    """Readable path: gather seed queries, merge KV, then run full attention."""
    seed_queries = gather_seed_queries(q, seed_rows)
    merged_keys, merged_values, routing_bias = _merge_kv(
        seed_queries,
        k,
        v,
        hard_route_backend,
        scale,
    )
    key, value, bias = _flatten_compressed_kv(merged_keys, merged_values, routing_bias)
    out = hard_route_compressed_attention(
        _flatten_queries(q),
        key,
        value,
        bias,
        hard_route_routing_backend,
        scale,
    )
    return _restore_attention_blocks(out, q)


def _route_direct_outputs(
    q,
    source_key,
    source_value,
    key,
    value,
    bias,
    key_scale,
    direct_config,
    routing_backend,
    scale,
    output,
):
    uses_indirect_key = direct_config.use_indirect_top_key or direct_config.use_indirect_top_kv
    uses_original_value = (
        direct_config.use_indirect_top_kv
        or direct_config.use_indirect_top_value
        or direct_config.use_indirect_seed_value
    )

    if direct_config.use_seed_key:
        return direct_seed_key_attention(
            q,
            value,
            bias,
            key_scale,
            routing_backend,
            scale,
            output=output,
            assume_contiguous=True,
        )

    if uses_indirect_key:
        return direct_top_key_attention(
            q,
            source_key,
            source_value if uses_original_value else None,
            key,
            value,
            bias,
            key_scale,
            routing_backend,
            scale,
            output=output,
            assume_contiguous=True,
        )

    if direct_config.use_indirect_top_value or direct_config.use_indirect_seed_value:
        return direct_indirect_value_attention(
            q,
            key,
            source_value,
            value,
            bias,
            routing_backend,
            scale,
            indirect_seed_value=direct_config.use_indirect_seed_value,
            output=output,
            assume_contiguous=True,
        )
    return direct_compressed_attention(
        q,
        key,
        value,
        bias,
        routing_backend,
        scale,
        output=output,
        assume_contiguous=True,
    )


def _raise_for_unsupported_direct_attention(direct_config):
    for flag, name in _DIRECT_ROUTE_ONLY_FLAGS:
        if getattr(direct_config, flag):
            raise ValueError(
                f"{name} hard-route backend requires a triton_direct_*_tokenmajor routing backend."
            )


def _run_direct_fallback_attention(q, key, value, bias, routing_backend, scale):
    """Route direct-merge tensors through a generic group-major attention path."""
    query = _flatten_queries(q)
    out = hard_route_compressed_attention(
        query,
        key.transpose(0, 1),
        value.transpose(0, 1),
        bias.transpose(0, 1),
        routing_backend,
        scale,
    )
    return _restore_attention_blocks(out, q)


def _run_direct_path(q, k, v, scale, seed_rows, direct_config, hard_route_routing_backend, output):
    """Optimized C4 path: Triton merge writes routing-ready token-major KV."""
    q = q.contiguous()
    key, value, bias, key_scale = merge_kv_direct_c4(
        q,
        seed_rows,
        k,
        v,
        direct_config,
        scale,
        assume_contiguous=True,
    )
    output = output if output is not None and output.is_contiguous() else None
    direct_out = _route_direct_outputs(
        q,
        k,
        v,
        key,
        value,
        bias,
        key_scale,
        direct_config,
        hard_route_routing_backend,
        scale,
        output,
    )
    if direct_out is not None:
        return direct_out

    _raise_for_unsupported_direct_attention(direct_config)

    return _run_direct_fallback_attention(q, key, value, bias, hard_route_routing_backend, scale)


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
    hard_route_routing_backend="sdpa",
    output=None,
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A fixed (a, f, j, k) block has I queries x L keys. This path merges
    # the L keys into several hard-assigned merged tokens instead of one.
    validate_block_seed_mode(seed_mode, "multi-token merging")
    direct_backend = parse_direct_route_backend(hard_route_backend)
    direct_path_uses_inline_seed = direct_backend.supported and direct_backend.inline_local_seed
    if direct_path_uses_inline_seed and seed_mode != "local_with_ith":
        raise ValueError("inline-local-seed hard-route backend requires seed_mode='local_with_ith'.")
    if num_merge_tokens > k.size(3):
        raise ValueError("num_merge_tokens cannot exceed key columns per block.")

    seed_rows = None
    if not direct_path_uses_inline_seed:
        seed_rows = _build_seed_rows(
            q,
            k,
            query_outer_offset,
            num_merge_tokens,
            seed,
            seed_mode,
        )

    if direct_backend.supported:
        return _run_direct_path(
            q,
            k,
            v,
            scale,
            seed_rows,
            direct_backend,
            hard_route_routing_backend,
            output,
        )

    return _run_reference_path(
        q,
        k,
        v,
        scale,
        seed_rows,
        hard_route_backend,
        hard_route_routing_backend,
    )


__all__ = [
    "hard_route_attention",
    "hard_route_compressed_attention",
]
