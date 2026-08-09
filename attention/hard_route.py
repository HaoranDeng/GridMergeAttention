import torch
from einops import rearrange

from .common import (
    biased_sdpa,
    gather_seed_queries,
    local_c4_seed_rows,
)
from .hard_route_backend import (
    direct_compressed_attention,
    is_hard_c4_backend,
    merge_kv_direct_c4,
)
from .hard_route_pytorch import hard_route_merge_kv_pytorch


_SEED_ROWS_CACHE = {}


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
):
    cache_key = (
        str(device),
        int(query_outer_offset),
        int(query_outer_count),
        int(key_outer_count),
        int(key_rows),
        int(query_columns),
        int(query_rows),
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
    seed_rows = local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows)
    _SEED_ROWS_CACHE[cache_key] = seed_rows
    return seed_rows


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
    # A fixed (a, f, j, k) block has I queries x L keys. Hard C4 merges the
    # L keys into four local-with-ith seed clusters.
    use_hard_c4 = is_hard_c4_backend(hard_route_backend)
    del num_merge_tokens, seed_mode, seed

    if use_hard_c4:
        # Optimized C4 path: Triton merge writes token-major compressed KV
        # directly into the layout expected by direct compressed attention.
        q = q.contiguous()
        key, value, bias, _ = merge_kv_direct_c4(
            q,
            k,
            v,
            scale,
            assume_contiguous=True,
        )
        output = output if output is not None and output.is_contiguous() else None
        return direct_compressed_attention(
            q,
            key,
            value,
            bias,
            hard_route_routing_backend,
            scale,
            output=output,
            assume_contiguous=True,
        )
    else:
        # Readable path: gather seed queries, merge KV, then run full
        # compressed attention over all merged tokens.
        seed_rows = _cached_seed_rows(
            q.device,
            query_outer_offset,
            q.size(1),
            k.size(1),
            k.size(2),
            q.size(3),
            q.size(2),
        )
        seed_queries = gather_seed_queries(q, seed_rows)
        backend = (hard_route_backend or "sdpa").lower()
        if backend == "sdpa":
            merged_keys, merged_values, routing_bias = hard_route_merge_kv_pytorch(
                seed_queries,
                k,
                v,
                scale,
            )
        else:
            raise ValueError(
                f"unsupported hard_route_backend={backend!r}; expected 'sdpa' "
                "or hard_route_backend='hard_c4'."
            )

        query = rearrange(q, "b a i j h d -> (b h a j) i d")
        key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
        value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
        bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")

    out = biased_sdpa(
        query,
        key,
        value,
        bias,
        scale=scale,
    )
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


__all__ = [
    "hard_route_attention",
]
