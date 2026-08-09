import torch
from einops import rearrange

from .common import (
    gather_seed_queries_transpose,
    seed_columns_by_mode,
    validate_block_seed_mode,
)
from .hard_route import hard_route_compressed_attention
from .hard_route_pytorch import hard_route_merge_kv_transpose_pytorch


def hard_route_attention_transpose(
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
    # A fixed (a, f, i, l) block has J queries x K keys.
    validate_block_seed_mode(seed_mode, "transposed multi-token merging")
    if num_merge_tokens > q.size(3):
        raise ValueError("num_merge_tokens cannot exceed query columns per block.")

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_cols = seed_columns_by_mode(
        query_outer,
        key_outer,
        q.size(2),
        k.size(3),
        q.size(3),
        num_merge_tokens,
        seed,
        seed_mode,
    )
    seed_queries = gather_seed_queries_transpose(q, seed_cols)

    if (hard_route_backend or "sdpa").lower() != "sdpa":
        raise ValueError("transpose hard-route keeps only hard_route_backend='sdpa'.")
    merged_keys, merged_values, routing_bias = hard_route_merge_kv_transpose_pytorch(
        seed_queries,
        k,
        v,
        scale,
    )

    query = rearrange(q, "b a i j h d -> (b h a i) j d")
    key = rearrange(merged_keys, "b a f i l c h d -> (b h a i) (f l c) d")
    value = rearrange(merged_values, "b a f i l c h d -> (b h a i) (f l c) d")
    bias = rearrange(routing_bias, "b h a f i l c -> (b h a i) (f l c)")
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
        "(b h a i) j d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        i=q.size(2),
    )


__all__ = ["hard_route_attention_transpose"]
