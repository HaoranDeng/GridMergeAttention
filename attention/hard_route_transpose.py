import torch
from einops import rearrange

from .common import (
    gather_seed_queries_transpose,
    seed_columns_by_mode,
    validate_block_seed_mode,
)
from .hard_route import hard_route_compressed_attention, hard_route_merge_weights
from .hard_route_triton import triton_hard_route_merge, triton_hard_route_merge_c4


def _torch_merge(seed_queries, k, v, scale):
    seed_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, k).float()
    seed_logits = seed_logits * float(scale)
    merge_weights, merge_logprob, _ = hard_route_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, k)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, v)
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    return merged_keys, merged_values, routing_bias


def _triton_merge(seed_queries, k, v, scale, use_c4_kernel=False, block_j_target=8):
    # Reuse the row-routing Triton kernel by treating original key columns as
    # fixed key rows and original key rows as the merged key-token dimension.
    seed_queries_for_kernel = rearrange(
        seed_queries,
        "b a f i l c h d -> b a f l i c h d",
    ).contiguous()
    key_for_kernel = rearrange(k, "b f k l h d -> b f l k h d").contiguous()
    value_for_kernel = rearrange(v, "b f k l h d -> b f l k h d").contiguous()

    if use_c4_kernel:
        merged_keys, merged_values, routing_bias, cluster_has_keys = triton_hard_route_merge_c4(
            seed_queries_for_kernel,
            key_for_kernel,
            value_for_kernel,
            scale=scale,
            block_j_target=block_j_target,
        )
    else:
        merged_keys, merged_values, routing_bias, cluster_has_keys = triton_hard_route_merge(
            seed_queries_for_kernel,
            key_for_kernel,
            value_for_kernel,
            scale=scale,
        )
    routing_bias = rearrange(routing_bias, "b h a f l i c -> b h a f i l c")
    del cluster_has_keys
    return merged_keys, merged_values, routing_bias


def _merge(seed_queries, k, v, backend, scale):
    backend = (backend or "sdpa").lower()
    if backend == "sdpa":
        return _torch_merge(seed_queries, k, v, scale)
    if backend == "triton":
        return _triton_merge(seed_queries, k, v, scale)
    if backend == "triton_c4":
        return _triton_merge(seed_queries, k, v, scale, use_c4_kernel=True)
    if backend == "triton_c4_j16":
        return _triton_merge(seed_queries, k, v, scale, use_c4_kernel=True, block_j_target=16)
    raise ValueError(
        f"unsupported hard_route_backend={backend!r}; expected 'sdpa', 'triton', "
        "'triton_c4', or 'triton_c4_j16'."
    )


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

    merged_keys, merged_values, routing_bias = _merge(
        seed_queries,
        k,
        v,
        hard_route_backend,
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
