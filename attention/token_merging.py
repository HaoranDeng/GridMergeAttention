"""Maintained Wan hard-C4 attention plus causal one-token GridMerge."""

from __future__ import annotations

import torch

from ._grid_utils import (
    blocks_to_video as general_blocks_to_video,
    crop_video_tokens_from_multiples,
    pad_video_tokens_to_multiples,
    video_to_blocks as general_video_to_blocks,
)
from .causal_gridmerge import causal_gridmerge_one
from .hard_route_pytorch import hard_route_attention_pytorch
from .hard_route_triton import (
    HARD_C4_BACKEND,
    HARD_C4_FUSED_BACKEND,
    hard_route_attention_triton,
    hard_route_attention_triton_fused,
)


LAYOUT_HARD_ROUTE = "hard_route"


def _hard_c4_video_to_blocks(x, height, width):
    batch, seq_len, heads, dim = x.shape
    frames = seq_len // (height * width)
    return x.reshape(batch, frames, height, width, heads, dim)


def _hard_c4_blocks_to_video(x):
    batch, frames, height, width, heads, dim = x.shape
    return x.reshape(batch, frames * height * width, heads, dim)


def _hard_c4_attention(
    query,
    key,
    value,
    *,
    h,
    w,
    sm_scale,
    query_outer_chunk,
    hard_route_backend,
):
    """Execute main's maintained hard-C4 PyTorch/Triton paths."""

    h = int(h)
    w = int(w)
    query_blocks = _hard_c4_video_to_blocks(query, h, w)
    key_blocks = _hard_c4_video_to_blocks(key, h, w)
    value_blocks = _hard_c4_video_to_blocks(value, h, w)
    backend = (hard_route_backend or "sdpa").lower()

    output = torch.empty_like(query_blocks)
    for start in range(0, query_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, query_blocks.size(1))
        query_chunk = query_blocks[:, start:end]
        output_slice = output[:, start:end]
        if backend == HARD_C4_BACKEND:
            chunk = hard_route_attention_triton(
                query_chunk,
                key_blocks,
                value_blocks,
                sm_scale,
                output=output_slice,
            )
        elif backend == HARD_C4_FUSED_BACKEND:
            chunk = hard_route_attention_triton_fused(
                query_chunk,
                key_blocks,
                value_blocks,
                sm_scale,
                output=output_slice,
            )
        else:
            chunk = hard_route_attention_pytorch(
                query_chunk,
                key_blocks,
                value_blocks,
                sm_scale,
                query_outer_offset=start,
            )
        if chunk is not output_slice:
            output_slice.copy_(chunk)
    return _hard_c4_blocks_to_video(output)


def _causal_one_token_attention(
    query,
    key,
    value,
    *,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale,
    q_init,
    query_outer_chunk,
):
    """Execute cache-compatible one-token GridMerge with unequal Q and K/V."""

    if key.size(1) != value.size(1):
        raise ValueError("causal one-token GridMerge expects K/V lengths to match.")
    if query.size(0) != key.size(0) or query.shape[2:] != key.shape[2:]:
        raise ValueError(
            "causal one-token GridMerge expects Q and K/V batch, head, and "
            "head-dimension shapes to match."
        )

    f_tied = int(f_tied)
    h_reduce = int(h_reduce)
    w_reduce = int(w_reduce)
    original_h = int(h)
    original_w = int(w)
    query, padded_h, padded_w, _ = pad_video_tokens_to_multiples(
        query, original_h, original_w, h_reduce, w_reduce
    )
    key, _, _, key_valid = pad_video_tokens_to_multiples(
        key, original_h, original_w, h_reduce, w_reduce
    )
    value, _, _, _ = pad_video_tokens_to_multiples(
        value, original_h, original_w, h_reduce, w_reduce
    )

    query_blocks = general_video_to_blocks(
        query, f_tied, h_reduce, w_reduce, padded_h, padded_w
    )
    key_blocks = general_video_to_blocks(
        key, f_tied, h_reduce, w_reduce, padded_h, padded_w
    )
    value_blocks = general_video_to_blocks(
        value, f_tied, h_reduce, w_reduce, padded_h, padded_w
    )
    key_valid_blocks = None
    if key_valid is not None:
        key_valid_blocks = general_video_to_blocks(
            key_valid, f_tied, h_reduce, w_reduce, padded_h, padded_w
        )[..., 0, 0]

    output_blocks = torch.empty_like(query_blocks)
    for start in range(0, query_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, query_blocks.size(1))
        output_blocks[:, start:end] = causal_gridmerge_one(
            query_blocks[:, start:end],
            key_blocks,
            value_blocks,
            sm_scale,
            q_init,
            key_valid=key_valid_blocks,
        )
    output = general_blocks_to_video(
        output_blocks, f_tied, h_reduce, w_reduce
    )
    return crop_video_tokens_from_multiples(
        output, original_h, original_w, padded_h, padded_w
    )


def token_merging_attn(
    q,
    k,
    v,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    q_init=None,
    query_outer_chunk=None,
    layout=LAYOUT_HARD_ROUTE,
    num_merge_tokens=4,
    seed_mode="local_with_ith",
    seed=0,
    hard_route_backend="sdpa",
    hard_route_routing_backend=None,
):
    """Dispatch the ordinary-Wan C4 and causal Self-Forcing C1 paths.

    Ordinary Wan supports main's hard-C4 backends:
      - hard_route_backend="sdpa": readable PyTorch reference.
      - hard_route_backend="hard_c4": optimized Triton path.
      - hard_route_backend="hard_c4_fused": experimental fused Triton path
        retained for manual optimization.

    Causal Self-Forcing uses one-token GridMerge because its current-block Q is
    shorter than its visible cached K/V prefix.
    """

    del seed_mode, seed, hard_route_routing_backend
    if (layout or LAYOUT_HARD_ROUTE) != LAYOUT_HARD_ROUTE:
        raise ValueError("the merged Wan implementation supports only hard_route.")
    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))
    num_merge_tokens = int(num_merge_tokens)

    if num_merge_tokens == 1:
        return _causal_one_token_attention(
            q,
            k,
            v,
            f_tied=f_tied,
            h_reduce=h_reduce,
            w_reduce=w_reduce,
            h=h,
            w=w,
            sm_scale=float(sm_scale),
            q_init=q_init,
            query_outer_chunk=query_outer_chunk,
        )
    if num_merge_tokens != 4:
        raise ValueError(
            "Wan supports num_merge_tokens=4 for hard C4 or "
            "num_merge_tokens=1 for causal Self-Forcing."
        )
    return _hard_c4_attention(
        q,
        k,
        v,
        h=h,
        w=w,
        sm_scale=float(sm_scale),
        query_outer_chunk=query_outer_chunk,
        hard_route_backend=hard_route_backend,
    )


def token_merging_attn_with_kv_cache(
    q,
    k_cache,
    v_cache,
    new_k,
    new_v,
    start_idx,
    end_idx,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    q_init=None,
    query_outer_chunk=None,
    layout=LAYOUT_HARD_ROUTE,
    num_merge_tokens=1,
    seed_mode="random",
    seed=0,
    hard_route_backend="sdpa",
    hard_route_routing_backend="sdpa",
):
    """Overwrite the current cache block and attend only to its visible prefix."""

    if (layout or LAYOUT_HARD_ROUTE) != LAYOUT_HARD_ROUTE or int(
        num_merge_tokens
    ) != 1:
        raise ValueError(
            "causal cached token merging supports only layout='hard_route' "
            "with num_merge_tokens=1."
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes.")
    if new_k.shape != new_v.shape:
        raise ValueError("new_k and new_v must have identical shapes.")
    if q.size(0) != k_cache.size(0) or q.shape[2:] != k_cache.shape[2:]:
        raise ValueError(
            "Q and cached K/V batch, head, and head-dimension shapes must match."
        )
    if new_k.size(0) != k_cache.size(0) or new_k.shape[2:] != k_cache.shape[2:]:
        raise ValueError(
            "new and cached K/V batch, head, and head-dimension shapes must match."
        )

    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if not 0 <= start_idx <= end_idx <= k_cache.size(1):
        raise ValueError(
            f"invalid cache interval [{start_idx}, {end_idx}) for capacity "
            f"{k_cache.size(1)}."
        )
    if end_idx - start_idx != new_k.size(1):
        raise ValueError(
            "cache interval length must equal the number of new K/V tokens."
        )

    k_cache[:, start_idx:end_idx].copy_(new_k)
    v_cache[:, start_idx:end_idx].copy_(new_v)
    return token_merging_attn(
        q,
        k_cache[:, :end_idx],
        v_cache[:, :end_idx],
        f_tied=f_tied,
        h_reduce=h_reduce,
        w_reduce=w_reduce,
        h=h,
        w=w,
        sm_scale=sm_scale,
        q_init=q_init,
        query_outer_chunk=query_outer_chunk,
        layout=layout,
        num_merge_tokens=num_merge_tokens,
        seed_mode=seed_mode,
        seed=seed,
        hard_route_backend=hard_route_backend,
        hard_route_routing_backend=hard_route_routing_backend,
    )


__all__ = [
    "LAYOUT_HARD_ROUTE",
    "token_merging_attn",
    "token_merging_attn_with_kv_cache",
]
