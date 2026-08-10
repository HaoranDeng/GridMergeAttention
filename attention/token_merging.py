import torch

from .hard_route_pytorch import hard_route_attention_pytorch
from .hard_route_triton import (
    HARD_C4_BACKEND,
    HARD_C4_FUSED_BACKEND,
    hard_route_attention_triton,
    hard_route_attention_triton_fused,
)


LAYOUT_HARD_ROUTE = "hard_route"


def video_to_blocks(x, height, width):
    """[B, F*H*W, heads, dim] -> [B, F, H, W, heads, dim]."""
    batch, seq_len, heads, dim = x.shape
    frames = seq_len // (height * width)
    return x.reshape(batch, frames, height, width, heads, dim)


def blocks_to_video(x):
    """[B, F, H, W, heads, dim] -> [B, F*H*W, heads, dim]."""
    batch, frames, height, width, heads, dim = x.shape
    return x.reshape(batch, frames * height * width, heads, dim)


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
):
    """Hard C4 GridMerge attention for Wan video self-attention.

    The two maintained paths are:
      - hard_route_backend="sdpa": readable PyTorch reference.
      - hard_route_backend="hard_c4": optimized Triton path.
      - hard_route_backend="hard_c4_fused": experimental fused Triton path.
    """
    del f_tied, h_reduce, w_reduce, q_init, layout, num_merge_tokens, seed_mode, seed

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))
    h = int(h)
    w = int(w)

    q_blocks = video_to_blocks(q, h, w)
    k_blocks = video_to_blocks(k, h, w)
    v_blocks = video_to_blocks(v, h, w)

    backend = (hard_route_backend or "sdpa").lower()
    out = torch.empty_like(q_blocks)
    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        q_chunk = q_blocks[:, start:end]
        out_slice = out[:, start:end]

        if backend == HARD_C4_BACKEND:
            chunk = hard_route_attention_triton(
                q_chunk,
                k_blocks,
                v_blocks,
                sm_scale,
                output=out_slice,
            )
        elif backend == HARD_C4_FUSED_BACKEND:
            chunk = hard_route_attention_triton_fused(
                q_chunk,
                k_blocks,
                v_blocks,
                sm_scale,
                output=out_slice,
            )
        else:
            chunk = hard_route_attention_pytorch(
                q_chunk,
                k_blocks,
                v_blocks,
                sm_scale,
                query_outer_offset=start,
            )

        if chunk is not out_slice:
            out_slice.copy_(chunk)

    return blocks_to_video(out)


__all__ = ["token_merging_attn", "LAYOUT_HARD_ROUTE"]
