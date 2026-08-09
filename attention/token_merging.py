import torch

from .common import (
    blocks_to_video,
    crop_video_tokens_from_multiples,
    pad_video_tokens_to_multiples,
    video_to_blocks,
)
from .hard_route import hard_route_attention


LAYOUT_HARD_ROUTE = "hard_route"
LAYOUT_ALIASES = {"query_column_key_row": LAYOUT_HARD_ROUTE}


def normalize_layout(layout):
    return LAYOUT_ALIASES.get(layout or LAYOUT_HARD_ROUTE, layout or LAYOUT_HARD_ROUTE)


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
    hard_route_routing_backend="sdpa",
):
    """Hard C4 GridMerge attention for Wan video self-attention.

    The two maintained paths are:
      - hard_route_backend="sdpa": readable PyTorch reference.
      - hard_route_backend="hard_c4": optimized Triton path.
    """
    del q_init  # Kept in the public signature for older caller compatibility.
    if q.size(1) != k.size(1) or k.size(1) != v.size(1):
        raise ValueError("hard C4 attention expects self-attention q/k/v lengths to match.")

    layout = normalize_layout(layout)
    if layout != LAYOUT_HARD_ROUTE:
        raise ValueError(f"only layout='hard_route' is supported, got {layout!r}.")
    if int(num_merge_tokens) != 4:
        raise ValueError(f"hard C4 attention requires num_merge_tokens=4, got {num_merge_tokens!r}.")
    if (int(f_tied), int(h_reduce), int(w_reduce)) != (1, 1, 1):
        raise ValueError("the maintained hard C4 path requires f_tied=h_reduce=w_reduce=1.")

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))

    original_h = int(h)
    original_w = int(w)
    if original_h % h_reduce or original_w % w_reduce:
        raise ValueError("hard C4 attention requires height/width divisible by h_reduce/w_reduce.")

    q, h, w, _ = pad_video_tokens_to_multiples(q, h, w, h_reduce, w_reduce)
    k, _, _, _ = pad_video_tokens_to_multiples(k, original_h, original_w, h_reduce, w_reduce)
    v, _, _, _ = pad_video_tokens_to_multiples(v, original_h, original_w, h_reduce, w_reduce)

    q_blocks = video_to_blocks(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = video_to_blocks(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = video_to_blocks(v, f_tied, h_reduce, w_reduce, h, w)

    out = torch.empty_like(q_blocks)
    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        q_chunk = q_blocks[:, start:end]
        out_slice = out[:, start:end]
        kwargs = {}
        if str(hard_route_routing_backend).startswith("triton_direct_"):
            kwargs["output"] = out_slice

        chunk = hard_route_attention(
            q_chunk,
            k_blocks,
            v_blocks,
            sm_scale,
            4,
            seed_mode,
            seed,
            start,
            hard_route_backend,
            hard_route_routing_backend,
            **kwargs,
        )
        if chunk is not out_slice:
            out_slice.copy_(chunk)

    out = blocks_to_video(out, f_tied, h_reduce, w_reduce)
    return crop_video_tokens_from_multiples(out, original_h, original_w, h, w)


__all__ = ["token_merging_attn", "normalize_layout", "LAYOUT_HARD_ROUTE"]
