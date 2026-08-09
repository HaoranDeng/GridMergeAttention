import torch

from .common import blocks_to_video, video_to_blocks
from .hard_route import hard_route_attention


LAYOUT_HARD_ROUTE = "hard_route"


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
    del f_tied, h_reduce, w_reduce, q_init, layout, num_merge_tokens, seed

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))
    h = int(h)
    w = int(w)

    q_blocks = video_to_blocks(q, h, w)
    k_blocks = video_to_blocks(k, h, w)
    v_blocks = video_to_blocks(v, h, w)

    out = torch.empty_like(q_blocks)
    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        q_chunk = q_blocks[:, start:end]
        out_slice = out[:, start:end]

        chunk = hard_route_attention(
            q_chunk,
            k_blocks,
            v_blocks,
            sm_scale,
            4,
            seed_mode,
            0,
            start,
            hard_route_backend,
            hard_route_routing_backend,
            output=out_slice,
        )
        if chunk is not out_slice:
            out_slice.copy_(chunk)

    return blocks_to_video(out)


__all__ = ["token_merging_attn", "LAYOUT_HARD_ROUTE"]
