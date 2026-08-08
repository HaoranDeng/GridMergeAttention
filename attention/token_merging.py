import torch

from .common import (
    blocks_to_video,
    crop_video_tokens_from_multiples,
    pad_video_tokens_to_multiples,
    video_to_blocks,
)
from .gridmerge_one import gridmerge_one
from .gridmerge_one_transpose import gridmerge_one_transpose
from .hard_route import hard_route_attention
from .hard_route_transpose import hard_route_attention_transpose
from .select_route import select_route_attention
from .select_route_transpose import select_route_attention_transpose


LAYOUT_HARD_ROUTE = "hard_route"
LAYOUT_SELECT_ROUTE = "select_route"
LAYOUT_HARD_ROUTE_TRANSPOSE = "hard_route_transpose"
LAYOUT_SELECT_ROUTE_TRANSPOSE = "select_route_transpose"

SUPPORTED_LAYOUTS = {
    LAYOUT_HARD_ROUTE,
    LAYOUT_SELECT_ROUTE,
    LAYOUT_HARD_ROUTE_TRANSPOSE,
    LAYOUT_SELECT_ROUTE_TRANSPOSE,
}
HARD_ROUTE_LAYOUTS = {
    LAYOUT_HARD_ROUTE,
    LAYOUT_HARD_ROUTE_TRANSPOSE,
}

LAYOUT_ALIASES = {
    # Backward-compatible names used by older configs.
    "query_column_key_row": LAYOUT_HARD_ROUTE,
    "query_column_key_row_select": LAYOUT_SELECT_ROUTE,
    "query_row_key_column": LAYOUT_HARD_ROUTE_TRANSPOSE,
    "query_row_key_column_select": LAYOUT_SELECT_ROUTE_TRANSPOSE,
}
ONE_TOKEN_IMPLS = {
    LAYOUT_HARD_ROUTE: gridmerge_one,
    LAYOUT_SELECT_ROUTE: gridmerge_one,
    LAYOUT_HARD_ROUTE_TRANSPOSE: gridmerge_one_transpose,
    LAYOUT_SELECT_ROUTE_TRANSPOSE: gridmerge_one_transpose,
}
MULTI_TOKEN_IMPLS = {
    LAYOUT_HARD_ROUTE: hard_route_attention,
    LAYOUT_SELECT_ROUTE: select_route_attention,
    LAYOUT_HARD_ROUTE_TRANSPOSE: hard_route_attention_transpose,
    LAYOUT_SELECT_ROUTE_TRANSPOSE: select_route_attention_transpose,
}


def normalize_layout(layout):
    layout = layout or LAYOUT_HARD_ROUTE
    return LAYOUT_ALIASES.get(layout, layout)


def _local_attention_impl(layout, num_merge_tokens):
    impls = MULTI_TOKEN_IMPLS if num_merge_tokens > 1 else ONE_TOKEN_IMPLS
    return impls[layout]


def _can_write_directly_to_output(layout, hard_route_routing_backend):
    return (
        layout == LAYOUT_HARD_ROUTE
        and str(hard_route_routing_backend).startswith("triton_direct_")
    )


def _run_attention_chunk(
    attention_impl,
    layout,
    q_chunk,
    k_blocks,
    v_blocks,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
    q_init,
    key_valid_blocks,
    hard_route_backend,
    hard_route_routing_backend,
    out_slice,
):
    if num_merge_tokens == 1:
        return attention_impl(
            q_chunk,
            k_blocks,
            v_blocks,
            scale,
            q_init,
            key_valid=key_valid_blocks,
        )

    if layout in HARD_ROUTE_LAYOUTS:
        kwargs = {}
        if _can_write_directly_to_output(layout, hard_route_routing_backend):
            kwargs["output"] = out_slice

        return attention_impl(
            q_chunk,
            k_blocks,
            v_blocks,
            scale,
            num_merge_tokens,
            seed_mode,
            seed,
            query_outer_offset,
            hard_route_backend,
            hard_route_routing_backend,
            **kwargs,
        )

    return attention_impl(
        q_chunk,
        k_blocks,
        v_blocks,
        scale,
        num_merge_tokens,
        seed_mode,
        seed,
        query_outer_offset,
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
    num_merge_tokens=1,
    seed_mode="random",
    seed=0,
    hard_route_backend="sdpa",
    hard_route_routing_backend="sdpa",
):
    """Grid-based token-merging attention for Wan video self-attention."""
    # Incoming q/k/v from Wan self-attention:
    #   q/k/v: [batch, sequence, head, dim]
    # Default 480p/81-frame Wan:
    #   sequence = frame * height * width = 21 * 30 * 52 = 32760.
    # After video_to_blocks:
    #   q/k/v: [batch, 21, 30, 52, head, dim]
    if q.size(1) != k.size(1) or k.size(1) != v.size(1):
        raise ValueError("token-merging attention expects self-attention q/k/v lengths to match.")

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))
    layout = normalize_layout(layout)
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            "layout must be 'hard_route', 'select_route', "
            "'hard_route_transpose', or 'select_route_transpose', "
            f"got {layout!r}."
        )

    num_merge_tokens = int(num_merge_tokens)
    if num_merge_tokens < 1:
        raise ValueError(f"num_merge_tokens must be >= 1, got {num_merge_tokens!r}.")

    original_h = int(h)
    original_w = int(w)
    needs_padding = bool(original_h % h_reduce or original_w % w_reduce)
    if needs_padding and num_merge_tokens > 1:
        raise ValueError(
            "non-divisible height/width padding currently supports only "
            "one-token MonarchRT attention."
        )

    q, h, w, valid_tokens = pad_video_tokens_to_multiples(q, h, w, h_reduce, w_reduce)
    k, _, _, _ = pad_video_tokens_to_multiples(k, original_h, original_w, h_reduce, w_reduce)
    v, _, _, _ = pad_video_tokens_to_multiples(v, original_h, original_w, h_reduce, w_reduce)

    q_blocks = video_to_blocks(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = video_to_blocks(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = video_to_blocks(v, f_tied, h_reduce, w_reduce, h, w)
    key_valid_blocks = None
    if valid_tokens is not None:
        key_valid_blocks = video_to_blocks(valid_tokens, f_tied, h_reduce, w_reduce, h, w)
        key_valid_blocks = key_valid_blocks[..., 0, 0]
    attention_impl = _local_attention_impl(layout, num_merge_tokens)

    out = torch.empty_like(q_blocks)
    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        q_chunk = q_blocks[:, start:end]
        out_slice = out[:, start:end]
        chunk = _run_attention_chunk(
            attention_impl,
            layout,
            q_chunk,
            k_blocks,
            v_blocks,
            sm_scale,
            num_merge_tokens,
            seed_mode,
            seed,
            start,
            q_init,
            key_valid_blocks,
            hard_route_backend,
            hard_route_routing_backend,
            out_slice,
        )
        if chunk is not out_slice:
            out_slice.copy_(chunk)

    out = blocks_to_video(out, f_tied, h_reduce, w_reduce)
    return crop_video_tokens_from_multiples(out, original_h, original_w, h, w)


__all__ = [
    "token_merging_attn",
    "normalize_layout",
    "LAYOUT_HARD_ROUTE",
    "LAYOUT_SELECT_ROUTE",
    "LAYOUT_HARD_ROUTE_TRANSPOSE",
    "LAYOUT_SELECT_ROUTE_TRANSPOSE",
]
