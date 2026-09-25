"""Token-Merging joint attention for Diffusers HunyuanVideo.

HunyuanVideo attends jointly over video and text tokens.  The video-query path
below applies the repository's one-token GridMerge rule to video keys while
keeping text keys exact.  Both key sets participate in one shared softmax, so
their relative attention mass is preserved.  Text queries use exact dense
attention over the original video and text keys because the text sequence is
small and is recurrently updated by Hunyuan's transformer blocks.

All public tensors use ``[batch, sequence, heads, head_dim]`` layout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from ._grid_utils import (
    blocks_to_video,
    crop_video_tokens_from_multiples,
    gather_seed_queries,
    masked_soft_merge_weights,
    merge_query_row,
    pad_video_tokens_to_multiples,
    random_seed_rows,
    scale_qk,
    validate_block_seed_mode,
    video_to_blocks,
)


_LAYOUT_ALIASES = {"query_column_key_row": "hard_route"}


def _hard_route_merge_weights(seed_logits, dtype):
    """Hard-assign keys, preserving the established C2/C4/C6 empty-seed rule."""

    num_seeds = seed_logits.size(-2)
    num_keys = seed_logits.size(-1)
    assignments = seed_logits.argmax(dim=-2)
    assignment_mask = torch.nn.functional.one_hot(
        assignments, num_classes=num_seeds
    ).movedim(-1, -2).bool()

    has_seed = assignment_mask.any(dim=-1)
    key_index = torch.arange(num_keys, device=seed_logits.device)
    key_index = key_index.view(*((1,) * has_seed.ndim), num_keys)
    # Match the branch's pre-merge hard-route behavior: an empty cluster shares
    # key zero so every requested compressed token remains numerically defined.
    force_mask = (~has_seed).unsqueeze(-1) & (key_index == 0)
    assignment_mask = assignment_mask | force_mask

    masked_logits = seed_logits.masked_fill(
        ~assignment_mask, torch.finfo(seed_logits.dtype).min
    )
    merge_logprob = torch.log_softmax(masked_logits, dim=-1)
    merge_weights = merge_logprob.exp().masked_fill(~assignment_mask, 0).to(dtype)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)
    return merge_weights, merge_logprob, assignment_mask.any(dim=-1)


def _validate_stream(name, query, key, value):
    if query is None or key is None or value is None:
        raise ValueError(f"{name} query, key, and value must all be provided.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"{name} query, key, and value must have [B, S, H, D] layout."
        )
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(f"{name} query, key, and value shapes must match.")


def _normalize_key_mask(attention_mask, batch, key_length, device):
    """Normalize Hunyuan's bool ``[B, 1, 1, K]`` mask to ``[B, K]``."""

    if attention_mask is None:
        return torch.ones(batch, key_length, dtype=torch.bool, device=device)
    if attention_mask.dtype != torch.bool:
        raise TypeError(
            "Hunyuan Token Merging expects the transformer's boolean key mask; "
            f"got dtype={attention_mask.dtype}."
        )

    mask = attention_mask.to(device=device)
    if mask.ndim == 4 and mask.shape[1:3] == (1, 1):
        mask = mask[:, 0, 0]
    elif mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape [B, K], [B, 1, K], or [B, 1, 1, K]."
        )

    if mask.shape[-1] != key_length:
        raise ValueError(
            f"attention_mask key length {mask.shape[-1]} does not match "
            f"video+text length {key_length}."
        )
    if mask.shape[0] == 1 and batch != 1:
        mask = mask.expand(batch, -1)
    elif mask.shape[0] != batch:
        raise ValueError(
            f"attention_mask batch {mask.shape[0]} does not match input batch {batch}."
        )
    if not mask.any(dim=-1).all():
        raise ValueError("each batch item must expose at least one attention key.")
    return mask


def _pad_video_key_mask(mask, frames, height, width, padded_height, padded_width):
    if padded_height == height and padded_width == width:
        return mask

    padded = torch.zeros(
        mask.shape[0],
        frames,
        padded_height,
        padded_width,
        dtype=torch.bool,
        device=mask.device,
    )
    padded[:, :, :height, :width] = mask.reshape(
        mask.shape[0], frames, height, width
    )
    return padded.reshape(mask.shape[0], frames * padded_height * padded_width)


def _video_query_chunk_attention(
    query,
    video_key,
    video_value,
    text_key,
    text_value,
    video_key_valid,
    text_key_valid,
    q_init,
):
    """GridMerge video keys and jointly normalize them with exact text keys."""

    # Dimension letters:
    #   a/f: query/key outer block, i/k: query/key row, j/l: column,
    #   h: head, d/e: head dimension, t: text token.
    selected_query = merge_query_row(query, video_key.size(2), q_init)
    merge_logits = torch.einsum(
        "bakjhd,bfklhd->bhafkjl", selected_query, video_key
    )
    merge_valid = video_key_valid[:, None, None, :, :, None, :]
    merge_weights, merge_logprob = masked_soft_merge_weights(
        merge_logits, merge_valid, video_key.dtype
    )

    merged_keys = torch.einsum(
        "bhafkjl,bfklhd->bafjkhd", merge_weights, video_key
    )
    merged_values = torch.einsum(
        "bhafkjl,bfklhe->bafjkhe", merge_weights, video_value
    )

    # The entropy correction is the same correction used by gridmerge_one.
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1, keepdim=True)
    routing_bias = routing_bias.transpose(-2, -3)
    video_logits = torch.einsum(
        "bafjkhd,baijhd->bhafjki", merged_keys, query
    )
    video_logits = video_logits - routing_bias
    video_logits = rearrange(
        video_logits, "b h a f j k i -> b h a j i (f k)"
    )

    # Text keys remain unmerged. Crucially, these logits are concatenated with
    # compressed-video logits before softmax; separate softmaxes would change
    # the relative video/text attention mass.
    text_logits = torch.einsum("baijhd,bthd->bhajit", query, text_key)
    joint_logits = torch.cat((video_logits, text_logits), dim=-1).float()

    compressed_video_valid = video_key_valid.any(dim=-1).flatten(1)
    joint_valid = torch.cat((compressed_video_valid, text_key_valid), dim=-1)
    joint_valid = joint_valid[:, None, None, None, None, :]
    joint_logits = joint_logits.masked_fill(
        ~joint_valid, torch.finfo(torch.float32).min
    )
    joint_weights = torch.softmax(joint_logits, dim=-1).to(query.dtype)

    video_key_count = video_logits.shape[-1]
    video_weights = joint_weights[..., :video_key_count]
    text_weights = joint_weights[..., video_key_count:]

    merged_values = rearrange(
        merged_values, "b a f j k h e -> b h a j (f k) e"
    )
    video_output = torch.einsum(
        "bhajir,bhajre->baijhe", video_weights, merged_values
    )
    text_output = torch.einsum(
        "bhajit,bthe->baijhe", text_weights, text_value
    )
    return video_output + text_output


def _hard_route_video_query_chunk_attention(
    query,
    video_key,
    video_value,
    text_key,
    text_value,
    text_key_valid,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
):
    """Apply multi-token hard routing to video keys with exact text keys."""

    include_ith = validate_block_seed_mode(
        seed_mode, "Hunyuan hard-route token merging"
    )
    fast_with_include = seed_mode == "fast_random_with_ith"
    if num_merge_tokens > query.size(2):
        raise ValueError(
            "num_merge_tokens cannot exceed query rows per Hunyuan block."
        )
    if num_merge_tokens > video_key.size(3):
        raise ValueError(
            "num_merge_tokens cannot exceed key columns per Hunyuan block."
        )

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + query.size(1),
        device=query.device,
    )
    key_outer = torch.arange(video_key.size(1), device=query.device)
    seed_rows = random_seed_rows(
        query_outer,
        key_outer,
        video_key.size(2),
        query.size(3),
        query.size(2),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
        fast_with_include=fast_with_include,
    )
    seed_queries = gather_seed_queries(query, seed_rows)

    seed_logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seed_queries, video_key
    ).float()
    seed_logits = seed_logits * float(scale)
    merge_weights, merge_logprob, compressed_valid = (
        _hard_route_merge_weights(seed_logits, video_key.dtype)
    )

    merged_keys = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", merge_weights, video_key
    )
    merged_values = torch.einsum(
        "bhafkjcl,bfklhe->bafjkche", merge_weights, video_value
    )
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)

    video_logits = torch.einsum(
        "bafjkchd,baijhd->bhafjkci", merged_keys, query
    ).float()
    video_logits = video_logits * float(scale)
    routing_bias = rearrange(
        routing_bias, "b h a f k j c -> b h a f j k c 1"
    )
    video_logits = video_logits - routing_bias
    video_logits = rearrange(
        video_logits, "b h a f j k c i -> b h a j i (f k c)"
    )

    text_logits = torch.einsum(
        "baijhd,bthd->bhajit", query, text_key
    ).float()
    text_logits = text_logits * float(scale)
    joint_logits = torch.cat((video_logits, text_logits), dim=-1)

    compressed_valid = rearrange(
        compressed_valid, "b h a f k j c -> b h a j (f k c)"
    )
    text_valid = text_key_valid[:, None, None, None, :].expand(
        -1,
        query.size(4),
        query.size(1),
        query.size(3),
        -1,
    )
    joint_valid = torch.cat((compressed_valid, text_valid), dim=-1)
    joint_logits = joint_logits.masked_fill(
        ~joint_valid.unsqueeze(-2), torch.finfo(torch.float32).min
    )
    joint_weights = torch.softmax(joint_logits, dim=-1).to(query.dtype)

    video_key_count = video_logits.shape[-1]
    video_weights = joint_weights[..., :video_key_count]
    text_weights = joint_weights[..., video_key_count:]
    merged_values = rearrange(
        merged_values, "b a f j k c h e -> b h a j (f k c) e"
    )
    video_output = torch.einsum(
        "bhajir,bhajre->baijhe", video_weights, merged_values
    )
    text_output = torch.einsum(
        "bhajit,bthe->baijhe", text_weights, text_value
    )
    return video_output + text_output


def hunyuan_token_merging_attn(
    video_q,
    video_k,
    video_v,
    text_q,
    text_k,
    text_v,
    *,
    f_tied=1,
    h_reduce=1,
    w_reduce=1,
    h,
    w,
    attention_mask=None,
    sm_scale=None,
    q_init="ith",
    query_outer_chunk=1,
    layout="hard_route",
    num_merge_tokens=1,
    seed_mode="random_with_ith",
    seed=0,
):
    """Run Hunyuan joint attention with GridMerge on video keys only.

    Hunyuan supports the non-transposed hard-route family. One-token execution
    uses GridMerge; C>1 execution uses deterministic seed rows and hard key
    assignment. Select-route and transposed Wan layouts remain unsupported.
    """

    _validate_stream("video", video_q, video_k, video_v)
    _validate_stream("text", text_q, text_k, text_v)
    if video_q.shape[0] != text_q.shape[0] or video_q.shape[2:] != text_q.shape[2:]:
        raise ValueError(
            "video and text streams must share batch, head, and head-dimension shapes."
        )
    stream_tensors = (video_q, video_k, video_v, text_q, text_k, text_v)
    if any(tensor.dtype != video_q.dtype for tensor in stream_tensors[1:]):
        raise ValueError("all video and text Q/K/V tensors must use one dtype.")

    layout = _LAYOUT_ALIASES.get(layout or "hard_route", layout or "hard_route")
    num_merge_tokens = int(num_merge_tokens)
    if layout != "hard_route" or num_merge_tokens < 1:
        raise ValueError(
            "Hunyuan joint token merging supports only layout='hard_route' "
            "with num_merge_tokens >= 1."
        )

    f_tied = int(f_tied)
    h_reduce = int(h_reduce)
    w_reduce = int(w_reduce)
    height = int(h)
    width = int(w)
    query_outer_chunk = max(1, int(query_outer_chunk or 1))
    if min(f_tied, h_reduce, w_reduce, height, width) <= 0:
        raise ValueError("f_tied, reduction factors, height, and width must be positive.")

    batch, video_length, _, head_dim = video_q.shape
    text_length = text_q.shape[1]
    if video_length % (height * width):
        raise ValueError(
            f"video sequence length {video_length} is not divisible by h*w="
            f"{height * width}."
        )
    frames = video_length // (height * width)
    if frames % f_tied:
        raise ValueError(
            f"video frame grid {frames} must be divisible by f_tied={f_tied}."
        )
    if text_length == 0:
        raise ValueError("Hunyuan joint attention requires at least one text token.")

    key_valid = _normalize_key_mask(
        attention_mask,
        batch=batch,
        key_length=video_length + text_length,
        device=video_q.device,
    )
    original_video_valid = key_valid[:, :video_length]
    text_key_valid = key_valid[:, video_length:]

    needs_spatial_padding = bool(
        height % h_reduce or width % w_reduce
    )
    if num_merge_tokens > 1 and needs_spatial_padding:
        raise ValueError(
            "multi-token Hunyuan hard routing requires height/width divisible "
            "by h_reduce/w_reduce."
        )
    if num_merge_tokens > 1 and not original_video_valid.all():
        raise ValueError(
            "multi-token Hunyuan hard routing expects all video keys to be valid."
        )

    # Preserve the original tensors for exact text-query attention. The video
    # path below creates scaled and, when necessary, spatially padded views.
    dense_video_key = video_k
    dense_video_value = video_v
    dense_text_key = text_k

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    sm_scale = float(sm_scale)

    original_height, original_width = height, width
    video_q, height, width, _ = pad_video_tokens_to_multiples(
        video_q, height, width, h_reduce, w_reduce
    )
    video_k, _, _, _ = pad_video_tokens_to_multiples(
        video_k, original_height, original_width, h_reduce, w_reduce
    )
    video_v, _, _, _ = pad_video_tokens_to_multiples(
        video_v, original_height, original_width, h_reduce, w_reduce
    )
    padded_video_valid = _pad_video_key_mask(
        original_video_valid,
        frames,
        original_height,
        original_width,
        height,
        width,
    )

    if num_merge_tokens == 1:
        # One-token GridMerge distributes the SDPA scale across Q and K.
        video_q, video_k = scale_qk(video_q, video_k, sm_scale)
        _, text_k = scale_qk(text_q, text_k, sm_scale)

    query_blocks = video_to_blocks(
        video_q, f_tied, h_reduce, w_reduce, height, width
    )
    key_blocks = video_to_blocks(
        video_k, f_tied, h_reduce, w_reduce, height, width
    )
    value_blocks = video_to_blocks(
        video_v, f_tied, h_reduce, w_reduce, height, width
    )
    valid_blocks = video_to_blocks(
        padded_video_valid[..., None, None],
        f_tied,
        h_reduce,
        w_reduce,
        height,
        width,
    )[..., 0, 0]

    output_blocks = torch.empty_like(query_blocks)
    for start in range(0, query_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, query_blocks.size(1))
        if num_merge_tokens == 1:
            output_blocks[:, start:end] = _video_query_chunk_attention(
                query_blocks[:, start:end],
                key_blocks,
                value_blocks,
                text_k,
                text_v,
                valid_blocks,
                text_key_valid,
                q_init,
            )
        else:
            output_blocks[:, start:end] = (
                _hard_route_video_query_chunk_attention(
                    query_blocks[:, start:end],
                    key_blocks,
                    value_blocks,
                    text_k,
                    text_v,
                    text_key_valid,
                    sm_scale,
                    num_merge_tokens,
                    seed_mode,
                    int(seed),
                    start,
                )
            )

    video_output = blocks_to_video(
        output_blocks, f_tied, h_reduce, w_reduce
    )
    video_output = crop_video_tokens_from_multiples(
        video_output,
        original_height,
        original_width,
        height,
        width,
    )

    # Text queries remain exact and attend to the original, unpadded key set.
    dense_key = torch.cat((dense_video_key, dense_text_key), dim=1)
    dense_value = torch.cat((dense_video_value, text_v), dim=1)
    text_output = F.scaled_dot_product_attention(
        text_q.transpose(1, 2),
        dense_key.transpose(1, 2),
        dense_value.transpose(1, 2),
        attn_mask=key_valid[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
        scale=sm_scale,
    ).transpose(1, 2)

    return video_output, text_output


__all__ = ["hunyuan_token_merging_attn"]
