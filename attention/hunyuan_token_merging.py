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

from .common import (
    blocks_to_video,
    crop_video_tokens_from_multiples,
    masked_soft_merge_weights,
    merge_query_row,
    pad_video_tokens_to_multiples,
    scale_qk,
    video_to_blocks,
)


_LAYOUT_ALIASES = {"query_column_key_row": "hard_route"}


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
):
    """Run Hunyuan joint attention with GridMerge on video keys only.

    The initial Hunyuan integration intentionally supports only the validated
    one-token, non-transposed GridMerge layout. Other experimental Wan layouts
    need their own joint video/text derivation before they can be enabled.
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
    if layout != "hard_route" or int(num_merge_tokens) != 1:
        raise ValueError(
            "Hunyuan joint token merging currently supports only "
            "layout='hard_route' with num_merge_tokens=1."
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

    # Match PyTorch SDPA's scale exactly by multiplying both Q and K by the
    # square root. This also matches the existing Wan GridMerge implementation.
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
