"""One-token GridMerge retained for causal Self-Forcing KV-cache attention."""

from __future__ import annotations

import torch
from einops import rearrange

from ._grid_utils import (
    masked_soft_merge_weights,
    merge_query_row,
    scale_qk,
    soft_merge_weights,
)


def causal_gridmerge_one(query, key, value, scale, q_init, key_valid=None):
    """Merge key columns to one token and route current-block queries."""

    query, key = scale_qk(query, key, scale)
    selected_query = merge_query_row(query, key.size(2), q_init)
    merge_logits = torch.einsum(
        "bakjhd,bfklhd->bhafkjl", selected_query, key
    )
    compressed_valid = None
    if key_valid is None:
        merge_weights, merge_logprob = soft_merge_weights(
            merge_logits, key.dtype
        )
    else:
        key_valid = key_valid.to(torch.bool)
        merge_valid = key_valid[:, None, None, :, :, None, :]
        merge_weights, merge_logprob = masked_soft_merge_weights(
            merge_logits, merge_valid, key.dtype
        )
        compressed_valid = key_valid.any(dim=-1)

    merged_keys = torch.einsum(
        "bhafkjl,bfklhd->bafjkhd", merge_weights, key
    )
    merged_values = torch.einsum(
        "bhafkjl,bfklhe->bafjkhe", merge_weights, value
    )
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1, keepdim=True)
    routing_bias = routing_bias.transpose(-2, -3)
    routing_logits = torch.einsum(
        "bafjkhd,baijhd->bhafjki", merged_keys, query
    )
    routing_logits = routing_logits - routing_bias
    if compressed_valid is not None:
        routing_valid = compressed_valid[:, None, None, :, None, :, None]
        routing_logits = routing_logits.float().masked_fill(
            ~routing_valid, torch.finfo(torch.float32).min
        )

    routing_weights = rearrange(
        routing_logits, "b h a f j k i -> b h a j i (f k)"
    )
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(
        query.dtype
    )
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k) -> b h a f j k i",
        f=key.size(1),
        k=key.size(2),
    )
    return torch.einsum(
        "bhafjki,bafjkhe->baijhe", routing_weights, merged_values
    )


__all__ = ["causal_gridmerge_one"]
