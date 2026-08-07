import torch
from einops import rearrange

from .common import masked_soft_merge_weights, merge_query_row, scale_qk, soft_merge_weights


def gridmerge_one(q, k, v, scale, q_init, key_valid=None):
    # Dimension letters used below:
    #   b batch, a query_outer, f key_outer, i query_row, j query_column,
    #   k key_row, l key_column, h head, d/e head_dim.
    #
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A single fixed (a, f, j, k) tile is 30 queries x 52 keys by default.
    q, k = scale_qk(q, k, scale)

    # 1. For each key row k, use the selected query row to merge key columns l.
    selected_q = merge_query_row(q, k.size(2), q_init)
    merge_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", selected_q, k)
    compressed_valid = None
    if key_valid is None:
        merge_weights, merge_logprob = soft_merge_weights(merge_logits, k.dtype)
    else:
        key_valid = key_valid.to(torch.bool)
        merge_valid = key_valid[:, None, None, :, :, None, :]
        merge_weights, merge_logprob = masked_soft_merge_weights(merge_logits, merge_valid, k.dtype)
        compressed_valid = key_valid.any(dim=-1)

    merged_keys = torch.einsum("bhafkjl,bfklhd->bafjkhd", merge_weights, k)
    merged_values = torch.einsum("bhafkjl,bfklhe->bafjkhe", merge_weights, v)

    # 2. Route each query row i to merged tokens indexed by (f, k).
    # The merge-logprob correction keeps this exactly equivalent to the old
    # one-step implementation while writing it as token merging.
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1, keepdim=True)
    routing_bias = routing_bias.transpose(-2, -3)
    routing_logits = torch.einsum("bafjkhd,baijhd->bhafjki", merged_keys, q)
    routing_logits = routing_logits - routing_bias
    if compressed_valid is not None:
        routing_valid = compressed_valid[:, None, None, :, None, :, None]
        routing_logits = routing_logits.float().masked_fill(
            ~routing_valid,
            torch.finfo(torch.float32).min,
        )

    routing_weights = rearrange(routing_logits, "b h a f j k i -> b h a j i (f k)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k) -> b h a f j k i",
        f=k.size(1),
        k=k.size(2),
    )

    return torch.einsum("bhafjki,bafjkhe->baijhe", routing_weights, merged_values)


__all__ = ["gridmerge_one"]
