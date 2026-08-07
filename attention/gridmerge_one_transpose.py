import torch
from einops import rearrange

from .common import masked_soft_merge_weights, merge_query_column, scale_qk, soft_merge_weights


def gridmerge_one_transpose(q, k, v, scale, q_init, key_valid=None):
    # Symmetric layout to gridmerge_one:
    #   q:   [B, A, I, J, H, D]
    #   k/v: [B, F, K, L, H, D]
    # A single fixed (a, f, i, l) tile is 52 queries x 30 keys by default:
    # fixed query row i routes across query columns j, while fixed key column l
    # merges key rows k.
    q, k = scale_qk(q, k, scale)

    # 1. For each key column l, use the selected query column to merge key rows k.
    selected_q = merge_query_column(q, k.size(3), q_init)
    merge_logits = torch.einsum("bailhd,bfklhd->bhafilk", selected_q, k)
    compressed_valid = None
    if key_valid is None:
        merge_weights, merge_logprob = soft_merge_weights(merge_logits, k.dtype)
    else:
        key_valid = key_valid.to(torch.bool)
        merge_valid = key_valid.permute(0, 1, 3, 2)[:, None, None, :, None, :, :]
        merge_weights, merge_logprob = masked_soft_merge_weights(merge_logits, merge_valid, k.dtype)
        compressed_valid = key_valid.any(dim=2)

    merged_keys = torch.einsum("bhafilk,bfklhd->bafilhd", merge_weights, k)
    merged_values = torch.einsum("bhafilk,bfklhe->bafilhe", merge_weights, v)

    # 2. Route each query column j to merged tokens indexed by (f, l).
    routing_bias = (merge_weights * merge_logprob).sum(dim=-1, keepdim=True)
    routing_logits = torch.einsum("bafilhd,baijhd->bhafilj", merged_keys, q)
    routing_logits = routing_logits - routing_bias
    if compressed_valid is not None:
        routing_valid = compressed_valid[:, None, None, :, None, :, None]
        routing_logits = routing_logits.float().masked_fill(
            ~routing_valid,
            torch.finfo(torch.float32).min,
        )

    routing_weights = rearrange(routing_logits, "b h a f i l j -> b h a i j (f l)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a i j (f l) -> b h a f i l j",
        f=k.size(1),
        l=k.size(3),
    )

    return torch.einsum("bhafilj,bafilhe->baijhe", routing_weights, merged_values)


__all__ = ["gridmerge_one_transpose"]
