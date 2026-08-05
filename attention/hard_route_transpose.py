import torch
from einops import rearrange

from .common import (
    gather_seed_queries_transpose,
    hard_route_merge_weights,
    random_seed_columns,
    scale_qk,
    validate_block_seed_mode,
)


def hard_route_attention_transpose(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
):
    # q:   [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    # A fixed (a, f, i, l) block has J queries x K keys.
    include_ith = validate_block_seed_mode(seed_mode, "transposed multi-token merging")
    if num_merge_tokens > q.size(3):
        raise ValueError("num_merge_tokens cannot exceed query columns per block.")

    q, k = scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_cols = random_seed_columns(
        query_outer,
        key_outer,
        q.size(2),
        k.size(3),
        q.size(3),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
    )
    seed_queries = gather_seed_queries_transpose(q, seed_cols)

    seed_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, k).float()
    merge_weights, merge_logprob, cluster_has_keys = hard_route_merge_weights(seed_logits, k.dtype)

    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, k)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, v)

    routing_bias = (merge_weights * merge_logprob).sum(dim=-1)
    routing_bias = routing_bias.unsqueeze(-1)

    routing_logits = torch.einsum("bafilchd,baijhd->bhafilcj", merged_keys, q)
    routing_logits = routing_logits - routing_bias
    routing_logits = routing_logits.masked_fill(
        ~cluster_has_keys.unsqueeze(-1),
        torch.finfo(routing_logits.dtype).min,
    )

    routing_weights = rearrange(routing_logits, "b h a f i l c j -> b h a i j (f l c)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a i j (f l c) -> b h a f i l c j",
        f=k.size(1),
        l=k.size(3),
        c=num_merge_tokens,
    )

    return torch.einsum("bhafilcj,bafilche->baijhe", routing_weights, merged_values)


__all__ = ["hard_route_attention_transpose"]
