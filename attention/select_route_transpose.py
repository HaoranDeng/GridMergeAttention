import torch
from einops import rearrange

from .common import (
    gather_seed_queries_transpose,
    scale_qk,
    seed_columns_by_mode,
    soft_merge_weights,
    validate_block_seed_mode,
)


def select_route_attention_transpose(
    q,
    k,
    v,
    scale,
    num_merge_tokens,
    seed_mode,
    seed,
    query_outer_offset,
):
    # Transposed selective path. Each seed compresses all K key rows in a fixed
    # (a, f, i, l) block; each query column picks the best compressed KV via
    # q_j dot K_c + H(alpha_c).
    validate_block_seed_mode(seed_mode, "transposed selective token merging")

    q, k = scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_cols = seed_columns_by_mode(
        query_outer,
        key_outer,
        q.size(2),
        k.size(3),
        q.size(3),
        num_merge_tokens,
        seed,
        seed_mode,
    )
    seed_queries = gather_seed_queries_transpose(q, seed_cols)

    merge_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, k)
    merge_weights, merge_logprob = soft_merge_weights(merge_logits, k.dtype)

    merged_keys = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, k)
    merged_values = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, v)

    entropy = -(merge_weights.float() * merge_logprob).sum(dim=-1).unsqueeze(-1)

    candidate_scores = torch.einsum("bafilchd,baijhd->bhafilcj", merged_keys, q).float()
    candidate_scores = candidate_scores + entropy
    selected_scores, selected_seed = candidate_scores.max(dim=-2)

    routing_weights = rearrange(selected_scores, "b h a f i l j -> b h a i j (f l)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a i j (f l) -> b h a f i l j",
        f=k.size(1),
        l=k.size(3),
    )

    selected_seed = torch.nn.functional.one_hot(
        selected_seed,
        num_classes=num_merge_tokens,
    ).movedim(-1, -2).to(q.dtype)
    routing_weights = routing_weights.unsqueeze(-2) * selected_seed

    return torch.einsum("bhafilcj,bafilche->baijhe", routing_weights, merged_values)


__all__ = ["select_route_attention_transpose"]
