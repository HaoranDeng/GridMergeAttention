import torch
from einops import rearrange

from .common import (
    gather_seed_queries,
    random_seed_rows,
    scale_qk,
    soft_merge_weights,
    validate_block_seed_mode,
)


def select_route_attention(
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
    # Each seed compresses all L keys in a fixed (a, f, j, k) block. Each
    # query then selects the seed whose compressed distribution maximizes
    # q_i dot K_c + H(alpha_c), the softmax variational objective.
    include_ith = validate_block_seed_mode(seed_mode, "selective token merging")

    q, k = scale_qk(q, k, scale)

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + q.size(1),
        device=q.device,
    )
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_rows = random_seed_rows(
        query_outer,
        key_outer,
        k.size(2),
        q.size(3),
        q.size(2),
        num_merge_tokens,
        seed,
        include_ith=include_ith,
    )
    seed_queries = gather_seed_queries(q, seed_rows)

    merge_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, k)
    merge_weights, merge_logprob = soft_merge_weights(merge_logits, k.dtype)

    merged_keys = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, k)
    merged_values = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, v)

    entropy = -(merge_weights.float() * merge_logprob).sum(dim=-1)
    entropy = rearrange(entropy, "b h a f k j c -> b h a f j k c 1")

    candidate_scores = torch.einsum("bafjkchd,baijhd->bhafjkci", merged_keys, q).float()
    candidate_scores = candidate_scores + entropy
    selected_scores, selected_seed = candidate_scores.max(dim=-2)

    routing_weights = rearrange(selected_scores, "b h a f j k i -> b h a j i (f k)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k) -> b h a f j k i",
        f=k.size(1),
        k=k.size(2),
    )

    selected_seed = torch.nn.functional.one_hot(
        selected_seed,
        num_classes=num_merge_tokens,
    ).movedim(-1, -2).to(q.dtype)
    routing_weights = routing_weights.unsqueeze(-2) * selected_seed

    return torch.einsum("bhafjkci,bafjkche->baijhe", routing_weights, merged_values)


__all__ = ["select_route_attention"]
