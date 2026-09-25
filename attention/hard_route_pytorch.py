"""Plain PyTorch reference for hard-route GridMerge.

This file is the algorithmic source of truth. It intentionally favors the
short math-like implementation over kernel-level tricks:

    seed/key logits -> key-softmax hard assignment -> masked softmax -> merge K/V + bias

The Triton kernels should match these tensor equations.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


FLOAT_NEG_INF = torch.finfo(torch.float32).min

# Shape letters used below:
#   B: batch, H: attention head, D: head dim
#   A: query outer block, F: key outer block
#   I: query rows, J: query columns
#   K: key rows, L: key columns
#   C: merged tokens / seed queries per block. Hard C4 keeps C=4.


_SEED_ROWS_CACHE = {}


def local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows):
    """Choose the four contiguous local seed query rows used by hard C4.

    For key row k, use the clamped window
    ``[start, start + 1, start + 2, start + 3]`` where
    ``start = clamp(k - 1, 0, query_rows - 4)``. This keeps the index-matched
    row inside the seed window while making the four seed rows contiguous.
    """
    device = query_outer.device
    key_row = torch.arange(key_rows, device=device).view(1, 1, key_rows, 1)
    seed_start = (key_row - 1).clamp(0, query_rows - 4)
    seed_offsets = torch.arange(4, device=device).view(1, 1, 1, 1, 4)
    seed_rows = (seed_start.unsqueeze(-1) + seed_offsets).long()
    return seed_rows.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns, 4)


def wide_local_c4_seed_rows(
    query_outer, key_outer, key_rows, query_columns, query_rows
):
    """Match the 14B 720p kernel's wider interior C4 seed layout."""

    device = query_outer.device
    key_row = torch.arange(key_rows, device=device).view(1, 1, key_rows, 1)
    seed_start = (key_row - 1).clamp(0, query_rows - 4)
    contiguous = seed_start.unsqueeze(-1) + torch.arange(
        4, device=device
    ).view(1, 1, 1, 1, 4)
    wide = key_row.unsqueeze(-1) + torch.tensor(
        [-2, 0, 1, 3], device=device
    ).view(1, 1, 1, 1, 4)
    use_wide = ((key_row >= 2) & (key_row <= query_rows - 4)).unsqueeze(-1)
    seed_rows = torch.where(use_wide, wide, contiguous).long()
    return seed_rows.expand(
        query_outer.numel(), key_outer.numel(), key_rows, query_columns, 4
    )


def cached_local_c4_seed_rows(
    device,
    query_outer_offset,
    query_outer_count,
    key_outer_count,
    key_rows,
    query_columns,
    query_rows,
):
    cache_key = (
        str(device),
        int(query_outer_offset),
        int(query_outer_count),
        int(key_outer_count),
        int(key_rows),
        int(query_columns),
        int(query_rows),
    )
    seed_rows = _SEED_ROWS_CACHE.get(cache_key)
    if seed_rows is not None:
        return seed_rows

    query_outer = torch.arange(
        query_outer_offset,
        query_outer_offset + query_outer_count,
        device=device,
    )
    key_outer = torch.arange(key_outer_count, device=device)
    seed_rows = local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows)
    _SEED_ROWS_CACHE[cache_key] = seed_rows
    return seed_rows


def gather_seed_queries(query, seed_rows):
    """Gather seed queries for every (query outer, key outer, key row, query column) block."""
    query_outer = torch.arange(query.size(1), device=query.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(query.size(3), device=query.device).view(1, 1, 1, -1, 1)
    return query[:, query_outer, seed_rows, query_col]


def biased_sdpa(query, key, value, bias, scale=1.0):
    """Full compressed attention with additive entropy/routing bias."""
    query_for_attn = query.to(key.dtype) if query.dtype != key.dtype else query
    out = F.scaled_dot_product_attention(
        query_for_attn.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        attn_mask=(-bias.float()).to(query_for_attn.dtype).unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(1).to(query.dtype)


def hard_route_merge_kv_pytorch(seed_queries, key, value, scale):
    """Readable PyTorch reference for hard C4 merge.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]

    A fixed (A, F, J, K) block has query rows I=30 and key columns L=52.
    The C seed queries compress those L keys into C merged K/V tokens.

    Returns:
      merged_key/value: [B, A, F, J, K, C, H, D]
      routing_bias:     [B, H, A, F, K, J, C]
    """
    seed_key_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    seed_key_logits = seed_key_logits * float(scale)

    # Hard-assign each key column L to the seed C whose distribution over keys
    # gives L the highest probability.
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assignment_scores = torch.log_softmax(seed_key_logits, dim=-1)
    best_seed = assignment_scores.argmax(dim=-2, keepdim=True)
    assignment.scatter_(dim=-2, index=best_seed, value=True)

    logits = seed_key_logits.masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(logits, dim=-1).masked_fill(~assignment, 0.0)
    weights_f32 = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights_f32 * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(~assignment.any(dim=-1), float("inf"))
    merge_weights = weights_f32.to(key.dtype)

    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", merge_weights, key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", merge_weights, value)
    return merged_key, merged_value, routing_bias


def hard_route_assignment_pytorch(seed_queries, key, scale):
    """Return the original hard key partition induced by routing prototypes."""

    logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seed_queries, key
    ).float()
    best_seed = torch.log_softmax(logits * float(scale), dim=-1).argmax(
        dim=-2, keepdim=True
    )
    assignment = torch.zeros_like(logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    return assignment


def hard_route_merge_fixed_assignment_pytorch(
    expansion_centers, key, value, assignment, scale
):
    """Recompute cluster moments without changing the hard key partition."""

    logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", expansion_centers, key
    ).float()
    routed_logits = (logits * float(scale)).masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(routed_logits, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(
        ~assignment.any(dim=-1), float("inf")
    )
    merge_weights = weights.to(key.dtype)
    merged_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", merge_weights, key
    )
    merged_value = torch.einsum(
        "bhafkjcl,bfklhe->bafjkche", merge_weights, value
    )
    return merged_key, merged_value, routing_bias


def hard_route_compressed_attention_pytorch(
    query, merged_key, merged_value, routing_bias, scale
):
    """Evaluate attention over route-contiguous compressed cluster moments."""

    query_flat = rearrange(query, "b a i j h d -> (b h a j) i d")
    key_flat = rearrange(
        merged_key, "b a f j k c h d -> (b h a j) (f k c) d"
    )
    value_flat = rearrange(
        merged_value, "b a f j k c h d -> (b h a j) (f k c) d"
    )
    bias_flat = rearrange(
        routing_bias, "b h a f k j c -> (b h a j) (f k c)"
    )
    output = biased_sdpa(
        query_flat, key_flat, value_flat, bias_flat, scale=scale
    )
    return rearrange(
        output,
        "(b h a j) i d -> b a i j h d",
        b=query.size(0),
        h=query.size(4),
        a=query.size(1),
        j=query.size(3),
    )


def responsibility_centroids_pytorch(
    query, merged_key, routing_bias, routing_prototypes, scale
):
    r"""Compute the responsibility-optimal expansion center for every cluster.

    For compressed attention probabilities :math:`\pi_{ic}`, this returns
    :math:`u_c = \sum_i \pi_{ic}q_i / \sum_i\pi_{ic}` and the exact weighted
    variance reduction :math:`G_c=s_c\lVert u_c-r_c\rVert_2^2`.
    """

    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    key_h = rearrange(
        merged_key, "b a f j k c h d -> b h a j (f k c) d"
    ).float()
    bias_h = rearrange(
        routing_bias, "b h a f k j c -> b h a j (f k c)"
    ).float()
    logits = torch.einsum("bhajid,bhajtd->bhajit", query_h, key_h)
    responsibility = torch.softmax(
        logits * float(scale) - bias_h.unsqueeze(-2), dim=-1
    )
    mass = responsibility.sum(dim=-2)
    weighted_query = torch.einsum(
        "bhajit,bhajid->bhajtd", responsibility, query_h
    )
    centers = weighted_query / mass.clamp_min(
        torch.finfo(torch.float32).tiny
    ).unsqueeze(-1)
    prototypes_h = rearrange(
        routing_prototypes, "b a f k j c h d -> b h a j (f k c) d"
    ).float()
    centers = torch.where(
        (mass > 0).unsqueeze(-1), centers, prototypes_h
    )
    gain = mass * (centers - prototypes_h).square().sum(dim=-1)
    key_frames = merged_key.size(2)
    key_rows = merged_key.size(4)
    clusters = merged_key.size(5)
    centers = rearrange(
        centers,
        "b h a j (f k c) d -> b a f k j c h d",
        f=key_frames,
        k=key_rows,
        c=clusters,
    ).to(query.dtype)
    gain = rearrange(
        gain,
        "b h a j (f k c) -> b a f k j c h",
        f=key_frames,
        k=key_rows,
        c=clusters,
    )
    return centers, gain


def hard_route_responsibility_centroid_from_seeds_pytorch(
    query, key, value, routing_prototypes, scale, top_r=2
):
    """Reference RC-GMA with fixed routing and budgeted centroid refinement."""

    clusters = routing_prototypes.size(5)
    top_r = int(top_r)
    if top_r < 1 or top_r > clusters:
        raise ValueError(f"top_r must be in [1, {clusters}], got {top_r}.")

    assignment = hard_route_assignment_pytorch(routing_prototypes, key, scale)
    initial_key, initial_value, initial_bias = hard_route_merge_kv_pytorch(
        routing_prototypes, key, value, scale
    )
    expansion_centers, gain = responsibility_centroids_pytorch(
        query, initial_key, initial_bias, routing_prototypes, scale
    )
    refined_key, refined_value, refined_bias = (
        hard_route_merge_fixed_assignment_pytorch(
            expansion_centers, key, value, assignment, scale
        )
    )

    selected = gain.topk(top_r, dim=5).indices
    selected_mask = torch.zeros_like(gain, dtype=torch.bool)
    selected_mask.scatter_(dim=5, index=selected, value=True)
    moment_mask = rearrange(
        selected_mask, "b a f k j c h -> b a f j k c h"
    ).unsqueeze(-1)
    bias_mask = rearrange(
        selected_mask, "b a f k j c h -> b h a f k j c"
    )
    merged_key = torch.where(moment_mask, refined_key, initial_key)
    merged_value = torch.where(moment_mask, refined_value, initial_value)
    routing_bias = torch.where(bias_mask, refined_bias, initial_bias)
    return hard_route_compressed_attention_pytorch(
        query, merged_key, merged_value, routing_bias, scale
    )


def hard_route_attention_pytorch(q, k, v, scale, query_outer_offset=0):
    """Readable end-to-end hard C4 attention reference.

    q:   [B, A, I, J, H, D]
    k/v: [B, F, K, L, H, D]

    A fixed (a, f, j, k) block has I query rows x L key columns. This PyTorch
    path computes the same C4 merge and full compressed attention as Triton.
    """
    seed_rows = cached_local_c4_seed_rows(
        q.device,
        query_outer_offset,
        q.size(1),
        k.size(1),
        k.size(2),
        q.size(3),
        q.size(2),
    )
    seed_queries = gather_seed_queries(q, seed_rows)
    merged_keys, merged_values, routing_bias = hard_route_merge_kv_pytorch(
        seed_queries,
        k,
        v,
        scale,
    )

    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    out = biased_sdpa(query, key, value, bias, scale=scale)
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


def hard_route_attention_streaming_pytorch(q, k, v, scale):
    """No-intermediate reference for the same hard-C4 equations.

    This computes one ``(key_outer, key_row)`` block at a time and immediately
    updates the second-stage softmax accumulator. It avoids materializing the
    large ``merged_k``, ``merged_v``, and ``routing_bias`` tensors, making the
    dataflow closer to the intended fused Triton kernel. It is a correctness
    reference, not a fast PyTorch implementation.
    """
    batch, query_outer, query_rows, query_cols, heads, dim = q.shape
    key_outer, key_rows = k.size(1), k.size(2)

    acc = torch.zeros(
        batch,
        heads,
        query_outer,
        query_rows,
        query_cols,
        dim,
        device=q.device,
        dtype=torch.float32,
    )
    row_max = torch.full(
        (batch, heads, query_outer, query_rows, query_cols),
        FLOAT_NEG_INF,
        device=q.device,
        dtype=torch.float32,
    )
    row_sum = torch.zeros_like(row_max)

    query = rearrange(q, "b a i j h d -> b h a i j d")
    for key_outer_idx in range(key_outer):
        for key_row_idx in range(key_rows):
            seed_start = min(max(key_row_idx - 1, 0), query_rows - 4)
            seed_rows = [seed_start + offset for offset in range(4)]
            seed_queries = torch.stack([q[:, :, row] for row in seed_rows], dim=3)
            key_row = k[:, key_outer_idx, key_row_idx]
            value_row = v[:, key_outer_idx, key_row_idx]

            seed_logits = torch.einsum("bajchd,blhd->bhajcl", seed_queries, key_row).float()
            seed_logits = seed_logits * float(scale)

            assignment = torch.zeros_like(seed_logits, dtype=torch.bool)
            best_seed = torch.log_softmax(seed_logits, dim=-1).argmax(dim=-2, keepdim=True)
            assignment.scatter_(dim=-2, index=best_seed, value=True)

            logits = seed_logits.masked_fill(~assignment, FLOAT_NEG_INF)
            log_weights = torch.log_softmax(logits, dim=-1).masked_fill(~assignment, 0.0)
            weights = log_weights.exp().masked_fill(~assignment, 0.0)
            bias = (weights * log_weights).sum(dim=-1)
            bias = bias.masked_fill(~assignment.any(dim=-1), float("inf"))

            merged_key = torch.einsum("bhajcl,blhd->bajchd", weights.to(k.dtype), key_row)
            merged_value = torch.einsum("bhajcl,blhd->bajchd", weights.to(v.dtype), value_row)
            scores = torch.einsum("bhaijd,bajchd->bhaijc", query, merged_key).float()
            scores = scores * float(scale) - bias[:, :, :, None]

            new_max = torch.maximum(row_max, scores.max(dim=-1).values)
            probs = torch.exp(scores - new_max[..., None])
            old_scale = torch.exp(row_max - new_max)
            acc = acc * old_scale[..., None] + torch.einsum(
                "bhaijc,bajchd->bhaijd",
                probs.to(merged_value.dtype),
                merged_value,
            ).float()
            row_sum = row_sum * old_scale + probs.sum(dim=-1)
            row_max = new_max

    return rearrange(acc / row_sum[..., None], "b h a i j d -> b a i j h d").to(q.dtype)


__all__ = [
    "biased_sdpa",
    "cached_local_c4_seed_rows",
    "gather_seed_queries",
    "hard_route_attention_pytorch",
    "hard_route_attention_streaming_pytorch",
    "hard_route_merge_kv_pytorch",
    "local_c4_seed_rows",
    "wide_local_c4_seed_rows",
]
