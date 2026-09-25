"""CUDA kernels for query-cluster-conditioned GridMerge tail representatives.

For query-cluster seed ``s_a`` and hard key cluster ``C_b``, the merged
representative is

    w_j = softmax_{j in C_b}(s_a^T k_j / sqrt(d)),
    k_ab = sum_j w_j k_j,  v_ab = sum_j w_j v_j,
    h_ab = -sum_j w_j log w_j.

The surrogate logit ``q^T k_ab / sqrt(d) + h_ab`` is tangent-exact at
``q = s_a`` for the cluster log-partition, while ``v_ab`` is the exact
cluster attention output at that seed.  Keys are expected to be contiguous by
hard cluster, as produced by SVG-EAR's semantic-aware permutation.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _segmented_softmax_entropy_kernel(
        scores,
        key_indptr,
        entropy,
        stride_sh,
        stride_sq,
        stride_sn,
        stride_ih,
        stride_ik,
        stride_eh,
        stride_eq,
        stride_ek,
        QUERY_CLUSTERS: tl.constexpr,
        KEY_CLUSTERS: tl.constexpr,
        MAX_CLUSTER_SIZE: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        key_cluster = tl.program_id(1)
        head = tl.program_id(2)

        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < QUERY_CLUSTERS
        indptr = key_indptr + head * stride_ih + key_cluster * stride_ik
        token_start = tl.load(indptr)
        token_end = tl.load(indptr + stride_ik)

        running_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_Q,), tl.float32)
        running_score_sum = tl.zeros((BLOCK_Q,), tl.float32)

        for relative_start in tl.static_range(0, MAX_CLUSTER_SIZE, BLOCK_N):
            token_offsets = token_start + relative_start + tl.arange(0, BLOCK_N)
            token_mask = token_offsets < token_end
            score_ptrs = (
                scores
                + head * stride_sh
                + q_offsets[:, None] * stride_sq
                + token_offsets[None, :] * stride_sn
            )
            score = tl.load(
                score_ptrs,
                mask=q_mask[:, None] & token_mask[None, :],
                other=-float("inf"),
            ).to(tl.float32)
            tile_max = tl.max(score, axis=1)
            next_max = tl.maximum(running_max, tile_max)
            old_scale = tl.exp(running_max - next_max)
            probability = tl.exp(score - next_max[:, None])
            probability = tl.where(token_mask[None, :], probability, 0.0)
            running_sum = running_sum * old_scale + tl.sum(probability, axis=1)
            safe_score = tl.where(token_mask[None, :], score, 0.0)
            running_score_sum = (
                running_score_sum * old_scale
                + tl.sum(probability * safe_score, axis=1)
            )
            running_max = next_max

        inverse_sum = 1.0 / tl.maximum(running_sum, 1.0e-20)
        for relative_start in tl.static_range(0, MAX_CLUSTER_SIZE, BLOCK_N):
            token_offsets = token_start + relative_start + tl.arange(0, BLOCK_N)
            token_mask = token_offsets < token_end
            score_ptrs = (
                scores
                + head * stride_sh
                + q_offsets[:, None] * stride_sq
                + token_offsets[None, :] * stride_sn
            )
            score = tl.load(
                score_ptrs,
                mask=q_mask[:, None] & token_mask[None, :],
                other=-float("inf"),
            ).to(tl.float32)
            probability = tl.exp(score - running_max[:, None]) * inverse_sum[:, None]
            tl.store(
                score_ptrs,
                probability,
                mask=q_mask[:, None] & token_mask[None, :],
            )

        cluster_entropy = (
            running_max
            + tl.log(tl.maximum(running_sum, 1.0e-20))
            - running_score_sum * inverse_sum
        )
        tl.store(
            entropy
            + head * stride_eh
            + q_offsets * stride_eq
            + key_cluster * stride_ek,
            cluster_entropy,
            mask=q_mask & (token_end > token_start),
        )

    @triton.jit
    def _segmented_weighted_sum_kernel(
        weights,
        source,
        key_indptr,
        output,
        stride_wh,
        stride_wq,
        stride_wn,
        stride_xh,
        stride_xn,
        stride_xd,
        stride_ih,
        stride_ik,
        stride_oh,
        stride_oq,
        stride_ok,
        stride_od,
        QUERY_CLUSTERS: tl.constexpr,
        KEY_CLUSTERS: tl.constexpr,
        D: tl.constexpr,
        MAX_CLUSTER_SIZE: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        key_cluster = tl.program_id(1)
        head = tl.program_id(2)

        q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < QUERY_CLUSTERS
        indptr = key_indptr + head * stride_ih + key_cluster * stride_ik
        token_start = tl.load(indptr)
        token_end = tl.load(indptr + stride_ik)

        for d_start in tl.static_range(0, D, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < D
            accumulator = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)
            for relative_start in tl.static_range(0, MAX_CLUSTER_SIZE, BLOCK_N):
                token_offsets = token_start + relative_start + tl.arange(0, BLOCK_N)
                token_mask = token_offsets < token_end
                weight_ptrs = (
                    weights
                    + head * stride_wh
                    + q_offsets[:, None] * stride_wq
                    + token_offsets[None, :] * stride_wn
                )
                source_ptrs = (
                    source
                    + head * stride_xh
                    + token_offsets[:, None] * stride_xn
                    + d_offsets[None, :] * stride_xd
                )
                weight = tl.load(
                    weight_ptrs,
                    mask=q_mask[:, None] & token_mask[None, :],
                    other=0.0,
                )
                source_tile = tl.load(
                    source_ptrs,
                    mask=token_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                accumulator += tl.dot(weight, source_tile)

            output_ptrs = (
                output
                + head * stride_oh
                + q_offsets[:, None] * stride_oq
                + key_cluster * stride_ok
                + d_offsets[None, :] * stride_od
            )
            tl.store(
                output_ptrs,
                accumulator,
                mask=q_mask[:, None] & d_mask[None, :],
            )


def _require_cuda_triton(*tensors: torch.Tensor):
    if triton is None:
        raise RuntimeError("Triton is required for conditioned GridMerge tails")
    if not tensors or any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("conditioned GridMerge tails require CUDA tensors")


def conditioned_gridmerge_representatives(
    query_centroids: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cluster_sizes: torch.Tensor,
):
    """Build all query-cluster-conditioned K/V representatives.

    Args:
        query_centroids: ``[B, H, Qc, D]`` query-cluster seeds.
        keys, values: cluster-contiguous tensors ``[B, H, S, D]``.
        key_cluster_sizes: ``[B, H, Kc]`` cluster lengths.

    Returns:
        ``(merged_keys, merged_values, entropy)`` with shapes
        ``[B, H, Qc, Kc, D]``, ``[B, H, Qc, Kc, D]``, and
        ``[B, H, Qc, Kc]``.
    """

    _require_cuda_triton(query_centroids, keys, values, key_cluster_sizes)
    if keys.shape != values.shape:
        raise ValueError("keys and values must have identical shapes")
    if query_centroids.shape[:2] != keys.shape[:2]:
        raise ValueError("query centroids and keys must share batch/head axes")
    if query_centroids.size(-1) != keys.size(-1):
        raise ValueError("query centroids and keys must share head dimension")

    batch, heads, query_clusters, dimension = query_centroids.shape
    sequence = keys.size(2)
    key_clusters = key_cluster_sizes.size(-1)
    flat_heads = batch * heads
    seed_flat = query_centroids.reshape(flat_heads, query_clusters, dimension)
    key_flat = keys.reshape(flat_heads, sequence, dimension)
    value_flat = values.reshape(flat_heads, sequence, dimension)
    sizes_flat = key_cluster_sizes.reshape(flat_heads, key_clusters)
    if not torch.all(sizes_flat.sum(dim=-1) == sequence):
        raise ValueError("key cluster sizes must sum to the sequence length")

    key_indptr = torch.zeros(
        (flat_heads, key_clusters + 1),
        dtype=torch.int32,
        device=keys.device,
    )
    key_indptr[:, 1:] = torch.cumsum(sizes_flat.to(torch.int32), dim=-1)
    max_cluster_size = max(32, triton.next_power_of_2(int(sizes_flat.max().item())))

    score_dtype = keys.dtype
    scores = torch.bmm(
        seed_flat.to(score_dtype), key_flat.transpose(1, 2)
    )
    scores.mul_(dimension**-0.5)
    entropy = torch.empty(
        (flat_heads, query_clusters, key_clusters),
        dtype=torch.float32,
        device=keys.device,
    )

    block_q = 8
    block_n = 32
    softmax_grid = (
        triton.cdiv(query_clusters, block_q),
        key_clusters,
        flat_heads,
    )
    _segmented_softmax_entropy_kernel[softmax_grid](
        scores,
        key_indptr,
        entropy,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        key_indptr.stride(0),
        key_indptr.stride(1),
        entropy.stride(0),
        entropy.stride(1),
        entropy.stride(2),
        QUERY_CLUSTERS=query_clusters,
        KEY_CLUSTERS=key_clusters,
        MAX_CLUSTER_SIZE=max_cluster_size,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )

    output_shape = (flat_heads, query_clusters, key_clusters, dimension)
    merged_keys = torch.empty(output_shape, dtype=keys.dtype, device=keys.device)
    merged_values = torch.empty_like(merged_keys)
    weighted_sum_grid = softmax_grid
    for source, output in ((key_flat, merged_keys), (value_flat, merged_values)):
        _segmented_weighted_sum_kernel[weighted_sum_grid](
            scores,
            source,
            key_indptr,
            output,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            source.stride(0),
            source.stride(1),
            source.stride(2),
            key_indptr.stride(0),
            key_indptr.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            QUERY_CLUSTERS=query_clusters,
            KEY_CLUSTERS=key_clusters,
            D=dimension,
            MAX_CLUSTER_SIZE=max_cluster_size,
            BLOCK_Q=block_q,
            BLOCK_N=block_n,
            BLOCK_D=32,
            num_warps=4,
            num_stages=2,
        )

    return (
        merged_keys.view(batch, heads, query_clusters, key_clusters, dimension),
        merged_values.view(batch, heads, query_clusters, key_clusters, dimension),
        entropy.view(batch, heads, query_clusters, key_clusters),
    )


def conditioned_gridmerge_flop_ratio(
    sequence: int,
    query_clusters: int,
    key_clusters: int,
    exact_density: float,
):
    """Attention FLOPs divided by dense QK+PV FLOPs, excluding shared routing."""

    tail_attention = key_clusters / sequence
    representative_construction = 3 * query_clusters / (2 * sequence)
    return exact_density + tail_attention + representative_construction
