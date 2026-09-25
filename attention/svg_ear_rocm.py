"""ROCm execution backend for SVG-EAR variable block-sparse attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_sparse_lse_kernel(
    q,
    k,
    block_map,
    q_indptr,
    k_indptr,
    lse_log2,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_mb,
    stride_mh,
    stride_mq,
    stride_mk,
    stride_qib,
    stride_qih,
    stride_qin,
    stride_kib,
    stride_kih,
    stride_kin,
    stride_lb,
    stride_lh,
    stride_ls,
    heads,
    scale,
    Q_CLUSTERS: tl.constexpr,
    K_CLUSTERS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    query_cluster = pid % Q_CLUSTERS
    head_batch = pid // Q_CLUSTERS
    head = head_batch % heads
    batch = head_batch // heads

    q_index = q_indptr + batch * stride_qib + head * stride_qih
    q_start = tl.load(q_index + query_cluster * stride_qin)
    q_end = tl.load(q_index + (query_cluster + 1) * stride_qin)
    q_size = q_end - q_start
    if q_size == 0:
        return

    dims = tl.arange(0, D)
    rows = tl.arange(0, BLOCK_M)
    q_base = q + batch * stride_qb + head * stride_qh
    k_base = k + batch * stride_kb + head * stride_kh
    map_base = block_map + batch * stride_mb + head * stride_mh + query_cluster * stride_mq
    k_index = k_indptr + batch * stride_kib + head * stride_kih

    for q_offset in range(0, q_size, BLOCK_M):
        q_rows = q_start + q_offset + rows
        q_mask = q_rows < q_end
        q_tile = tl.load(
            q_base + q_rows[:, None] * stride_qs + dims[None, :],
            mask=q_mask[:, None],
            other=0.0,
        )
        row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
        row_sum = tl.zeros([BLOCK_M], tl.float32)

        for key_cluster in range(K_CLUSTERS):
            if tl.load(map_base + key_cluster * stride_mk):
                k_start = tl.load(k_index + key_cluster * stride_kin)
                k_end = tl.load(k_index + (key_cluster + 1) * stride_kin)
                k_size = k_end - k_start
                if k_size > 0:
                    cols = tl.arange(0, BLOCK_N)
                    for k_offset in range(0, k_size, BLOCK_N):
                        k_rows = k_start + k_offset + cols
                        k_mask = k_rows < k_end
                        k_tile = tl.load(
                            k_base + k_rows[:, None] * stride_ks + dims[None, :],
                            mask=k_mask[:, None],
                            other=0.0,
                        )
                        scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
                        scores = tl.where(
                            q_mask[:, None] & k_mask[None, :], scores, -float("inf")
                        )
                        tile_max = tl.max(scores, axis=1)
                        next_max = tl.maximum(row_max, tile_max)
                        probabilities = tl.exp(scores - next_max[:, None])
                        probabilities = tl.where(k_mask[None, :], probabilities, 0.0)
                        row_sum = row_sum * tl.exp(row_max - next_max) + tl.sum(
                            probabilities, axis=1
                        )
                        row_max = next_max

        # FlashInfer exposes log-sum-exp in base 2; keep that contract for the
        # unchanged centroid-fusion kernel used by SVG-EAR.
        log2e: tl.constexpr = 1.4426950408889634
        lse = (row_max + tl.math.log(row_sum)) * log2e
        tl.store(
            lse_log2 + batch * stride_lb + head * stride_lh + q_rows * stride_ls,
            lse,
            mask=q_mask,
        )


def dynamic_block_sparse_prune_fwd_triton_rocm(
    query,
    key,
    value,
    key_centroids,
    value_centroids,
    block_map,
    query_cluster_sizes,
    key_cluster_sizes,
    **_kwargs,
):
    """Run SVG-EAR's exact token/centroid mixture without FlashInfer."""
    from svg.kmeans_utils import (
        _fused_qc_kernel_opt,
        dynamic_block_sparse_fwd_triton,
    )

    block_map = block_map.to(device=query.device, dtype=torch.bool)
    query_cluster_sizes = query_cluster_sizes.to(query.device)
    key_cluster_sizes = key_cluster_sizes.to(query.device)
    token_output = dynamic_block_sparse_fwd_triton(
        query,
        key,
        value,
        block_map,
        query_cluster_sizes,
        key_cluster_sizes,
    )

    batch, heads, sequence, dim = query.shape
    query_clusters = query_cluster_sizes.size(-1)
    key_clusters = key_cluster_sizes.size(-1)
    query_indptr = torch.cat(
        [torch.zeros_like(query_cluster_sizes[..., :1]), query_cluster_sizes], dim=-1
    ).cumsum(dim=-1).int()
    key_indptr = torch.cat(
        [torch.zeros_like(key_cluster_sizes[..., :1]), key_cluster_sizes], dim=-1
    ).cumsum(dim=-1).int()
    token_lse_log2 = torch.empty(
        (batch, heads, sequence), device=query.device, dtype=torch.float32
    )

    _block_sparse_lse_kernel[(batch * heads * query_clusters,)](
        query,
        key,
        block_map,
        query_indptr,
        key_indptr,
        token_lse_log2,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        query_indptr.stride(0),
        query_indptr.stride(1),
        query_indptr.stride(2),
        key_indptr.stride(0),
        key_indptr.stride(1),
        key_indptr.stride(2),
        token_lse_log2.stride(0),
        token_lse_log2.stride(1),
        token_lse_log2.stride(2),
        heads,
        dim**-0.5,
        Q_CLUSTERS=query_clusters,
        K_CLUSTERS=key_clusters,
        D=dim,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=4,
        num_stages=1,
    )

    flat_heads = batch * heads
    query_flat = query.view(flat_heads, sequence, dim)
    token_output_flat = token_output.view(flat_heads, sequence, dim)
    key_centroids_flat = key_centroids.view(flat_heads, key_clusters, dim)
    value_centroids_flat = value_centroids.view(flat_heads, key_clusters, dim)
    output = torch.empty_like(query_flat)
    final_lse = torch.empty(
        (flat_heads, sequence), device=query.device, dtype=torch.float32
    )
    _fused_qc_kernel_opt[(query_clusters, flat_heads)](
        Q=query_flat,
        K_centroids=key_centroids_flat,
        V_centroids=value_centroids_flat,
        block_mask_map=block_map,
        block_col_sz=key_cluster_sizes,
        O_flash=token_output_flat,
        LSE_flash=token_lse_log2.view(flat_heads, sequence),
        QC_INDPTR=query_indptr,
        LSE_final=final_lse,
        Out=output,
        stride_qh=query_flat.stride(0),
        stride_qs=query_flat.stride(1),
        stride_qd=query_flat.stride(2),
        stride_ch=key_centroids_flat.stride(0),
        stride_cn=key_centroids_flat.stride(1),
        stride_cd=key_centroids_flat.stride(2),
        stride_mb=block_map.stride(0),
        stride_mh=block_map.stride(1),
        stride_mq=block_map.stride(2),
        stride_mk=block_map.stride(3),
        stride_szb=key_cluster_sizes.stride(0),
        stride_szh=key_cluster_sizes.stride(1),
        stride_szn=key_cluster_sizes.stride(2),
        stride_ipb=query_indptr.stride(0),
        stride_iph=query_indptr.stride(1),
        stride_ipn=query_indptr.stride(2),
        sm_scale=dim**-0.5,
        B=batch,
        H=heads,
        S=sequence,
        D=dim,
        KC_NUM=key_clusters,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=4,
        num_stages=1,
    )
    return output.view(batch, heads, sequence, dim)
