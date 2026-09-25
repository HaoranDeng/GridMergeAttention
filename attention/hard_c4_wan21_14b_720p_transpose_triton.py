"""Transpose hard-C4 for Wan2.1-14B at 720p.

The normal layout groups one column of 45 queries against one row of 80 keys.
This layout groups one row of 80 queries against one column of 45 valid keys
(conceptually padded to 48), then compresses that key column to four tokens.
The route stage remains full attention over every compressed token.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


ROWS = 45
COLS = 80
CLUSTERS = 4
SEED_SLOTS = 4
OUTER_GROUP = 21
HEAD_DIM = 128

MERGE_QUERY_BLOCK = 16
MERGE_KEY_BLOCK = 64
MERGE_NUM_WARPS = 4
MERGE_NUM_STAGES = 2

ROUTE_QUERY_BLOCK = 128
ROUTE_TOKEN_BLOCK = 64
ROUTE_NUM_WARPS = 4
ROUTE_NUM_STAGES = 1


if triton is not None:

    @triton.jit
    def _transpose_merge_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        Q_OUTER: tl.constexpr,
        Q_GROUPS: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        OUTER_GROUP: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Compress one 45-key column for all query frames and one head."""

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)

        query_group = block_id % Q_GROUPS
        block_id //= Q_GROUPS
        key_outer = block_id % K_OUTER
        key_col = block_id // K_OUTER

        key_rows = tl.arange(0, 64)
        dims = tl.arange(0, 128)
        key_mask = key_rows < ROWS
        key_base = (
            ((batch * TOTAL_HEADS + global_head) * K_OUTER + key_outer)
            * ROWS
            * COLS
            * D_HEAD
        )
        key_offsets = (
            key_base
            + (key_rows[:, None] * COLS + key_col) * D_HEAD
            + dims[None, :]
        )
        key = tl.load(
            k_ptr + key_offsets,
            mask=key_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        value = tl.load(
            v_ptr + key_offsets,
            mask=key_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        lanes = tl.arange(0, 64)
        seed_ids = lanes % C_PAD
        valid_seed = seed_ids < C
        local_query_rows = lanes // C_PAD
        seed_start = tl.minimum(tl.maximum(key_col - 1, 0), COLS - C)
        seed_cols = seed_start + seed_ids

        for outer_offset in tl.range(0, OUTER_GROUP, num_stages=2):
            query_outer = query_group * OUTER_GROUP + outer_offset
            query_base = (
                (batch * TOTAL_HEADS + global_head)
                * Q_OUTER
                * ROWS
                * COLS
                * D_HEAD
                + query_outer * ROWS * COLS * D_HEAD
            )

            for query_row_start in tl.range(0, ROWS, 16):
                query_rows = query_row_start + local_query_rows
                query_mask = query_rows < ROWS
                seed_offsets = (
                    query_base
                    + (query_rows[:, None] * COLS + seed_cols[:, None]) * D_HEAD
                    + dims[None, :]
                )
                seed_query = tl.load(
                    q_ptr + seed_offsets,
                    mask=(query_mask & valid_seed)[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )

                scores = tl.dot(
                    seed_query,
                    tl.trans(key),
                    input_precision="ieee",
                )
                scores *= SCALE * 1.4426950408889634
                scores = tl.where(key_mask[None, :], scores, -float("inf"))
                active_seed = valid_seed & query_mask
                scores = tl.where(
                    active_seed[:, None],
                    scores,
                    tl.where(key_mask[None, :], 0.0, -float("inf")),
                )
                row_max = tl.max(scores, axis=1)
                row_sum = tl.sum(tl.exp2(scores - row_max[:, None]), axis=1)
                log_probs = tl.reshape(
                    scores - (row_max + tl.log2(row_sum))[:, None],
                    (16, C_PAD, 64),
                )
                log_probs = tl.where(
                    tl.arange(0, C_PAD)[None, :, None] < C,
                    log_probs,
                    -float("inf"),
                )

                winner, route = tl.max(
                    log_probs,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                cluster_ids = tl.arange(0, C_PAD)[None, :, None]
                assigned = (
                    (route[:, None, :] == cluster_ids)
                    & (cluster_ids < C)
                    & key_mask[None, None, :]
                )
                probabilities = tl.where(
                    assigned, tl.exp2(winner)[:, None, :], 0.0
                )
                probability_sum = tl.sum(probabilities, axis=2)
                has_key = probability_sum > 0.0
                safe_sum = tl.where(has_key, probability_sum, 1.0)
                # Padded keys have winner=-inf. Mask them before multiplying so
                # their zero probability cannot produce 0 * -inf = NaN.
                assigned_log_probability = tl.where(
                    assigned, winner[:, None, :], 0.0
                )
                expected_log_probability = tl.sum(
                    probabilities * assigned_log_probability, axis=2
                ) / safe_sum
                weights = probabilities / safe_sum[:, :, None]
                route_bias = tl.where(
                    has_key,
                    (expected_log_probability - tl.log2(safe_sum))
                    * 0.6931471805599453,
                    float("inf"),
                )

                weights = tl.reshape(weights, (64, 64))
                key_value = tl.interleave(key, value)
                merged_pair = tl.dot(
                    weights.to(key.dtype),
                    key_value,
                    input_precision="ieee",
                )
                merged_key, merged_value = tl.split(
                    tl.reshape(merged_pair, (64, 128, 2))
                )

                merged_base = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * ROWS
                         + query_rows[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * COLS
                      + key_col)
                     * C)
                    + seed_ids[:, None]
                )
                merged_offsets = merged_base * D_HEAD + dims[None, :]
                tl.store(
                    merged_k_ptr + merged_offsets,
                    merged_key,
                    mask=(query_mask & valid_seed)[:, None],
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + merged_offsets,
                    merged_value,
                    mask=(query_mask & valid_seed)[:, None],
                    eviction_policy="evict_first",
                )

                output_rows = query_row_start + tl.arange(0, 16)
                output_mask = output_rows < ROWS
                output_clusters = tl.arange(0, C_PAD)
                bias_base = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * ROWS
                         + output_rows[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * COLS
                      + key_col)
                     * C)
                    + output_clusters[None, :]
                )
                tl.store(
                    bias_ptr + bias_base,
                    route_bias,
                    mask=output_mask[:, None] & (output_clusters[None, :] < C),
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _transpose_route_kernel(
        q_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        out_ptr,
        log_z_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        STORE_LOG_Z: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Full attention over compressed K/V for one 80-query row."""

        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        query_outer = query_group // ROWS
        query_row = query_group % ROWS

        query_cols = tl.arange(0, 128)
        token_lanes = tl.arange(0, 64)
        dims = tl.arange(0, 128)
        query_mask = query_cols < COLS
        query_tokens = (
            (query_outer * ROWS + query_row) * COLS + query_cols[:, None]
        )
        query_offsets = (
            ((batch * TOTAL_HEADS + global_head) * Q_OUTER * ROWS * COLS
             + query_tokens)
            * D_HEAD
            + dims[None, :]
        )
        query = tl.load(
            q_ptr + query_offsets,
            mask=query_mask[:, None],
            other=0.0,
        )

        total_tokens = K_OUTER * COLS * C
        route_base = (
            (((batch_i64 * Q_OUTER + query_outer) * ROWS + query_row)
             * ACTIVE_HEADS
             + head)
            * total_tokens
        )
        row_max = tl.full((128,), -float("inf"), tl.float32)
        row_sum = tl.zeros((128,), tl.float32)
        accumulator = tl.zeros((128, 128), tl.float32)

        for token_start in tl.range(0, total_tokens, 64, num_stages=3):
            tokens = token_start + token_lanes
            token_mask = tokens < total_tokens
            merged_offsets = (route_base + tokens[:, None]) * D_HEAD + dims[None, :]
            merged_key = tl.load(
                merged_k_ptr + merged_offsets,
                mask=token_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            merged_value = tl.load(
                merged_v_ptr + merged_offsets,
                mask=token_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            bias = tl.load(
                bias_ptr + route_base + tokens,
                mask=token_mask,
                other=float("inf"),
                cache_modifier=".cg",
            )

            logits = tl.dot(
                query,
                tl.trans(merged_key),
                input_precision="ieee",
            )
            logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
            logits = tl.where(token_mask[None, :], logits, -float("inf"))
            next_max = tl.maximum(row_max, tl.max(logits, axis=1))
            probabilities = tl.exp2(logits - next_max[:, None])
            old_scale = tl.exp2(row_max - next_max)
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(query.dtype),
                merged_value,
                input_precision="ieee",
            )
            row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
            row_max = next_max

        tl.store(
            out_ptr + query_offsets,
            accumulator / row_sum[:, None],
            mask=query_mask[:, None],
        )
        if STORE_LOG_Z:
            log_z_address = (
                ((((batch * Q_OUTER + query_outer) * ROWS + query_row)
                  * COLS
                  + query_cols)
                 * ACTIVE_HEADS)
                + head
            )
            log_z = (
                row_max + tl.log2(row_sum)
            ) * 0.6931471805599453
            tl.store(
                log_z_ptr + log_z_address,
                log_z,
                mask=query_mask,
            )


def _hard_c_720p_transpose_attention_triton_head_major(
    q,
    k,
    v,
    clusters,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    return_log_partition=False,
):
    """Run transpose hard C on head-major Wan Q/K/V."""

    if triton is None:
        raise RuntimeError("Triton is required for transpose hard C4.")

    scale = q.size(-1) ** -0.5 if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, total_heads, query_length, dim = q.shape
    query_outer = query_length // (ROWS * COLS)
    key_outer = k.size(2) // (ROWS * COLS)
    active_heads = int(head_count or total_heads)
    query_groups = query_outer // OUTER_GROUP
    clusters = int(clusters)
    if clusters not in {3, 4}:
        raise ValueError("transpose Hard-C supports exactly C=3 or C=4.")

    compressed_tokens = (
        batch
        * query_outer
        * ROWS
        * active_heads
        * key_outer
        * COLS
        * clusters
    )
    merged_k = torch.empty((compressed_tokens, dim), device=q.device, dtype=q.dtype)
    merged_v = torch.empty_like(merged_k)
    bias = torch.empty(compressed_tokens, device=q.device, dtype=torch.float32)
    output = torch.empty_like(q) if output is None else output
    if return_log_partition:
        log_partition = torch.empty(
            (batch, query_outer, ROWS, COLS, active_heads),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        log_partition = torch.empty(1, device=q.device, dtype=torch.float32)

    merge_grid = (query_groups * key_outer * COLS, batch * active_heads)
    _transpose_merge_kernel[merge_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        bias,
        query_outer,
        query_groups,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        ROWS,
        COLS,
        clusters,
        SEED_SLOTS,
        OUTER_GROUP,
        SCALE=float(scale),
        num_warps=MERGE_NUM_WARPS,
        num_stages=MERGE_NUM_STAGES,
    )

    route_grid = (1, batch * active_heads, query_outer * ROWS)
    _transpose_route_kernel[route_grid](
        q,
        merged_k,
        merged_v,
        bias,
        output,
        log_partition,
        query_outer,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        ROWS,
        COLS,
        clusters,
        bool(return_log_partition),
        SCALE=float(scale),
        num_warps=ROUTE_NUM_WARPS,
        num_stages=ROUTE_NUM_STAGES,
    )
    if return_log_partition:
        return output, log_partition
    return output


def hard_c4_720p_transpose_attention_triton_head_major(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run transpose hard C4 on Diffusers ``[B, H, 21*45*80, 128]`` QKV."""

    return _hard_c_720p_transpose_attention_triton_head_major(
        q,
        k,
        v,
        4,
        scale,
        output,
        head_start,
        head_count,
    )


def hard_c3_720p_transpose_attention_triton_head_major(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    return_log_partition=False,
):
    """Run symmetric local-seed transpose Hard C3 and optionally return log Z."""

    return _hard_c_720p_transpose_attention_triton_head_major(
        q,
        k,
        v,
        3,
        scale,
        output,
        head_start,
        head_count,
        return_log_partition,
    )


__all__ = [
    "hard_c3_720p_transpose_attention_triton_head_major",
    "hard_c4_720p_transpose_attention_triton_head_major",
]
