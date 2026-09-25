"""Mean-centered two-child refinement for Wan2.1 720p Hard-C4.

The four coarse GMA clusters keep their original hard key partition.  Within
each parent, a query-tangent coordinate is split at its parent-weighted mean.
The two parents with the largest explained tangent variance are replaced by
their low/high conditional moments, yielding six matched K/V/entropy tokens.
"""

from __future__ import annotations

import torch

from attention import hard_c4_wan21_14b_720p_triton as hard_c4

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


HIERARCHICAL_C = 4
HIERARCHICAL_TOP_R = 2
HIERARCHICAL_OUT_C = HIERARCHICAL_C + HIERARCHICAL_TOP_R
HIERARCHICAL_BLOCK_J = 16


if triton is not None:

    @triton.jit
    def _hierarchical_c4_mean_split_merge_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        Q_OUTER: tl.constexpr,
        Q_GROUPS: tl.constexpr,
        Q_GROUP: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        TOP_R: tl.constexpr,
        OUT_C: tl.constexpr,
        BLOCK_J: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Fuse coarse routing, mean split, allocation, and child moments."""

        tl.static_assert(C == 4)
        tl.static_assert(TOP_R == 2)
        tl.static_assert(OUT_C == 6)

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)

        query_group = block_id % Q_GROUPS
        block_id //= Q_GROUPS
        key_outer = block_id % K_OUTER
        key_row = block_id // K_OUTER

        parent_rows = tl.arange(0, BLOCK_J * C)
        parent_ids = parent_rows % C
        main_cols = tl.arange(0, 64)
        tail_cols = 64 + tl.arange(0, 16)
        dims = tl.arange(0, 128)

        row_base = (
            (batch * K_OUTER + key_outer)
            * ROWS
            * COLS
            * TOTAL_HEADS
            * D_HEAD
            + key_row * COLS * TOTAL_HEADS * D_HEAD
        )
        key_offsets = (
            row_base
            + (main_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
            + dims[None, :]
        )
        key_tail_offsets = (
            row_base
            + (tail_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
            + dims[None, :]
        )
        key_main = tl.load(k_ptr + key_offsets, eviction_policy="evict_last")
        key_tail = tl.load(k_ptr + key_tail_offsets, eviction_policy="evict_last")
        value_main = tl.load(v_ptr + key_offsets, eviction_policy="evict_last")
        value_tail = tl.load(v_ptr + key_tail_offsets, eviction_policy="evict_last")
        key_value_main = tl.interleave(key_main, value_main)
        key_value_tail = tl.interleave(key_tail, value_tail)

        seed_start = tl.minimum(tl.maximum(key_row - 1, 0), ROWS - C)
        seed_rows = seed_start + parent_ids
        wide_offset = tl.where(
            parent_ids == 0,
            -2,
            tl.where(parent_ids == 1, 0, tl.where(parent_ids == 2, 1, 3)),
        )
        use_wide_seeds = (key_row >= 2) & (key_row <= ROWS - 4)
        seed_rows = tl.where(use_wide_seeds, key_row + wide_offset, seed_rows)

        for q_offset in tl.range(0, Q_GROUP, num_stages=LOOP_STAGES):
            query_outer = query_group * Q_GROUP + q_offset
            q_base = (
                (batch * Q_OUTER + query_outer)
                * ROWS
                * COLS
                * TOTAL_HEADS
                * D_HEAD
            )
            for query_col_start in tl.range(0, COLS, BLOCK_J):
                parent_query_cols = query_col_start + parent_rows // C
                seed_query_offsets = (
                    q_base
                    + (
                        (
                            seed_rows[:, None] * COLS
                            + parent_query_cols[:, None]
                        )
                        * TOTAL_HEADS
                        + global_head
                    )
                    * D_HEAD
                    + dims[None, :]
                )
                seed_query = tl.load(
                    q_ptr + seed_query_offsets,
                    eviction_policy="evict_last",
                )
                scores_main = tl.dot(
                    seed_query,
                    tl.trans(key_main),
                    input_precision=INPUT_PRECISION,
                )
                scores_tail = tl.dot(
                    seed_query,
                    tl.trans(key_tail),
                    input_precision=INPUT_PRECISION,
                )
                scores_main *= SCALE * 1.4426950408889634
                scores_tail *= SCALE * 1.4426950408889634

                row_max = tl.maximum(
                    tl.max(scores_main, axis=1), tl.max(scores_tail, axis=1)
                )
                row_sum = tl.sum(
                    tl.exp2(scores_main - row_max[:, None]), axis=1
                )
                row_sum += tl.sum(
                    tl.exp2(scores_tail - row_max[:, None]), axis=1
                )
                log_norm = row_max + tl.log2(row_sum)
                log_probs_main = tl.reshape(
                    scores_main - log_norm[:, None], (BLOCK_J, C, 64)
                )
                log_probs_tail = tl.reshape(
                    scores_tail - log_norm[:, None], (BLOCK_J, C, 16)
                )

                winner_main, route_main = tl.max(
                    log_probs_main,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                winner_tail, route_tail = tl.max(
                    log_probs_tail,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                cluster_ids = tl.arange(0, C)[None, :, None]
                mask_main = route_main[:, None, :] == cluster_ids
                mask_tail = route_tail[:, None, :] == cluster_ids
                probs_main = tl.where(
                    mask_main, tl.exp2(winner_main)[:, None, :], 0.0
                )
                probs_tail = tl.where(
                    mask_tail, tl.exp2(winner_tail)[:, None, :], 0.0
                )
                parent_sum = tl.sum(probs_main, axis=2) + tl.sum(
                    probs_tail, axis=2
                )
                parent_valid = parent_sum > 0.0
                safe_parent_sum = tl.where(parent_valid, parent_sum, 1.0)
                parent_weights_main = probs_main / safe_parent_sum[:, :, None]
                parent_weights_tail = probs_tail / safe_parent_sum[:, :, None]
                parent_expected = tl.sum(
                    probs_main
                    * tl.where(mask_main, winner_main[:, None, :], 0.0),
                    axis=2,
                )
                parent_expected += tl.sum(
                    probs_tail
                    * tl.where(mask_tail, winner_tail[:, None, :], 0.0),
                    axis=2,
                )
                parent_bias = (
                    parent_expected / safe_parent_sum
                    - tl.log2(safe_parent_sum)
                ) * 0.6931471805599453
                parent_bias = tl.where(
                    parent_valid, parent_bias, float("inf")
                )

                # Unpack the four seed-score rows while retaining the query
                # column and key-column dimensions.
                score_main_by_cluster = tl.permute(
                    tl.reshape(scores_main, (BLOCK_J, C, 64)), 0, 2, 1
                )
                score_main_even, score_main_odd = tl.split(
                    tl.reshape(score_main_by_cluster, (BLOCK_J, 64, 2, 2))
                )
                score_main_0, score_main_2 = tl.split(score_main_even)
                score_main_1, score_main_3 = tl.split(score_main_odd)
                tangent_main_01 = tl.join(
                    score_main_1 - score_main_0,
                    score_main_2 - score_main_0,
                )
                tangent_main_23 = tl.join(
                    score_main_3 - score_main_1,
                    score_main_3 - score_main_2,
                )
                tangent_main = tl.permute(
                    tl.reshape(
                        tl.permute(
                            tl.join(tangent_main_01, tangent_main_23),
                            0,
                            1,
                            3,
                            2,
                        ),
                        (BLOCK_J, 64, C),
                    ),
                    0,
                    2,
                    1,
                )

                score_tail_by_cluster = tl.permute(
                    tl.reshape(scores_tail, (BLOCK_J, C, 16)), 0, 2, 1
                )
                score_tail_even, score_tail_odd = tl.split(
                    tl.reshape(score_tail_by_cluster, (BLOCK_J, 16, 2, 2))
                )
                score_tail_0, score_tail_2 = tl.split(score_tail_even)
                score_tail_1, score_tail_3 = tl.split(score_tail_odd)
                tangent_tail_01 = tl.join(
                    score_tail_1 - score_tail_0,
                    score_tail_2 - score_tail_0,
                )
                tangent_tail_23 = tl.join(
                    score_tail_3 - score_tail_1,
                    score_tail_3 - score_tail_2,
                )
                tangent_tail = tl.permute(
                    tl.reshape(
                        tl.permute(
                            tl.join(tangent_tail_01, tangent_tail_23),
                            0,
                            1,
                            3,
                            2,
                        ),
                        (BLOCK_J, 16, C),
                    ),
                    0,
                    2,
                    1,
                )

                tangent_mean = tl.sum(
                    parent_weights_main * tangent_main, axis=2
                )
                tangent_mean += tl.sum(
                    parent_weights_tail * tangent_tail, axis=2
                )
                low_mask_main = mask_main & (
                    tangent_main <= tangent_mean[:, :, None]
                )
                low_mask_tail = mask_tail & (
                    tangent_tail <= tangent_mean[:, :, None]
                )
                high_mask_main = mask_main & ~low_mask_main
                high_mask_tail = mask_tail & ~low_mask_tail
                low_probs_main = tl.where(low_mask_main, probs_main, 0.0)
                low_probs_tail = tl.where(low_mask_tail, probs_tail, 0.0)
                high_probs_main = tl.where(high_mask_main, probs_main, 0.0)
                high_probs_tail = tl.where(high_mask_tail, probs_tail, 0.0)
                low_sum = tl.sum(low_probs_main, axis=2) + tl.sum(
                    low_probs_tail, axis=2
                )
                high_sum = tl.sum(high_probs_main, axis=2) + tl.sum(
                    high_probs_tail, axis=2
                )
                low_valid = low_sum > 0.0
                high_valid = high_sum > 0.0
                safe_low_sum = tl.where(low_valid, low_sum, 1.0)
                safe_high_sum = tl.where(high_valid, high_sum, 1.0)
                low_weights_main = low_probs_main / safe_low_sum[:, :, None]
                low_weights_tail = low_probs_tail / safe_low_sum[:, :, None]
                high_weights_main = high_probs_main / safe_high_sum[:, :, None]
                high_weights_tail = high_probs_tail / safe_high_sum[:, :, None]

                low_log_norm = tl.log2(safe_low_sum)
                high_log_norm = tl.log2(safe_high_sum)
                low_bias = tl.sum(
                    low_weights_main
                    * tl.where(
                        low_mask_main,
                        winner_main[:, None, :] - low_log_norm[:, :, None],
                        0.0,
                    ),
                    axis=2,
                )
                low_bias += tl.sum(
                    low_weights_tail
                    * tl.where(
                        low_mask_tail,
                        winner_tail[:, None, :] - low_log_norm[:, :, None],
                        0.0,
                    ),
                    axis=2,
                )
                high_bias = tl.sum(
                    high_weights_main
                    * tl.where(
                        high_mask_main,
                        winner_main[:, None, :] - high_log_norm[:, :, None],
                        0.0,
                    ),
                    axis=2,
                )
                high_bias += tl.sum(
                    high_weights_tail
                    * tl.where(
                        high_mask_tail,
                        winner_tail[:, None, :] - high_log_norm[:, :, None],
                        0.0,
                    ),
                    axis=2,
                )
                low_bias *= 0.6931471805599453
                high_bias *= 0.6931471805599453
                low_bias = tl.where(low_valid, low_bias, float("inf"))
                high_bias = tl.where(high_valid, high_bias, float("inf"))

                low_mean = tl.sum(low_weights_main * tangent_main, axis=2)
                low_mean += tl.sum(low_weights_tail * tangent_tail, axis=2)
                high_mean = tl.sum(high_weights_main * tangent_main, axis=2)
                high_mean += tl.sum(high_weights_tail * tangent_tail, axis=2)
                low_mass = low_sum / safe_parent_sum
                high_mass = high_sum / safe_parent_sum
                split_gain = (
                    low_mass
                    * high_mass
                    * (high_mean - low_mean)
                    * (high_mean - low_mean)
                )
                split_gain = tl.where(
                    low_valid & high_valid, split_gain, -float("inf")
                )
                _, first_id = tl.max(
                    split_gain,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                candidate_ids = tl.arange(0, C)[None, :]
                second_gain = tl.where(
                    candidate_ids != first_id[:, None],
                    split_gain,
                    -float("inf"),
                )
                _, second_id = tl.max(
                    second_gain,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                selected = (candidate_ids == first_id[:, None]) | (
                    candidate_ids == second_id[:, None]
                )

                base_weights_main = tl.where(
                    selected[:, :, None], low_weights_main, parent_weights_main
                )
                base_weights_tail = tl.where(
                    selected[:, :, None], low_weights_tail, parent_weights_tail
                )
                base_bias = tl.where(selected, low_bias, parent_bias)
                base_pair = tl.dot(
                    tl.reshape(base_weights_main, (BLOCK_J * C, 64)).to(
                        key_main.dtype
                    ),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                base_pair += tl.dot(
                    tl.reshape(base_weights_tail, (BLOCK_J * C, 16)).to(
                        key_tail.dtype
                    ),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                base_key, base_value = tl.split(
                    tl.reshape(base_pair, (BLOCK_J * C, 128, 2))
                )

                # Gather the high child of each selected parent. The unpacking
                # follows the same C4 layout as the score tensor above.
                high_main_by_cluster = tl.permute(
                    high_weights_main, 0, 2, 1
                )
                high_main_even, high_main_odd = tl.split(
                    tl.reshape(high_main_by_cluster, (BLOCK_J, 64, 2, 2))
                )
                high_main_0, high_main_2 = tl.split(high_main_even)
                high_main_1, high_main_3 = tl.split(high_main_odd)
                first_high_main = tl.where(
                    first_id[:, None] == 0,
                    high_main_0,
                    tl.where(
                        first_id[:, None] == 1,
                        high_main_1,
                        tl.where(first_id[:, None] == 2, high_main_2, high_main_3),
                    ),
                )
                second_high_main = tl.where(
                    second_id[:, None] == 0,
                    high_main_0,
                    tl.where(
                        second_id[:, None] == 1,
                        high_main_1,
                        tl.where(second_id[:, None] == 2, high_main_2, high_main_3),
                    ),
                )
                extra_weights_main = tl.reshape(
                    tl.permute(
                        tl.join(first_high_main, second_high_main), 0, 2, 1
                    ),
                    (BLOCK_J * TOP_R, 64),
                )

                high_tail_by_cluster = tl.permute(
                    high_weights_tail, 0, 2, 1
                )
                high_tail_even, high_tail_odd = tl.split(
                    tl.reshape(high_tail_by_cluster, (BLOCK_J, 16, 2, 2))
                )
                high_tail_0, high_tail_2 = tl.split(high_tail_even)
                high_tail_1, high_tail_3 = tl.split(high_tail_odd)
                first_high_tail = tl.where(
                    first_id[:, None] == 0,
                    high_tail_0,
                    tl.where(
                        first_id[:, None] == 1,
                        high_tail_1,
                        tl.where(first_id[:, None] == 2, high_tail_2, high_tail_3),
                    ),
                )
                second_high_tail = tl.where(
                    second_id[:, None] == 0,
                    high_tail_0,
                    tl.where(
                        second_id[:, None] == 1,
                        high_tail_1,
                        tl.where(second_id[:, None] == 2, high_tail_2, high_tail_3),
                    ),
                )
                extra_weights_tail = tl.reshape(
                    tl.permute(
                        tl.join(first_high_tail, second_high_tail), 0, 2, 1
                    ),
                    (BLOCK_J * TOP_R, 16),
                )
                extra_pair = tl.dot(
                    extra_weights_main.to(key_main.dtype),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                extra_pair += tl.dot(
                    extra_weights_tail.to(key_tail.dtype),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                extra_key, extra_value = tl.split(
                    tl.reshape(extra_pair, (BLOCK_J * TOP_R, 128, 2))
                )

                high_bias_even, high_bias_odd = tl.split(
                    tl.reshape(high_bias, (BLOCK_J, 2, 2))
                )
                high_bias_0, high_bias_2 = tl.split(high_bias_even)
                high_bias_1, high_bias_3 = tl.split(high_bias_odd)
                first_high_bias = tl.where(
                    first_id == 0,
                    high_bias_0,
                    tl.where(
                        first_id == 1,
                        high_bias_1,
                        tl.where(first_id == 2, high_bias_2, high_bias_3),
                    ),
                )
                second_high_bias = tl.where(
                    second_id == 0,
                    high_bias_0,
                    tl.where(
                        second_id == 1,
                        high_bias_1,
                        tl.where(second_id == 2, high_bias_2, high_bias_3),
                    ),
                )
                extra_bias = tl.reshape(
                    tl.join(first_high_bias, second_high_bias),
                    (BLOCK_J * TOP_R,),
                )

                key_block = key_outer * ROWS + key_row
                base_output_token = key_block * OUT_C + parent_ids
                base_route = (
                    (((batch_i64 * Q_OUTER + query_outer) * COLS
                      + parent_query_cols)
                     * ACTIVE_HEADS
                     + head)
                    * K_OUTER
                    * ROWS
                    * OUT_C
                )
                base_output = (
                    base_route[:, None] + base_output_token[:, None]
                ) * D_HEAD + dims[None, :]
                tl.store(merged_k_ptr + base_output, base_key)
                tl.store(merged_v_ptr + base_output, base_value)
                tl.store(
                    bias_ptr + base_route + base_output_token,
                    tl.reshape(base_bias, (BLOCK_J * C,)),
                )

                extra_rows = tl.arange(0, BLOCK_J * TOP_R)
                extra_query_cols = query_col_start + extra_rows // TOP_R
                extra_rank = extra_rows % TOP_R
                extra_output_token = key_block * OUT_C + C + extra_rank
                extra_route = (
                    (((batch_i64 * Q_OUTER + query_outer) * COLS
                      + extra_query_cols)
                     * ACTIVE_HEADS
                     + head)
                    * K_OUTER
                    * ROWS
                    * OUT_C
                )
                extra_output = (
                    extra_route[:, None] + extra_output_token[:, None]
                ) * D_HEAD + dims[None, :]
                tl.store(merged_k_ptr + extra_output, extra_key)
                tl.store(merged_v_ptr + extra_output, extra_value)
                tl.store(
                    bias_ptr + extra_route + extra_output_token,
                    extra_bias,
                )


def hard_c4_720p_hierarchical_mean_split_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run all-layer C4 plus two mean-centered child refinements."""

    if triton is None:
        raise RuntimeError("Triton is required for hierarchical Hard-C4.")
    scale = q.size(-1) ** -0.5 if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, query_outer, rows, cols, total_heads, dim = q.shape
    key_outer = k.size(1)
    active_heads = int(head_count or total_heads)
    if (rows, cols, dim) != (
        hard_c4.HARD_720_ROWS,
        hard_c4.HARD_720_COLS,
        128,
    ):
        raise ValueError("hierarchical Hard-C4 requires 45x80 tokens with D=128.")
    if query_outer % hard_c4.HARD_720_QUERY_OUTER_GROUP:
        raise ValueError("hierarchical Hard-C4 requires complete 21-frame chunks.")
    query_groups = query_outer // hard_c4.HARD_720_QUERY_OUTER_GROUP

    compressed_tokens = (
        batch
        * query_outer
        * cols
        * active_heads
        * key_outer
        * rows
        * HIERARCHICAL_OUT_C
    )
    merged_k = torch.empty(
        (compressed_tokens, dim), device=q.device, dtype=q.dtype
    )
    merged_v = torch.empty_like(merged_k)
    unused_compressed_scale = torch.empty(
        1, device=q.device, dtype=torch.float32
    )
    bias = torch.empty(compressed_tokens, device=q.device, dtype=torch.float32)
    if output is None:
        output = torch.empty_like(q)

    merge_grid = (
        query_groups * key_outer * rows,
        batch * active_heads,
    )
    _hierarchical_c4_mean_split_merge_kernel[merge_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        unused_compressed_scale,
        unused_compressed_scale,
        bias,
        query_outer,
        query_groups,
        hard_c4.HARD_720_QUERY_OUTER_GROUP,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        rows,
        cols,
        HIERARCHICAL_C,
        HIERARCHICAL_TOP_R,
        HIERARCHICAL_OUT_C,
        HIERARCHICAL_BLOCK_J,
        hard_c4.HARD_720_MERGE_LOOP_STAGES,
        hard_c4.HARD_720_INPUT_PRECISION,
        SCALE=float(scale),
        num_warps=hard_c4.HARD_720_MERGE_NUM_WARPS,
        num_stages=hard_c4.HARD_720_MERGE_NUM_STAGES,
    )

    route_loop_stages = 1 if torch.version.hip is not None else 3
    route_grid = (1, batch * active_heads, query_outer * cols)
    unused_distortion = torch.empty(1, device=q.device, dtype=torch.float32)
    hard_c4._hard_c4_720p_route_kernel[route_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        bias,
        unused_distortion,
        unused_distortion,
        output,
        query_outer,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        False,
        False,
        rows,
        cols,
        HIERARCHICAL_OUT_C,
        hard_c4.HARD_720_ROUTE_BLOCK_I,
        hard_c4.HARD_720_ROUTE_BLOCK_TOKENS,
        route_loop_stages,
        hard_c4.HARD_720_INPUT_PRECISION,
        False,
        0,
        False,
        0,
        SINGLETON_TOP_R=0,
        SINGLETON_QUERY_COLUMN=False,
        SINGLETON_SECONDARY_FRAME_STRIDE=1,
        QUERY_COLUMN_MAJOR=False,
        PACKED_SINGLETON_KV=False,
        SCALED_FP8_KV=False,
        COMPRESSED_FP8=False,
        SCALE=float(scale),
        num_warps=hard_c4.HARD_720_ROUTE_NUM_WARPS,
        num_stages=hard_c4.HARD_720_ROUTE_NUM_STAGES,
    )
    return output


__all__ = ["hard_c4_720p_hierarchical_mean_split_attention_triton_blocks"]
