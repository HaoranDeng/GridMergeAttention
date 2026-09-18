"""Legacy singleton experiments retained only for non-C4/C8 archaeology.

C4 and C8 are exact capacities and may not enter this module's singleton
paths. The internal parameterization remains solely so older non-C4/C8
experiments can be inspected without silently redefining C4 or C8.
"""

from __future__ import annotations

import torch

from attention import hard_c4_wan21_14b_720p_triton as hard_c4
from attention.capacity_contract import require_exact_merged_capacity

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


SINGLETON_C = 4
SINGLETON_TOP_R = 2
SINGLETON_TOP_R3 = 3
SINGLETON_TOP_R6 = 6
SINGLETON_TOP_RS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32)
SINGLETON_EXTRA_BLOCK = 16
SINGLETON_EXTRA_BLOCK_LARGE = 32
SINGLETON_BLOCK_J = 16
CLUSTER_MASS_CORRECTION_BLOCK = 16


if triton is not None:

    @triton.jit
    def _singleton_c4_leverage_merge_kernel(
        q_ptr,
        seed_row_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        Q_OUTER: tl.constexpr,
        Q_GROUPS: tl.constexpr,
        Q_GROUP: tl.constexpr,
        ROUTER_FRAMES: tl.constexpr,
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
        EXTRA_BLOCK: tl.constexpr,
        BLOCK_J: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_EXTERNAL_SEEDS: tl.constexpr,
        MASS_SELECTOR: tl.constexpr,
        CLUSTER_MASS_SELECTOR: tl.constexpr,
        QUERY_COLUMN_SELECTOR: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        tl.static_assert(C == 4)
        tl.static_assert(
            (TOP_R == 1)
            | (TOP_R == 2)
            | (TOP_R == 3)
            | (TOP_R == 4)
            | (TOP_R == 6)
            | (TOP_R == 8)
            | (TOP_R == 12)
            | (TOP_R == 16)
            | (TOP_R == 20)
            | (TOP_R == 24)
            | (TOP_R == 28)
            | (TOP_R == 32)
        )
        tl.static_assert(OUT_C == C + TOP_R)
        tl.static_assert(
            (EXTRA_BLOCK == 1)
            | (EXTRA_BLOCK == 2)
            | (EXTRA_BLOCK == 4)
            | (EXTRA_BLOCK == 8)
            | (EXTRA_BLOCK == 16)
            | (EXTRA_BLOCK == 32)
        )
        tl.static_assert(TOP_R <= EXTRA_BLOCK)

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
                if USE_EXTERNAL_SEEDS:
                    router_slot = (query_outer * ROUTER_FRAMES) // Q_OUTER
                    seed_row_offset = (
                        ((((((batch_i64 * ROUTER_FRAMES + router_slot) * COLS
                              + parent_query_cols)
                             * ACTIVE_HEADS
                             + head)
                            * K_OUTER
                            + key_outer)
                           * ROWS
                           + key_row)
                          * C)
                        + parent_ids
                    )
                    seed_rows = tl.load(seed_row_ptr + seed_row_offset).to(
                        tl.int32
                    )
                else:
                    seed_start = tl.minimum(
                        tl.maximum(key_row - 1, 0), ROWS - C
                    )
                    seed_rows = seed_start + parent_ids
                    wide_offset = tl.where(
                        parent_ids == 0,
                        -2,
                        tl.where(
                            parent_ids == 1,
                            0,
                            tl.where(parent_ids == 2, 1, 3),
                        ),
                    )
                    use_wide_seeds = (key_row >= 2) & (key_row <= ROWS - 4)
                    seed_rows = tl.where(
                        use_wide_seeds, key_row + wide_offset, seed_rows
                    )
                seed_query_offsets = (
                    q_base
                    + (
                        (seed_rows[:, None] * COLS + parent_query_cols[:, None])
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

                if CLUSTER_MASS_SELECTOR:
                    main_value, main_index = tl.max(
                        parent_weights_main,
                        axis=2,
                        return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    tail_value, tail_index = tl.max(
                        parent_weights_tail,
                        axis=2,
                        return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    selected_cols_by_rank = tl.where(
                        tail_value > main_value,
                        64 + tail_index,
                        main_index,
                    )
                    selected_weight = tl.maximum(main_value, tail_value)
                elif MASS_SELECTOR:
                    key_leverage_main = tl.max(parent_weights_main, axis=1)
                    key_leverage_tail = tl.max(parent_weights_tail, axis=1)
                else:
                    score_main_by_seed = tl.permute(
                        tl.reshape(scores_main, (BLOCK_J, C, 64)), 0, 2, 1
                    )
                    score_main_even, score_main_odd = tl.split(
                        tl.reshape(score_main_by_seed, (BLOCK_J, 64, 2, 2))
                    )
                    score_main_0, score_main_2 = tl.split(score_main_even)
                    score_main_1, score_main_3 = tl.split(score_main_odd)

                    score_tail_by_seed = tl.permute(
                        tl.reshape(scores_tail, (BLOCK_J, C, 16)), 0, 2, 1
                    )
                    score_tail_even, score_tail_odd = tl.split(
                        tl.reshape(score_tail_by_seed, (BLOCK_J, 16, 2, 2))
                    )
                    score_tail_0, score_tail_2 = tl.split(score_tail_even)
                    score_tail_1, score_tail_3 = tl.split(score_tail_odd)

                    mean_0 = tl.sum(
                        parent_weights_main * score_main_0[:, None, :], axis=2
                    )
                    mean_0 += tl.sum(
                        parent_weights_tail * score_tail_0[:, None, :], axis=2
                    )
                    mean_1 = tl.sum(
                        parent_weights_main * score_main_1[:, None, :], axis=2
                    )
                    mean_1 += tl.sum(
                        parent_weights_tail * score_tail_1[:, None, :], axis=2
                    )
                    mean_2 = tl.sum(
                        parent_weights_main * score_main_2[:, None, :], axis=2
                    )
                    mean_2 += tl.sum(
                        parent_weights_tail * score_tail_2[:, None, :], axis=2
                    )
                    mean_3 = tl.sum(
                        parent_weights_main * score_main_3[:, None, :], axis=2
                    )
                    mean_3 += tl.sum(
                        parent_weights_tail * score_tail_3[:, None, :], axis=2
                    )

                    main_delta_0 = score_main_0[:, None, :] - mean_0[:, :, None]
                    main_delta_1 = score_main_1[:, None, :] - mean_1[:, :, None]
                    main_delta_2 = score_main_2[:, None, :] - mean_2[:, :, None]
                    main_delta_3 = score_main_3[:, None, :] - mean_3[:, :, None]
                    distance_main = (
                        main_delta_0 * main_delta_0
                        + main_delta_1 * main_delta_1
                        + main_delta_2 * main_delta_2
                        + main_delta_3 * main_delta_3
                    )
                    tail_delta_0 = score_tail_0[:, None, :] - mean_0[:, :, None]
                    tail_delta_1 = score_tail_1[:, None, :] - mean_1[:, :, None]
                    tail_delta_2 = score_tail_2[:, None, :] - mean_2[:, :, None]
                    tail_delta_3 = score_tail_3[:, None, :] - mean_3[:, :, None]
                    distance_tail = (
                        tail_delta_0 * tail_delta_0
                        + tail_delta_1 * tail_delta_1
                        + tail_delta_2 * tail_delta_2
                        + tail_delta_3 * tail_delta_3
                    )
                    leverage_main = (
                        parent_weights_main
                        / tl.maximum(1.0 - parent_weights_main, 1.0e-6)
                        * distance_main
                    )
                    leverage_tail = (
                        parent_weights_tail
                        / tl.maximum(1.0 - parent_weights_tail, 1.0e-6)
                        * distance_tail
                    )
                    leverage_main = tl.where(
                        mask_main, leverage_main, -float("inf")
                    )
                    leverage_tail = tl.where(
                        mask_tail, leverage_tail, -float("inf")
                    )
                    key_leverage_main = tl.max(leverage_main, axis=1)
                    key_leverage_tail = tl.max(leverage_tail, axis=1)

                extra_lanes = tl.arange(0, EXTRA_BLOCK)
                if QUERY_COLUMN_SELECTOR:
                    tl.static_assert(TOP_R == 1)
                    query_cols = query_col_start + tl.arange(0, BLOCK_J)
                    selected_cols_by_rank = tl.where(
                        extra_lanes[None, :] == 0,
                        query_cols[:, None],
                        0,
                    )
                    selected_main = (
                        main_cols[None, :] == query_cols[:, None]
                    )
                    selected_tail = (
                        tail_cols[None, :] == query_cols[:, None]
                    )
                    bulk_mask_main = (
                        mask_main & ~selected_main[:, None, :]
                    )
                    bulk_mask_tail = (
                        mask_tail & ~selected_tail[:, None, :]
                    )
                elif not CLUSTER_MASS_SELECTOR:
                    selected_main = tl.full(
                        (BLOCK_J, 64), False, tl.int1
                    )
                    selected_tail = tl.full(
                        (BLOCK_J, 16), False, tl.int1
                    )
                    selected_cols_by_rank = tl.zeros(
                        (BLOCK_J, EXTRA_BLOCK), tl.int32
                    )
                    for rank in tl.static_range(0, TOP_R):
                        rank_main_value, rank_main_index = tl.max(
                            tl.where(
                                ~selected_main,
                                key_leverage_main,
                                -float("inf"),
                            ),
                            axis=1,
                            return_indices=True,
                            return_indices_tie_break_left=True,
                        )
                        rank_tail_value, rank_tail_index = tl.max(
                            tl.where(
                                ~selected_tail,
                                key_leverage_tail,
                                -float("inf"),
                            ),
                            axis=1,
                            return_indices=True,
                            return_indices_tie_break_left=True,
                        )
                        rank_is_tail = rank_tail_value > rank_main_value
                        rank_col = tl.where(
                            rank_is_tail,
                            64 + rank_tail_index,
                            rank_main_index,
                        )
                        selected_main = selected_main | (
                            main_cols[None, :] == rank_col[:, None]
                        )
                        selected_tail = selected_tail | (
                            tail_cols[None, :] == rank_col[:, None]
                        )
                        selected_cols_by_rank = tl.where(
                            extra_lanes[None, :] == rank,
                            rank_col[:, None],
                            selected_cols_by_rank,
                        )
                    bulk_mask_main = mask_main & ~selected_main[:, None, :]
                    bulk_mask_tail = mask_tail & ~selected_tail[:, None, :]
                if CLUSTER_MASS_SELECTOR:
                    bulk_weights_main = parent_weights_main
                    bulk_weights_tail = parent_weights_tail
                    parent_log_norm = tl.log2(safe_parent_sum)
                    parent_bias = tl.sum(
                        parent_weights_main
                        * tl.where(
                            mask_main,
                            winner_main[:, None, :] - parent_log_norm[:, :, None],
                            0.0,
                        ),
                        axis=2,
                    )
                    parent_bias += tl.sum(
                        parent_weights_tail
                        * tl.where(
                            mask_tail,
                            winner_tail[:, None, :] - parent_log_norm[:, :, None],
                            0.0,
                        ),
                        axis=2,
                    )
                    parent_bias *= 0.6931471805599453
                    residual_weight = 1.0 - selected_weight
                    bulk_valid = parent_valid & (residual_weight > 1.0e-6)
                    safe_residual = tl.where(bulk_valid, residual_weight, 1.0)
                    selected_w_log_w = selected_weight * tl.log2(
                        tl.maximum(selected_weight, 1.0e-20)
                    ) * 0.6931471805599453
                    bulk_bias = (
                        (parent_bias - selected_w_log_w) / safe_residual
                        - tl.log2(safe_residual) * 0.6931471805599453
                    )
                    bulk_bias = tl.where(bulk_valid, bulk_bias, float("inf"))
                else:
                    bulk_probs_main = tl.where(bulk_mask_main, probs_main, 0.0)
                    bulk_probs_tail = tl.where(bulk_mask_tail, probs_tail, 0.0)
                    bulk_sum = tl.sum(bulk_probs_main, axis=2) + tl.sum(
                        bulk_probs_tail, axis=2
                    )
                    bulk_valid = bulk_sum > 0.0
                    safe_bulk_sum = tl.where(bulk_valid, bulk_sum, 1.0)
                    bulk_weights_main = bulk_probs_main / safe_bulk_sum[:, :, None]
                    bulk_weights_tail = bulk_probs_tail / safe_bulk_sum[:, :, None]
                    bulk_log_norm = tl.log2(safe_bulk_sum)
                    bulk_bias = tl.sum(
                        bulk_weights_main
                        * tl.where(
                            bulk_mask_main,
                            winner_main[:, None, :] - bulk_log_norm[:, :, None],
                            0.0,
                        ),
                        axis=2,
                    )
                    bulk_bias += tl.sum(
                        bulk_weights_tail
                        * tl.where(
                            bulk_mask_tail,
                            winner_tail[:, None, :] - bulk_log_norm[:, :, None],
                            0.0,
                        ),
                        axis=2,
                    )
                    bulk_bias *= 0.6931471805599453
                    bulk_bias = tl.where(bulk_valid, bulk_bias, float("inf"))

                bulk_pair = tl.dot(
                    tl.reshape(bulk_weights_main, (BLOCK_J * C, 64)).to(
                        key_main.dtype
                    ),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                bulk_pair += tl.dot(
                    tl.reshape(bulk_weights_tail, (BLOCK_J * C, 16)).to(
                        key_tail.dtype
                    ),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                bulk_key, bulk_value = tl.split(
                    tl.reshape(bulk_pair, (BLOCK_J * C, 128, 2))
                )

                selected_cols = tl.reshape(
                    selected_cols_by_rank, (BLOCK_J * EXTRA_BLOCK,)
                )
                extra_rows = tl.arange(0, BLOCK_J * EXTRA_BLOCK)
                extra_rank = extra_rows % EXTRA_BLOCK
                valid_extra = extra_rank < TOP_R
                singleton_offsets = (
                    row_base
                    + (
                        selected_cols[:, None] * TOTAL_HEADS + global_head
                    )
                    * D_HEAD
                    + dims[None, :]
                )
                singleton_key = tl.load(
                    k_ptr + singleton_offsets,
                    mask=valid_extra[:, None],
                    other=0.0,
                )
                singleton_value = tl.load(
                    v_ptr + singleton_offsets,
                    mask=valid_extra[:, None],
                    other=0.0,
                )
                if CLUSTER_MASS_SELECTOR:
                    selected_weight_flat = tl.reshape(
                        selected_weight, (BLOCK_J * C,)
                    )
                    safe_residual_flat = tl.reshape(
                        safe_residual, (BLOCK_J * C,)
                    )
                    bulk_key = (
                        bulk_key
                        - selected_weight_flat[:, None] * singleton_key
                    ) / safe_residual_flat[:, None]
                    bulk_value = (
                        bulk_value
                        - selected_weight_flat[:, None] * singleton_value
                    ) / safe_residual_flat[:, None]

                key_block = key_outer * ROWS + key_row
                bulk_output_token = key_block * OUT_C + parent_ids
                bulk_route = (
                    (((batch_i64 * Q_OUTER + query_outer) * COLS
                      + parent_query_cols)
                     * ACTIVE_HEADS
                     + head)
                    * K_OUTER
                    * ROWS
                    * OUT_C
                )
                bulk_output = (
                    bulk_route[:, None] + bulk_output_token[:, None]
                ) * D_HEAD + dims[None, :]
                tl.store(merged_k_ptr + bulk_output, bulk_key)
                tl.store(merged_v_ptr + bulk_output, bulk_value)
                tl.store(
                    bias_ptr + bulk_route + bulk_output_token,
                    tl.reshape(bulk_bias, (BLOCK_J * C,)),
                )

                extra_query_cols = query_col_start + extra_rows // EXTRA_BLOCK
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
                tl.store(
                    merged_k_ptr + extra_output,
                    singleton_key,
                    mask=valid_extra[:, None],
                )
                tl.store(
                    merged_v_ptr + extra_output,
                    singleton_value,
                    mask=valid_extra[:, None],
                )
                extra_bias = tl.zeros(
                    (BLOCK_J * EXTRA_BLOCK,), tl.float32
                )
                if CLUSTER_MASS_SELECTOR:
                    extra_bias = tl.where(
                        tl.reshape(parent_valid, (BLOCK_J * C,)),
                        0.0,
                        float("inf"),
                    )
                tl.store(
                    bias_ptr + extra_route + extra_output_token,
                    extra_bias,
                    mask=valid_extra,
                )

    @triton.jit
    def _cluster_mass_correction_kernel(
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        selected_col_ptr,
        selected_weight_ptr,
        PARENT_TOKENS: tl.constexpr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        OUT_C: tl.constexpr,
        BLOCK_PARENTS: tl.constexpr,
    ):
        parent = tl.program_id(0) * BLOCK_PARENTS + tl.arange(0, BLOCK_PARENTS)
        valid = parent < PARENT_TOKENS
        parent_i64 = parent.to(tl.int64)
        route = parent_i64 // C
        cluster = parent_i64 % C

        index = route
        key_row = index % ROWS
        index //= ROWS
        key_outer = index % K_OUTER
        index //= K_OUTER
        head = index % ACTIVE_HEADS
        index //= ACTIVE_HEADS
        query_col = index % COLS
        index //= COLS
        query_outer = index % Q_OUTER
        batch = index // Q_OUTER

        selected_col = tl.load(
            selected_col_ptr + parent_i64, mask=valid, other=0
        ).to(tl.int64)
        selected_weight = tl.load(
            selected_weight_ptr + parent_i64, mask=valid, other=0.0
        ).to(tl.float32)
        residual = 1.0 - selected_weight
        bulk_valid = valid & (residual > 1.0e-6)
        singleton_valid = valid & (selected_weight > 0.0)
        safe_residual = tl.where(bulk_valid, residual, 1.0)

        dims = tl.arange(0, 128)
        bulk_token = route * OUT_C + cluster
        singleton_token = route * OUT_C + C + cluster
        bulk_offset = bulk_token[:, None] * D_HEAD + dims[None, :]
        singleton_offset = singleton_token[:, None] * D_HEAD + dims[None, :]
        centroid_key = tl.load(
            merged_k_ptr + bulk_offset, mask=valid[:, None], other=0.0
        ).to(tl.float32)
        centroid_value = tl.load(
            merged_v_ptr + bulk_offset, mask=valid[:, None], other=0.0
        ).to(tl.float32)

        global_head = HEAD_START + head
        raw_token = (
            (((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS
              + selected_col)
            * TOTAL_HEADS
            + global_head
        )
        raw_offset = raw_token[:, None] * D_HEAD + dims[None, :]
        singleton_key = tl.load(
            k_ptr + raw_offset, mask=singleton_valid[:, None], other=0.0
        ).to(tl.float32)
        singleton_value = tl.load(
            v_ptr + raw_offset, mask=singleton_valid[:, None], other=0.0
        ).to(tl.float32)
        bulk_key = (
            centroid_key - selected_weight[:, None] * singleton_key
        ) / safe_residual[:, None]
        bulk_value = (
            centroid_value - selected_weight[:, None] * singleton_value
        ) / safe_residual[:, None]
        tl.store(merged_k_ptr + bulk_offset, bulk_key, mask=bulk_valid[:, None])
        tl.store(merged_v_ptr + bulk_offset, bulk_value, mask=bulk_valid[:, None])
        tl.store(
            merged_k_ptr + singleton_offset,
            singleton_key,
            mask=singleton_valid[:, None],
        )
        tl.store(
            merged_v_ptr + singleton_offset,
            singleton_value,
            mask=singleton_valid[:, None],
        )

        parent_bias = tl.load(bias_ptr + bulk_token, mask=valid, other=0.0)
        selected_w_log_w = selected_weight * tl.log2(
            tl.maximum(selected_weight, 1.0e-20)
        ) * 0.6931471805599453
        bulk_bias = (
            (parent_bias - selected_w_log_w) / safe_residual
            - tl.log2(safe_residual) * 0.6931471805599453
        )
        tl.store(
            bias_ptr + bulk_token,
            tl.where(bulk_valid, bulk_bias, float("inf")),
            mask=valid,
        )
        tl.store(
            bias_ptr + singleton_token,
            tl.where(singleton_valid, 0.0, float("inf")),
            mask=valid,
        )


def _hard_c4_720p_singleton_leverage_attention_triton_blocks(
    q,
    k,
    v,
    *,
    top_r,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    exact_intra_frame=False,
    query_outer_offset=0,
    temporal_shared_seed_rows=None,
    timings=None,
    selector="routing_leverage",
    block_j=SINGLETON_BLOCK_J,
):
    """Run a legacy singleton experiment outside the C4/C8 contract."""

    require_exact_merged_capacity(SINGLETON_C, top_r)
    if triton is None:
        raise RuntimeError("Triton is required for singleton Hard-C4.")
    scale = q.size(-1) ** -0.5 if scale is None else scale
    top_r = int(top_r)
    selector = str(selector).lower()
    if selector not in {
        "routing_leverage",
        "mass",
        "cluster_mass",
        "query_column",
    }:
        raise ValueError(
            "singleton selector must be 'routing_leverage', 'mass', "
            "'cluster_mass', or 'query_column'."
        )
    if selector == "query_column" and top_r != 1:
        raise ValueError("query_column currently requires R=1")
    if selector == "cluster_mass" and top_r != SINGLETON_C:
        raise ValueError("cluster_mass requires one singleton per C4 cluster.")
    block_j = int(block_j)
    if block_j not in {4, 8, 16}:
        raise ValueError("singleton merge block_j must be 4, 8, or 16.")
    if top_r not in SINGLETON_TOP_RS:
        raise ValueError(
            "singleton Hard-C4 supports 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, "
            "or 32 atoms."
        )
    extra_block = 1 << (top_r - 1).bit_length()
    out_c = SINGLETON_C + top_r
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, query_outer, rows, cols, total_heads, dim = q.shape
    key_outer = k.size(1)
    active_heads = int(head_count or total_heads)
    use_external_seeds = temporal_shared_seed_rows is not None
    if (rows, cols, dim) != (
        hard_c4.HARD_720_ROWS,
        hard_c4.HARD_720_COLS,
        128,
    ):
        raise ValueError("singleton Hard-C4 requires 45x80 tokens with D=128.")
    if query_outer % hard_c4.HARD_720_QUERY_OUTER_GROUP:
        raise ValueError("singleton Hard-C4 requires complete 21-frame chunks.")
    query_groups = query_outer // hard_c4.HARD_720_QUERY_OUTER_GROUP
    if use_external_seeds:
        if temporal_shared_seed_rows.device != q.device:
            raise ValueError("singleton seed rows must be on the QKV device.")
        if temporal_shared_seed_rows.ndim != 7:
            raise ValueError(
                "singleton seed rows must be "
                "[B, R, column, active_head, K, row, C]."
            )
        router_frames = int(temporal_shared_seed_rows.size(1))
        expected = (
            batch,
            router_frames,
            cols,
            active_heads,
            key_outer,
            rows,
            SINGLETON_C,
        )
        if tuple(temporal_shared_seed_rows.shape) != expected:
            raise ValueError(
                f"singleton seed rows must be {expected}, got "
                f"{tuple(temporal_shared_seed_rows.shape)}."
            )
        temporal_shared_seed_rows = (
            temporal_shared_seed_rows.to(torch.uint8).contiguous()
        )
    else:
        router_frames = 1
        temporal_shared_seed_rows = q

    compressed_tokens = (
        batch
        * query_outer
        * cols
        * active_heads
        * key_outer
        * rows
        * out_c
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

    merge_grid = (query_groups * key_outer * rows, batch * active_heads)
    timing_events = None
    if timings is not None:
        timing_events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        timing_events[0].record()
    _singleton_c4_leverage_merge_kernel[merge_grid](
        q,
        temporal_shared_seed_rows,
        k,
        v,
        merged_k,
        merged_v,
        bias,
        query_outer,
        query_groups,
        hard_c4.HARD_720_QUERY_OUTER_GROUP,
        router_frames,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        rows,
        cols,
        SINGLETON_C,
        top_r,
        out_c,
        extra_block,
        block_j,
        hard_c4.HARD_720_MERGE_LOOP_STAGES,
        hard_c4.HARD_720_INPUT_PRECISION,
        bool(use_external_seeds),
        selector == "mass",
        selector == "cluster_mass",
        selector == "query_column",
        SCALE=float(scale),
        num_warps=hard_c4.HARD_720_MERGE_NUM_WARPS,
        num_stages=hard_c4.HARD_720_MERGE_NUM_STAGES,
    )
    if timing_events is not None:
        timing_events[1].record()

    route_loop_stages = 1 if torch.version.hip is not None else 3
    route_grid = (1, batch * active_heads, query_outer * cols)
    unused_distortion = torch.empty(1, device=q.device, dtype=torch.float32)
    hard_c4._hard_c4_720p_route_kernel[route_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        unused_compressed_scale,
        unused_compressed_scale,
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
        out_c,
        hard_c4.HARD_720_ROUTE_BLOCK_I,
        hard_c4.HARD_720_ROUTE_BLOCK_TOKENS,
        route_loop_stages,
        hard_c4.HARD_720_INPUT_PRECISION,
        bool(exact_intra_frame),
        0,
        False,
        int(query_outer_offset),
        SINGLETON_TOP_R=0,
        SINGLETON_QUERY_COLUMN=False,
        SINGLETON_SECONDARY_FRAME_STRIDE=1,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
        QUERY_COLUMN_MAJOR=False,
        PACKED_SINGLETON_KV=False,
        SCALED_FP8_KV=False,
        COMPRESSED_FP8=False,
        SCALE=float(scale),
        num_warps=hard_c4.HARD_720_ROUTE_NUM_WARPS,
        num_stages=hard_c4.HARD_720_ROUTE_NUM_STAGES,
    )
    if timing_events is not None:
        timing_events[2].record()
        timing_events[2].synchronize()
        timings["merge_ms"] = timing_events[0].elapsed_time(timing_events[1])
        timings["route_ms"] = timing_events[1].elapsed_time(timing_events[2])
    return output


def _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    top_r,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    query_outer_offset=0,
    timings=None,
    block_j=SINGLETON_BLOCK_J,
    bias_only=False,
    query_column_selector=False,
    defer_correction=False,
    compressed_fp8_kv=False,
    scaled_fp8_kv=False,
    mean_centered_routing=False,
    gaussian_partition_routing=False,
    reuse_partition_exp=False,
    direct_bulk_normalization=False,
    fp8_score_dot=False,
    fp8_merge_k_dot=False,
    fp8_merge_v_dot=False,
    query_frame_group=21,
    split_merge=False,
    materialize_singleton=False,
    query_column_major=False,
    query_column_secondary_frame_stride=1,
    query_column_primary_row_mod4_omit2=False,
    packed_singleton_kv=False,
    seed_rows_native_layout=False,
    merge_frame_group=1,
    merge_col_group=1,
    merge_loop_stages=1,
    route_loop_stages=None,
    merge_disable_licm=None,
    split_kv_accum=False,
):
    """Run a legacy singleton experiment outside the C4/C8 contract."""

    require_exact_merged_capacity(SINGLETON_C, top_r)
    if triton is None:
        raise RuntimeError("Triton is required for singleton Hard-C4")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA/ROCm device")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("global-mass singleton Hard-C4 requires bfloat16 q/k/v")
    if q.ndim != 6 or k.ndim != 6 or v.ndim != 6:
        raise ValueError("global-mass singleton Hard-C4 expects blocked rank-6 q/k/v")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v shapes must match")

    top_r = int(top_r)
    if top_r not in (1, 2, 3, 4):
        raise ValueError("optimized global-mass singleton path supports R=1..4")
    block_j = int(block_j)
    if block_j not in (4, 8, 16):
        raise ValueError("global-mass singleton block_j must be 4, 8, or 16")
    if not query_column_selector and block_j != 16:
        raise ValueError("split-index global-mass path requires block_j=16")
    compressed_fp8_kv = bool(compressed_fp8_kv)
    scaled_fp8_kv = bool(scaled_fp8_kv)
    mean_centered_routing = bool(mean_centered_routing)
    gaussian_partition_routing = bool(gaussian_partition_routing)
    reuse_partition_exp = bool(reuse_partition_exp)
    direct_bulk_normalization = bool(direct_bulk_normalization)
    fp8_score_dot = bool(fp8_score_dot)
    fp8_merge_k_dot = bool(fp8_merge_k_dot)
    fp8_merge_v_dot = bool(fp8_merge_v_dot)
    if compressed_fp8_kv and scaled_fp8_kv:
        raise ValueError("compressed FP8 K/V cannot also use per-token scaling")
    if mean_centered_routing and gaussian_partition_routing:
        raise ValueError(
            "mean-centered and Gaussian-partition routing are mutually exclusive"
        )
    if reuse_partition_exp and (
        mean_centered_routing or gaussian_partition_routing
    ):
        raise ValueError(
            "partition-exp reuse requires exact log-partition routing"
        )
    split_merge = bool(split_merge)
    materialize_singleton = bool(materialize_singleton)
    packed_singleton_kv = bool(packed_singleton_kv)
    seed_rows_native_layout = bool(seed_rows_native_layout)
    merge_frame_group = int(merge_frame_group)
    if merge_frame_group not in (1, 3, 7, 21):
        raise ValueError("merge_frame_group must be 1, 3, 7, or 21")
    merge_col_group = int(merge_col_group)
    if merge_col_group not in (1, 2, 4, 8, 16):
        raise ValueError("merge_col_group must be 1, 2, 4, 8, or 16")
    merge_loop_stages = int(merge_loop_stages)
    if merge_loop_stages not in (1, 2, 3, 4):
        raise ValueError("merge_loop_stages must be 1, 2, 3, or 4")
    if route_loop_stages is None:
        route_loop_stages = 1 if torch.version.hip is not None else 3
    route_loop_stages = int(route_loop_stages)
    if route_loop_stages not in (1, 2, 3, 4):
        raise ValueError("route_loop_stages must be 1, 2, 3, or 4")
    if merge_disable_licm is None:
        merge_disable_licm = hard_c4.HARD_720_MERGE_DISABLE_LICM
    merge_disable_licm = bool(merge_disable_licm)
    split_kv_accum = bool(split_kv_accum)
    query_column_major = bool(query_column_major or packed_singleton_kv)
    query_column_secondary_frame_stride = int(
        query_column_secondary_frame_stride
    )
    query_column_primary_row_mod4_omit2 = bool(
        query_column_primary_row_mod4_omit2
    )
    if query_column_secondary_frame_stride < 1:
        raise ValueError("query-column secondary frame stride must be positive")
    if query_column_secondary_frame_stride > 1 and (
        not query_column_selector
        or top_r != 2
        or materialize_singleton
        or packed_singleton_kv
        or scaled_fp8_kv
        or defer_correction
        or bias_only
    ):
        raise ValueError(
            "selective query-column R2 requires the exact raw singleton path"
        )
    if query_column_primary_row_mod4_omit2 and (
        not query_column_selector
        or top_r != 1
        or query_column_secondary_frame_stride != 1
        or materialize_singleton
        or packed_singleton_kv
        or scaled_fp8_kv
        or defer_correction
        or bias_only
        or split_merge
    ):
        raise ValueError(
            "selective query-column R1 requires the exact raw singleton path"
        )
    if fp8_score_dot and (
        torch.version.hip is not None
        or torch.cuda.get_device_capability(q.device)[0] < 9
    ):
        raise ValueError("FP8 score dot requires an NVIDIA compute capability 9+ GPU")
    if (fp8_merge_k_dot or fp8_merge_v_dot) and (
        torch.version.hip is not None
        or torch.cuda.get_device_capability(q.device)[0] < 9
    ):
        raise ValueError("FP8 merge dot requires an NVIDIA compute capability 9+ GPU")
    if (fp8_merge_k_dot or fp8_merge_v_dot) and not split_kv_accum:
        raise ValueError("FP8 merge dot requires split K/V accumulation")
    if compressed_fp8_kv and (
        torch.version.hip is not None
        or torch.cuda.get_device_capability(q.device)[0] < 9
    ):
        raise ValueError(
            "compressed FP8 K/V requires an NVIDIA compute capability 9+ GPU"
        )
    if scaled_fp8_kv and (
        torch.version.hip is not None
        or not query_column_selector
        or top_r != 1
        or defer_correction
        or bias_only
    ):
        raise ValueError(
            "scaled FP8 K/V requires NVIDIA query-column R1 exact correction"
        )
    if materialize_singleton and (
        not query_column_selector
        or top_r != 1
        or bias_only
        or defer_correction
        or scaled_fp8_kv
        or split_merge
        or int(query_frame_group) != 21
    ):
        raise ValueError(
            "materialized singleton requires the shared query-column R1 path"
        )
    if packed_singleton_kv and (
        not query_column_selector
        or top_r != 1
        or bias_only
        or defer_correction
        or scaled_fp8_kv
        or split_merge
        or materialize_singleton
        or int(query_frame_group) != 21
    ):
        raise ValueError(
            "packed singleton K/V requires the shared query-column R1 path"
        )
    if query_column_major and (
        not query_column_selector
        or bias_only
        or defer_correction
        or split_merge
        or materialize_singleton
        or int(query_frame_group) != 21
    ):
        raise ValueError(
            "query-column-major routing requires the exact shared query-column path"
        )
    if split_kv_accum and (
        not query_column_selector
        or bias_only
        or defer_correction
        or split_merge
        or materialize_singleton
        or (packed_singleton_kv and not getattr(hard_c4, "PACKED_SPLIT_KV_SUPPORTED", False))
        or int(query_frame_group) != 21
    ):
        raise ValueError(
            "split K/V accumulation requires the exact shared query-column path"
        )
    if seed_rows_native_layout and (
        not query_column_selector
        or split_merge
        or int(query_frame_group) != 21
    ):
        raise ValueError(
            "native seed layout requires the shared query-column merge path"
        )
    if scale is None:
        scale = q.size(-1) ** -0.5
    batch, query_outer, rows, cols, total_heads, dim = q.shape
    key_outer = k.size(1)
    router_frames = int(seed_rows.size(1))
    query_frame_group = int(query_frame_group)
    if query_frame_group not in (1, 3, 7, 21):
        raise ValueError("query_frame_group must be 1, 3, 7, or 21")
    if query_outer % query_frame_group:
        raise ValueError("query_frame_group must divide query_outer")
    if query_outer % merge_frame_group:
        raise ValueError("merge_frame_group must divide query_outer")
    if cols % merge_col_group:
        raise ValueError("merge_col_group must divide the query columns")
    if merge_frame_group > 1 and (
        router_frames != 1
        or not query_column_selector
        or top_r != 1
        or bias_only
        or defer_correction
        or scaled_fp8_kv
        or split_merge
        or materialize_singleton
        or query_frame_group != query_outer
    ):
        raise ValueError(
            "temporal merge-weight sharing requires the T1 query-column R1 path"
        )
    if merge_col_group > 1 and (
        router_frames != 1
        or not query_column_selector
        or top_r != 1
        or bias_only
        or defer_correction
        or scaled_fp8_kv
        or split_merge
        or materialize_singleton
        or query_frame_group != query_outer
    ):
        raise ValueError(
            "spatial merge-weight sharing requires the T1 query-column R1 path"
        )
    if seed_rows_native_layout and query_frame_group < query_outer:
        raise ValueError("grouped query-frame merge does not support native seed layout")
    if (rows, cols, dim) != (
        hard_c4.HARD_720_ROWS,
        hard_c4.HARD_720_COLS,
        128,
    ):
        raise ValueError("global-mass singleton path is specialized for Wan 720p")
    if head_count is None:
        head_count = total_heads - int(head_start)
    active_heads = int(head_count)
    if seed_rows_native_layout:
        expected = (
            batch,
            router_frames,
            key_outer,
            rows,
            cols,
            total_heads,
            SINGLETON_C,
        )
    else:
        expected = (
            batch,
            router_frames,
            cols,
            active_heads,
            key_outer,
            rows,
            SINGLETON_C,
        )
    if tuple(seed_rows.shape) != expected:
        raise ValueError(f"global-mass singleton rows must be {expected}")
    seed_rows = seed_rows.to(torch.uint8).contiguous()

    out_c = SINGLETON_C + (top_r if materialize_singleton else 0)
    extra_block = 1 << (top_r - 1).bit_length()
    route_count = batch * query_outer * cols * active_heads * key_outer * rows
    compressed_tokens = route_count * out_c
    compressed_dtype = (
        torch.float8_e4m3fn
        if scaled_fp8_kv or compressed_fp8_kv
        else q.dtype
    )
    merged_k = torch.empty(
        (compressed_tokens, dim), device=q.device, dtype=compressed_dtype
    )
    merged_v = torch.empty_like(merged_k)
    merged_k_scale = torch.empty(
        compressed_tokens if scaled_fp8_kv else 1,
        device=q.device,
        dtype=torch.float32,
    )
    merged_v_scale = torch.empty_like(merged_k_scale)
    bias = torch.empty(compressed_tokens, device=q.device, dtype=torch.float32)
    if packed_singleton_kv:
        packed_shape = (
            batch,
            cols,
            active_heads,
            key_outer,
            rows,
            dim,
        )
        route_key = torch.empty(packed_shape, device=q.device, dtype=q.dtype)
        route_value = torch.empty_like(route_key)
        # These merge-kernel outputs are otherwise unused by the exact
        # query-column path, so reuse the pointer slots for fused packing.
        selected_cols = route_key
        metadata = route_value
    else:
        route_key = k
        route_value = v
        selected_cols = torch.empty(
            1 if query_column_selector else route_count * top_r,
            device=q.device,
            dtype=torch.uint8,
        )
        metadata = (
            torch.empty(route_count, device=q.device, dtype=torch.float32)
            if defer_correction
            else torch.empty(1, device=q.device, dtype=torch.float32)
        )
    unused = torch.empty(1, device=q.device, dtype=torch.float32)
    split_weights = (
        torch.empty(
            batch
            * cols
            * active_heads
            * key_outer
            * rows
            * SINGLETON_C
            * cols,
            device=q.device,
            dtype=q.dtype,
        )
        if split_merge
        else None
    )
    if output is None:
        output = torch.empty_like(q)

    events = None
    if timings is not None:
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        events[0].record()
    if split_merge:
        if (
            router_frames != 1
            or not query_column_selector
            or top_r != 1
            or bias_only
            or defer_correction
            or scaled_fp8_kv
            or mean_centered_routing
            or gaussian_partition_routing
            or reuse_partition_exp
            or fp8_score_dot
            or query_frame_group != query_outer
        ):
            raise ValueError(
                "split merge requires Hellinger T1 query-column R1 exact "
                "correction"
            )
        split_grid = (
            key_outer * rows * (cols // int(block_j)),
            batch * active_heads,
        )
        for query_outer_index in range(query_outer):
            hard_c4._hard_c4_720p_query_column_weight_kernel[split_grid](
                q,
                seed_rows,
                k,
                split_weights,
                bias,
                query_outer_index,
                query_outer,
                key_outer,
                total_heads,
                active_heads,
                int(head_start),
                dim,
                rows,
                cols,
                SINGLETON_C,
                int(block_j),
                hard_c4.HARD_720_INPUT_PRECISION,
                SCALE=float(scale),
                num_warps=hard_c4.HARD_720_MERGE_NUM_WARPS,
                num_stages=hard_c4.HARD_720_MERGE_NUM_STAGES,
            )
            hard_c4._hard_c4_720p_query_column_weighted_merge_kernel[
                split_grid
            ](
                split_weights,
                k,
                v,
                merged_k,
                merged_v,
                query_outer_index,
                query_outer,
                key_outer,
                total_heads,
                active_heads,
                int(head_start),
                dim,
                rows,
                cols,
                SINGLETON_C,
                int(block_j),
                hard_c4.HARD_720_INPUT_PRECISION,
                num_warps=hard_c4.HARD_720_MERGE_NUM_WARPS,
                num_stages=hard_c4.HARD_720_MERGE_NUM_STAGES,
            )
    elif query_frame_group < query_outer:
        if (
            router_frames != 1
            or not query_column_selector
            or top_r != 1
            or bias_only
            or defer_correction
            or scaled_fp8_kv
            or mean_centered_routing
            or gaussian_partition_routing
            or reuse_partition_exp
            or fp8_score_dot
        ):
            raise ValueError(
                "grouped query-frame merge requires Hellinger T1 query-column "
                "R1 exact correction"
            )
        grouped_grid = (
            key_outer * rows * (query_outer // query_frame_group),
            batch * active_heads,
        )
        hard_c4._hard_c4_720p_query_column_grouped_merge_kernel[grouped_grid](
            q,
            seed_rows,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            query_outer,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            rows,
            cols,
            SINGLETON_C,
            int(block_j),
            query_frame_group,
            hard_c4.HARD_720_INPUT_PRECISION,
            SCALE=float(scale),
            num_warps=hard_c4.HARD_720_MERGE_NUM_WARPS,
            num_stages=hard_c4.HARD_720_MERGE_NUM_STAGES,
        )
    else:
        merge_grid = (key_outer * rows, batch * active_heads)
        hard_c4._hard_c4_720p_temporal_shared_merge_kernel[merge_grid](
            q,
            seed_rows,
            q,
            k,
            v,
            merged_k,
            merged_v,
            merged_k_scale,
            merged_v_scale,
            bias,
            selected_cols,
            metadata,
            query_outer,
            router_frames,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            rows,
            cols,
            SINGLETON_C,
            out_c,
            int(block_j),
            merge_loop_stages,
            merge_frame_group,
            merge_col_group,
            merge_disable_licm,
            hard_c4.HARD_720_INPUT_PRECISION,
            True,
            ROUTE_BIAS_SCALE=1.0,
            LOGIT_VARIANCE_BIAS_SCALE=0.0,
            STORE_CLUSTER_MASS=False,
            FUSE_CLUSTER_MASS=False,
            CORRECT_CLUSTER_VECTORS=False,
            GLOBAL_MASS_TOP_R=top_r,
            GLOBAL_MASS_EXTRA_BLOCK=extra_block,
            MATERIALIZE_GLOBAL_MASS=materialize_singleton,
            GLOBAL_MASS_BIAS_ONLY=bool(bias_only),
            GLOBAL_QUERY_COLUMN_SELECTOR=bool(query_column_selector),
            GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE=(
                query_column_secondary_frame_stride
            ),
            GLOBAL_QUERY_COLUMN_PRIMARY_ROW_MOD4_OMIT2=(
                query_column_primary_row_mod4_omit2
            ),
            GLOBAL_MASS_DEFER_CORRECTION=bool(defer_correction),
            MEAN_CENTERED_ROUTING=mean_centered_routing,
            GAUSSIAN_PARTITION_ROUTING=gaussian_partition_routing,
            REUSE_PARTITION_EXP=reuse_partition_exp,
            DIRECT_BULK_NORMALIZATION=direct_bulk_normalization,
            FP8_SCORE_DOT=fp8_score_dot,
            FP8_MERGE_K_DOT=fp8_merge_k_dot,
            FP8_MERGE_V_DOT=fp8_merge_v_dot,
            SCALED_FP8_KV=scaled_fp8_kv,
            SPLIT_KV_ACCUM=split_kv_accum,
            STORE_PACKED_SINGLETON_KV=packed_singleton_kv,
            SEED_ROWS_NATIVE_LAYOUT=seed_rows_native_layout,
            USE_TEMPORAL_ROW_MAP=False,
            SCALE=float(scale),
            num_warps=hard_c4.HARD_720_MERGE_NUM_WARPS,
            num_stages=hard_c4.HARD_720_MERGE_NUM_STAGES,
        )
    if events is not None:
        events[1].record()

    route_grid = (1, batch * active_heads, query_outer * cols)
    if defer_correction:
        hard_c4._hard_c4_720p_query_column_deferred_route_kernel[route_grid](
            q,
            route_key,
            route_value,
            merged_k,
            merged_v,
            bias,
            metadata,
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
            out_c,
            hard_c4.HARD_720_ROUTE_BLOCK_I,
            hard_c4.HARD_720_ROUTE_BLOCK_TOKENS,
            1,
            hard_c4.HARD_720_INPUT_PRECISION,
            SCALE=float(scale),
            num_warps=hard_c4.HARD_720_ROUTE_NUM_WARPS,
            num_stages=hard_c4.HARD_720_ROUTE_NUM_STAGES,
        )
    else:
        hard_c4._hard_c4_720p_route_kernel[route_grid](
            q,
            route_key,
            route_value,
            merged_k,
            merged_v,
            merged_k_scale,
            merged_v_scale,
            bias,
            selected_cols,
            unused,
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
            out_c,
            hard_c4.HARD_720_ROUTE_BLOCK_I,
            hard_c4.HARD_720_ROUTE_BLOCK_TOKENS,
            route_loop_stages,
            hard_c4.HARD_720_INPUT_PRECISION,
            False,
            0,
            False,
            int(query_outer_offset),
            SINGLETON_TOP_R=0 if materialize_singleton else top_r,
            SINGLETON_QUERY_COLUMN=(
                bool(query_column_selector) and not materialize_singleton
            ),
            SINGLETON_SECONDARY_FRAME_STRIDE=(
                query_column_secondary_frame_stride
            ),
            SINGLETON_PRIMARY_ROW_MOD4_OMIT2=(
                query_column_primary_row_mod4_omit2
            ),
            QUERY_COLUMN_MAJOR=query_column_major,
            PACKED_SINGLETON_KV=packed_singleton_kv,
            SCALED_FP8_KV=scaled_fp8_kv,
            COMPRESSED_FP8=compressed_fp8_kv,
            SCALE=float(scale),
            num_warps=hard_c4.HARD_720_ROUTE_NUM_WARPS,
            num_stages=hard_c4.HARD_720_ROUTE_NUM_STAGES,
        )
    if events is not None:
        events[2].record()
        events[2].synchronize()
        timings["merge_ms"] = events[0].elapsed_time(events[1])
        timings["route_ms"] = events[1].elapsed_time(events[2])
    return output


def _hard_c4_720p_temporal_shared_cluster_mass_staged_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    *,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    query_outer_offset=0,
    timings=None,
):
    """Run exact one-per-cluster peeling as merge, correction, and route stages."""

    if triton is None:
        raise RuntimeError("Triton is required for staged cluster-mass C4.")
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
        raise ValueError("staged cluster-mass C4 requires 45x80 tokens with D=128.")
    if query_outer != hard_c4.HARD_720_QUERY_OUTER_GROUP:
        raise ValueError("staged cluster-mass C4 requires exactly 21 frames.")
    if seed_rows.ndim != 7 or seed_rows.device != q.device:
        raise ValueError(
            "staged cluster-mass rows must be on-device "
            "[B, R, column, active_head, K, row, C]."
        )
    router_frames = int(seed_rows.size(1))
    expected = (
        batch,
        router_frames,
        cols,
        active_heads,
        key_outer,
        rows,
        SINGLETON_C,
    )
    if tuple(seed_rows.shape) != expected:
        raise ValueError(f"staged cluster-mass rows must be {expected}.")
    seed_rows = seed_rows.to(torch.uint8).contiguous()

    out_c = 2 * SINGLETON_C
    route_count = batch * query_outer * cols * active_heads * key_outer * rows
    parent_tokens = route_count * SINGLETON_C
    compressed_tokens = route_count * out_c
    compressed_fp8 = torch.version.hip is None
    compressed_dtype = torch.float8_e4m3fn if compressed_fp8 else q.dtype
    merged_k = torch.empty(
        (compressed_tokens, dim), device=q.device, dtype=compressed_dtype
    )
    merged_v = torch.empty_like(merged_k)
    unused_compressed_scale = torch.empty(
        1, device=q.device, dtype=torch.float32
    )
    bias = torch.empty(compressed_tokens, device=q.device, dtype=torch.float32)
    selected_col = torch.empty(1, device=q.device, dtype=torch.uint8)
    selected_weight = torch.empty(1, device=q.device, dtype=torch.float32)
    if output is None:
        output = torch.empty_like(q)

    events = None
    if timings is not None:
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        events[0].record()
    merge_grid = (key_outer * rows, batch * active_heads)
    hard_c4._hard_c4_720p_temporal_shared_merge_kernel[merge_grid](
        q,
        seed_rows,
        q,
        k,
        v,
        merged_k,
        merged_v,
        unused_compressed_scale,
        unused_compressed_scale,
        bias,
        selected_col,
        selected_weight,
        query_outer,
        router_frames,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        rows,
        cols,
        SINGLETON_C,
        out_c,
        hard_c4.HARD_720_MERGE_BLOCK_J,
        1,
        1,
        1,
        hard_c4.HARD_720_MERGE_DISABLE_LICM,
        hard_c4.HARD_720_INPUT_PRECISION,
        True,
        ROUTE_BIAS_SCALE=1.0,
        LOGIT_VARIANCE_BIAS_SCALE=0.0,
        STORE_CLUSTER_MASS=False,
        FUSE_CLUSTER_MASS=True,
        CORRECT_CLUSTER_VECTORS=False,
        GLOBAL_MASS_TOP_R=0,
        GLOBAL_MASS_EXTRA_BLOCK=1,
        MATERIALIZE_GLOBAL_MASS=False,
        GLOBAL_MASS_BIAS_ONLY=False,
        GLOBAL_QUERY_COLUMN_SELECTOR=False,
        GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE=1,
        GLOBAL_QUERY_COLUMN_PRIMARY_ROW_MOD4_OMIT2=False,
        GLOBAL_MASS_DEFER_CORRECTION=False,
        MEAN_CENTERED_ROUTING=False,
        GAUSSIAN_PARTITION_ROUTING=False,
        REUSE_PARTITION_EXP=False,
        DIRECT_BULK_NORMALIZATION=False,
        FP8_SCORE_DOT=False,
        FP8_MERGE_K_DOT=False,
        FP8_MERGE_V_DOT=False,
        SCALED_FP8_KV=False,
        SPLIT_KV_ACCUM=False,
        STORE_PACKED_SINGLETON_KV=False,
        SEED_ROWS_NATIVE_LAYOUT=False,
        USE_TEMPORAL_ROW_MAP=False,
        SCALE=float(scale),
        num_warps=hard_c4.HARD_720_MERGE_NUM_WARPS,
        num_stages=1,
    )
    if events is not None:
        events[1].record()

    route_loop_stages = 1 if torch.version.hip is not None else 3
    route_grid = (1, batch * active_heads, query_outer * cols)
    unused = torch.empty(1, device=q.device, dtype=torch.float32)
    hard_c4._hard_c4_720p_route_kernel[route_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        unused_compressed_scale,
        unused_compressed_scale,
        bias,
        unused,
        unused,
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
        out_c,
        hard_c4.HARD_720_ROUTE_BLOCK_I,
        hard_c4.HARD_720_ROUTE_BLOCK_TOKENS,
        route_loop_stages,
        hard_c4.HARD_720_INPUT_PRECISION,
        False,
        0,
        False,
        int(query_outer_offset),
        SINGLETON_TOP_R=0,
        SINGLETON_QUERY_COLUMN=False,
        SINGLETON_SECONDARY_FRAME_STRIDE=1,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
        QUERY_COLUMN_MAJOR=False,
        PACKED_SINGLETON_KV=False,
        SCALED_FP8_KV=False,
        COMPRESSED_FP8=compressed_fp8,
        SCALE=float(scale),
        num_warps=hard_c4.HARD_720_ROUTE_NUM_WARPS,
        num_stages=hard_c4.HARD_720_ROUTE_NUM_STAGES,
    )
    if events is not None:
        events[2].record()
        events[2].synchronize()
        stage1_ms = events[0].elapsed_time(events[1])
        timings["stage1_merge_ms"] = stage1_ms
        timings["correction_ms"] = 0.0
        timings["merge_ms"] = stage1_ms
        timings["route_ms"] = events[1].elapsed_time(events[2])
    return output


def hard_c4_720p_singleton_leverage_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    top_r=SINGLETON_TOP_R,
    exact_intra_frame=False,
    query_outer_offset=0,
    timings=None,
    selector="routing_leverage",
    block_j=SINGLETON_BLOCK_J,
):
    """Run C4 with a selected routing-leverage singleton budget."""

    return _hard_c4_720p_singleton_leverage_attention_triton_blocks(
        q,
        k,
        v,
        top_r=top_r,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        exact_intra_frame=exact_intra_frame,
        query_outer_offset=query_outer_offset,
        timings=timings,
        selector=selector,
        block_j=block_j,
    )


def hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    top_r=4,
    query_outer_offset=0,
    timings=None,
    selector="routing_leverage",
    block_j=SINGLETON_BLOCK_J,
    compressed_fp8_kv=False,
    scaled_fp8_kv=False,
    mean_centered_routing=False,
    gaussian_partition_routing=False,
    reuse_partition_exp=False,
    direct_bulk_normalization=False,
    fp8_score_dot=False,
    fp8_merge_k_dot=False,
    fp8_merge_v_dot=False,
    query_frame_group=21,
    split_merge=False,
    materialize_singleton=False,
    query_column_major=False,
    query_column_secondary_frame_stride=1,
    query_column_primary_row_mod4_omit2=False,
    packed_singleton_kv=False,
    seed_rows_native_layout=False,
    merge_frame_group=1,
    merge_col_group=1,
    merge_loop_stages=1,
    route_loop_stages=None,
    merge_disable_licm=None,
    split_kv_accum=False,
):
    """Run singleton-residual C4 from one to four shared seed-row tables."""

    selector = str(selector).lower()
    if selector in {
        "mass",
        "mass_bias",
        "query_column",
        "query_column_deferred",
    }:
        return _hard_c4_720p_temporal_shared_global_mass_attention_triton_blocks(
            q,
            k,
            v,
            seed_rows,
            top_r=top_r,
            scale=scale,
            output=output,
            head_start=head_start,
            head_count=head_count,
            query_outer_offset=query_outer_offset,
            timings=timings,
            block_j=block_j,
            bias_only=selector == "mass_bias",
            query_column_selector=selector.startswith("query_column"),
            defer_correction=selector == "query_column_deferred",
            compressed_fp8_kv=compressed_fp8_kv,
            scaled_fp8_kv=scaled_fp8_kv,
            mean_centered_routing=mean_centered_routing,
            gaussian_partition_routing=gaussian_partition_routing,
            reuse_partition_exp=reuse_partition_exp,
            direct_bulk_normalization=direct_bulk_normalization,
            fp8_score_dot=fp8_score_dot,
            fp8_merge_k_dot=fp8_merge_k_dot,
            fp8_merge_v_dot=fp8_merge_v_dot,
            query_frame_group=query_frame_group,
            split_merge=split_merge,
            materialize_singleton=materialize_singleton,
            query_column_major=query_column_major,
            query_column_secondary_frame_stride=(
                query_column_secondary_frame_stride
            ),
            query_column_primary_row_mod4_omit2=(
                query_column_primary_row_mod4_omit2
            ),
            packed_singleton_kv=packed_singleton_kv,
            seed_rows_native_layout=seed_rows_native_layout,
            merge_frame_group=merge_frame_group,
            merge_col_group=merge_col_group,
            merge_loop_stages=merge_loop_stages,
            route_loop_stages=route_loop_stages,
            merge_disable_licm=merge_disable_licm,
            split_kv_accum=split_kv_accum,
        )
    if selector == "cluster_mass":
        if (
            compressed_fp8_kv
            or scaled_fp8_kv
            or mean_centered_routing
            or gaussian_partition_routing
            or reuse_partition_exp
            or fp8_score_dot
            or fp8_merge_k_dot
            or fp8_merge_v_dot
            or int(query_frame_group) != 21
            or split_merge
            or materialize_singleton
        ):
            raise ValueError(
                "FP8 and mean-routing options are not supported by cluster_mass"
            )
        if int(top_r) != SINGLETON_C:
            raise ValueError("cluster_mass requires one singleton per C4 cluster.")
        return _hard_c4_720p_temporal_shared_cluster_mass_staged_attention_triton_blocks(
            q,
            k,
            v,
            seed_rows,
            scale=scale,
            output=output,
            head_start=head_start,
            head_count=head_count,
            query_outer_offset=query_outer_offset,
            timings=timings,
        )
    if compressed_fp8_kv or scaled_fp8_kv:
        raise ValueError(
            "FP8 K/V currently requires the query_column selector"
        )
    if mean_centered_routing:
        raise ValueError(
            "mean-centered routing currently requires a global-mass selector"
        )
    if gaussian_partition_routing:
        raise ValueError(
            "Gaussian-partition routing currently requires a global-mass selector"
        )
    if reuse_partition_exp:
        raise ValueError(
            "partition-exp reuse currently requires a global-mass selector"
        )
    if fp8_score_dot:
        raise ValueError("FP8 score dot currently requires a global-mass selector")
    if fp8_merge_k_dot or fp8_merge_v_dot:
        raise ValueError("FP8 merge dot currently requires a global-mass selector")
    if int(query_frame_group) != 21:
        raise ValueError("grouped query frames require a global-mass selector")
    if split_merge:
        raise ValueError("split merge requires a global-mass selector")
    if materialize_singleton:
        raise ValueError("materialized singleton requires a global-mass selector")
    if query_column_major:
        raise ValueError("query-column-major routing requires a global-mass selector")
    if packed_singleton_kv:
        raise ValueError("packed singleton K/V requires a global-mass selector")
    return _hard_c4_720p_singleton_leverage_attention_triton_blocks(
        q,
        k,
        v,
        top_r=top_r,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        temporal_shared_seed_rows=seed_rows,
        query_outer_offset=query_outer_offset,
        timings=timings,
        selector=selector,
        block_j=block_j,
    )


def hard_c4_720p_singleton_leverage_top6_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run all-layer C4 with six routing-leverage singleton children."""

    return _hard_c4_720p_singleton_leverage_attention_triton_blocks(
        q,
        k,
        v,
        top_r=SINGLETON_TOP_R6,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
    )


def hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    query_outer_offset=0,
):
    """Use exact same-frame K/V and C4 plus three atoms for remote frames."""

    return _hard_c4_720p_singleton_leverage_attention_triton_blocks(
        q,
        k,
        v,
        top_r=SINGLETON_TOP_R3,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        exact_intra_frame=True,
        query_outer_offset=query_outer_offset,
    )


__all__ = [
    "hard_c4_720p_singleton_leverage_attention_triton_blocks",
    "hard_c4_720p_temporal_shared_seed_rows_singleton_leverage_attention_triton_blocks",
    "hard_c4_720p_singleton_leverage_top3_exact_intra_frame_attention_triton_blocks",
    "hard_c4_720p_singleton_leverage_top6_attention_triton_blocks",
]
