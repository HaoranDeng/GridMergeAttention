"""Generated C4 scheduling and residual-division experiments."""
import triton
import triton.language as tl

@triton.jit
def _divide_residual(value, residual, MODE: tl.constexpr):
    if MODE == 0:
        return value / residual[:, None]
    inverse = 1.0 / residual
    quotient = value * inverse[:, None]
    if MODE == 2:
        error = tl.fma(-quotient, residual[:, None], value)
        quotient = tl.fma(error, inverse[:, None], quotient)
    return quotient

@triton.jit
def merge_kernel(
    q_ptr,
    response_seed_row_ptr,
    temporal_row_map_ptr,
    k_ptr,
    v_ptr,
    merged_k_ptr,
    merged_v_ptr,
    merged_k_scale_ptr,
    merged_v_scale_ptr,
    bias_ptr,
    selected_col_ptr,
    selected_weight_ptr,
    Q_OUTER: tl.constexpr,
    ROUTER_FRAMES: tl.constexpr,
    K_OUTER: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    ACTIVE_HEADS: tl.constexpr,
    HEAD_START: tl.constexpr,
    D_HEAD: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    C: tl.constexpr,
    OUT_C: tl.constexpr,
    BLOCK_J: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    MERGE_FRAME_GROUP: tl.constexpr,
    MERGE_COL_GROUP: tl.constexpr,
    DISABLE_LICM: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_ENTROPY_BIAS: tl.constexpr,
    ROUTE_BIAS_SCALE: tl.constexpr,
    LOGIT_VARIANCE_BIAS_SCALE: tl.constexpr,
    STORE_CLUSTER_MASS: tl.constexpr,
    FUSE_CLUSTER_MASS: tl.constexpr,
    CORRECT_CLUSTER_VECTORS: tl.constexpr,
    GLOBAL_MASS_TOP_R: tl.constexpr,
    GLOBAL_MASS_EXTRA_BLOCK: tl.constexpr,
    MATERIALIZE_GLOBAL_MASS: tl.constexpr,
    GLOBAL_MASS_BIAS_ONLY: tl.constexpr,
    GLOBAL_QUERY_COLUMN_SELECTOR: tl.constexpr,
    GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE: tl.constexpr,
    GLOBAL_QUERY_COLUMN_PRIMARY_ROW_MOD4_OMIT2: tl.constexpr,
    GLOBAL_MASS_DEFER_CORRECTION: tl.constexpr,
    MEAN_CENTERED_ROUTING: tl.constexpr,
    GAUSSIAN_PARTITION_ROUTING: tl.constexpr,
    REUSE_PARTITION_EXP: tl.constexpr,
    DIRECT_BULK_NORMALIZATION: tl.constexpr,
    FP8_SCORE_DOT: tl.constexpr,
    FP8_MERGE_K_DOT: tl.constexpr,
    FP8_MERGE_V_DOT: tl.constexpr,
    SCALED_FP8_KV: tl.constexpr,
    SPLIT_KV_ACCUM: tl.constexpr,
    STORE_PACKED_SINGLETON_KV: tl.constexpr,
    SEED_ROWS_NATIVE_LAYOUT: tl.constexpr,
    USE_TEMPORAL_ROW_MAP: tl.constexpr,
    SCALE: tl.constexpr,
    MEAN_SEED_ZERO: tl.constexpr = False,
    mean_seed_ptr=None,
    PACK_Q: tl.constexpr = False,
    COL_SPLITS: tl.constexpr = 1,
    FRAME_SPLITS: tl.constexpr = 1,
    RESIDUAL_DIV_MODE: tl.constexpr = 0,
):
    """Merge with one route-contiguous seed table shared by all Q frames."""

    tl.static_assert(ROUTER_FRAMES >= 1)
    tl.static_assert(ROUTER_FRAMES <= 4)
    tl.static_assert(
        not (MEAN_CENTERED_ROUTING and GAUSSIAN_PARTITION_ROUTING)
    )
    tl.static_assert(
        not (
            REUSE_PARTITION_EXP
            and (MEAN_CENTERED_ROUTING or GAUSSIAN_PARTITION_ROUTING)
        )
    )
    tl.static_assert(
        not (FP8_MERGE_K_DOT or FP8_MERGE_V_DOT) or SPLIT_KV_ACCUM
    )
    tl.static_assert(Q_OUTER % MERGE_FRAME_GROUP == 0)
    tl.static_assert(COLS % MERGE_COL_GROUP == 0)
    if MERGE_FRAME_GROUP > 1:
        tl.static_assert(ROUTER_FRAMES == 1)
    if SCALED_FP8_KV:
        tl.static_assert(OUT_C == C)
        tl.static_assert(not FUSE_CLUSTER_MASS)
        tl.static_assert(not MATERIALIZE_GLOBAL_MASS)
    if STORE_PACKED_SINGLETON_KV:
        tl.static_assert(GLOBAL_QUERY_COLUMN_SELECTOR)
        tl.static_assert(GLOBAL_MASS_TOP_R == 1)
        tl.static_assert(not GLOBAL_MASS_DEFER_CORRECTION)
        tl.static_assert(not STORE_CLUSTER_MASS)
        tl.static_assert(not FUSE_CLUSTER_MASS)
        tl.static_assert(not MATERIALIZE_GLOBAL_MASS)
        tl.static_assert(not SCALED_FP8_KV)
    if GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE > 1:
        tl.static_assert(GLOBAL_QUERY_COLUMN_SELECTOR)
        tl.static_assert(GLOBAL_MASS_TOP_R == 2)
        tl.static_assert(not MATERIALIZE_GLOBAL_MASS)
    if GLOBAL_QUERY_COLUMN_PRIMARY_ROW_MOD4_OMIT2:
        tl.static_assert(GLOBAL_QUERY_COLUMN_SELECTOR)
        tl.static_assert(GLOBAL_MASS_TOP_R == 1)
        tl.static_assert(GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE == 1)
        tl.static_assert(not MATERIALIZE_GLOBAL_MASS)

    block_id = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % ACTIVE_HEADS
    global_head = HEAD_START + head
    batch = batch_head // ACTIVE_HEADS
    batch_i64 = batch.to(tl.int64)
    key_outer = block_id % K_OUTER
    key_row = block_id // K_OUTER

    flat_rows = tl.arange(0, BLOCK_J * C)
    seed_ids = flat_rows % C
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
    if STORE_PACKED_SINGLETON_KV:
        # Reuse the raw K/V already resident in registers. The packed
        # layout [B, query_col, active_head, K, row, D] lets all 21 query
        # frames read one route-contiguous singleton stream.
        packed_main = (
            (((((batch_i64 * COLS + main_cols[:, None]) * ACTIVE_HEADS + head)
                * K_OUTER + key_outer) * ROWS + key_row) * D_HEAD)
            + dims[None, :]
        )
        packed_tail = (
            (((((batch_i64 * COLS + tail_cols[:, None]) * ACTIVE_HEADS + head)
                * K_OUTER + key_outer) * ROWS + key_row) * D_HEAD)
            + dims[None, :]
        )
        tl.store(
            selected_col_ptr + packed_main,
            key_main,
            eviction_policy="evict_first",
        )
        tl.store(
            selected_col_ptr + packed_tail,
            key_tail,
            eviction_policy="evict_first",
        )
        tl.store(
            selected_weight_ptr + packed_main,
            value_main,
            eviction_policy="evict_first",
        )
        tl.store(
            selected_weight_ptr + packed_tail,
            value_tail,
            eviction_policy="evict_first",
        )
    if SPLIT_KV_ACCUM:
        tl.static_assert(GLOBAL_MASS_TOP_R > 0)
        tl.static_assert(GLOBAL_MASS_TOP_R <= 4)
        tl.static_assert(GLOBAL_QUERY_COLUMN_SELECTOR)
        tl.static_assert(not GLOBAL_MASS_DEFER_CORRECTION)
        tl.static_assert(not GLOBAL_MASS_BIAS_ONLY)
        tl.static_assert(not MATERIALIZE_GLOBAL_MASS)
        tl.static_assert(not FUSE_CLUSTER_MASS)
        # Packed singleton storage is independent of the split bulk K/V dots.
        # This combination is validated by the opt-in C8 entry point.
        tl.static_assert(not STORE_PACKED_SINGLETON_KV or C == 8)
    if DIRECT_BULK_NORMALIZATION:
        tl.static_assert(GLOBAL_MASS_TOP_R == 1)
        tl.static_assert(GLOBAL_QUERY_COLUMN_SELECTOR)
        tl.static_assert(not GLOBAL_MASS_DEFER_CORRECTION)
        tl.static_assert(not GLOBAL_MASS_BIAS_ONLY)
        tl.static_assert(not MATERIALIZE_GLOBAL_MASS)
        tl.static_assert(not FUSE_CLUSTER_MASS)
        tl.static_assert(not SCALED_FP8_KV)
        tl.static_assert(not STORE_PACKED_SINGLETON_KV)
    else:
        key_value_main = tl.interleave(key_main, value_main)
        key_value_tail = tl.interleave(key_tail, value_tail)

    tl.static_assert((COLS // BLOCK_J) % COL_SPLITS == 0)
    tl.static_assert(Q_OUTER % FRAME_SPLITS == 0)
    split_col = tl.program_id(2) % COL_SPLITS
    split_frame = tl.program_id(2) // COL_SPLITS
    for query_col_start in tl.range(split_col * (COLS // COL_SPLITS), (split_col + 1) * (COLS // COL_SPLITS), BLOCK_J):
        query_cols = query_col_start + flat_rows // C
        merge_query_cols = (
            (query_cols // MERGE_COL_GROUP) * MERGE_COL_GROUP
            + MERGE_COL_GROUP // 2
        )
        response_seed_base = batch_i64 * ROUTER_FRAMES
        if SEED_ROWS_NATIVE_LAYOUT:
            response_seed_base_0 = response_seed_base * K_OUTER + key_outer
            response_seed_base_0 = response_seed_base_0 * ROWS + key_row
            response_seed_base_0 = response_seed_base_0 * COLS + merge_query_cols
            response_seed_base_0 = (
                response_seed_base_0 * TOTAL_HEADS + global_head
            )
            response_seed_base_0 = response_seed_base_0 * C + seed_ids
        else:
            response_seed_base_0 = response_seed_base * COLS + merge_query_cols
            response_seed_base_0 = (
                response_seed_base_0 * ACTIVE_HEADS + head
            )
            response_seed_base_0 = response_seed_base_0 * K_OUTER + key_outer
            response_seed_base_0 = response_seed_base_0 * ROWS + key_row
            response_seed_base_0 = response_seed_base_0 * C + seed_ids
        shared_seed_rows_0 = tl.load(
            response_seed_row_ptr + response_seed_base_0
        ).to(tl.int32)
        if ROUTER_FRAMES > 1:
            if SEED_ROWS_NATIVE_LAYOUT:
                response_seed_base_1 = (
                    (response_seed_base + 1) * K_OUTER + key_outer
                )
                response_seed_base_1 = response_seed_base_1 * ROWS + key_row
                response_seed_base_1 = response_seed_base_1 * COLS + merge_query_cols
                response_seed_base_1 = (
                    response_seed_base_1 * TOTAL_HEADS + global_head
                )
                response_seed_base_1 = response_seed_base_1 * C + seed_ids
            else:
                response_seed_base_1 = (
                    (response_seed_base + 1) * COLS + merge_query_cols
                )
                response_seed_base_1 = (
                    response_seed_base_1 * ACTIVE_HEADS + head
                )
                response_seed_base_1 = response_seed_base_1 * K_OUTER + key_outer
                response_seed_base_1 = response_seed_base_1 * ROWS + key_row
                response_seed_base_1 = response_seed_base_1 * C + seed_ids
            shared_seed_rows_1 = tl.load(
                response_seed_row_ptr + response_seed_base_1
            ).to(tl.int32)
        if ROUTER_FRAMES > 2:
            if SEED_ROWS_NATIVE_LAYOUT:
                response_seed_base_2 = (
                    (response_seed_base + 2) * K_OUTER + key_outer
                )
                response_seed_base_2 = response_seed_base_2 * ROWS + key_row
                response_seed_base_2 = response_seed_base_2 * COLS + merge_query_cols
                response_seed_base_2 = (
                    response_seed_base_2 * TOTAL_HEADS + global_head
                )
                response_seed_base_2 = response_seed_base_2 * C + seed_ids
            else:
                response_seed_base_2 = (
                    (response_seed_base + 2) * COLS + merge_query_cols
                )
                response_seed_base_2 = (
                    response_seed_base_2 * ACTIVE_HEADS + head
                )
                response_seed_base_2 = response_seed_base_2 * K_OUTER + key_outer
                response_seed_base_2 = response_seed_base_2 * ROWS + key_row
                response_seed_base_2 = response_seed_base_2 * C + seed_ids
            shared_seed_rows_2 = tl.load(
                response_seed_row_ptr + response_seed_base_2
            ).to(tl.int32)
        if ROUTER_FRAMES > 3:
            if SEED_ROWS_NATIVE_LAYOUT:
                response_seed_base_3 = (
                    (response_seed_base + 3) * K_OUTER + key_outer
                )
                response_seed_base_3 = response_seed_base_3 * ROWS + key_row
                response_seed_base_3 = response_seed_base_3 * COLS + merge_query_cols
                response_seed_base_3 = (
                    response_seed_base_3 * TOTAL_HEADS + global_head
                )
                response_seed_base_3 = response_seed_base_3 * C + seed_ids
            else:
                response_seed_base_3 = (
                    (response_seed_base + 3) * COLS + merge_query_cols
                )
                response_seed_base_3 = (
                    response_seed_base_3 * ACTIVE_HEADS + head
                )
                response_seed_base_3 = response_seed_base_3 * K_OUTER + key_outer
                response_seed_base_3 = response_seed_base_3 * ROWS + key_row
                response_seed_base_3 = response_seed_base_3 * C + seed_ids
            shared_seed_rows_3 = tl.load(
                response_seed_row_ptr + response_seed_base_3
            ).to(tl.int32)

        for query_outer in tl.range(
            split_frame * (Q_OUTER // FRAME_SPLITS),
            (split_frame + 1) * (Q_OUTER // FRAME_SPLITS),
            num_stages=LOOP_STAGES,
            disable_licm=DISABLE_LICM,
        ):
            router_slot = (query_outer * ROUTER_FRAMES) // Q_OUTER
            if ROUTER_FRAMES == 4:
                # Nearest-router map for temporal anchors [0, 7, 13, 20].
                router_slot = tl.where(
                    query_outer <= 3,
                    0,
                    tl.where(query_outer <= 10, 1, tl.where(query_outer <= 16, 2, 3)),
                )
            active_seed_rows = shared_seed_rows_0
            if ROUTER_FRAMES > 1:
                active_seed_rows = tl.where(
                    router_slot == 1,
                    shared_seed_rows_1,
                    active_seed_rows,
                )
            if ROUTER_FRAMES > 2:
                active_seed_rows = tl.where(
                    router_slot == 2,
                    shared_seed_rows_2,
                    active_seed_rows,
                )
            if ROUTER_FRAMES > 3:
                active_seed_rows = tl.where(
                    router_slot == 3,
                    shared_seed_rows_3,
                    active_seed_rows,
                )
            if USE_TEMPORAL_ROW_MAP:
                row_map_offset = (
                    ((((batch_i64 * Q_OUTER + query_outer) * COLS
                       + merge_query_cols)
                      * TOTAL_HEADS
                      + global_head)
                     * ROWS)
                    + active_seed_rows
                )
                tracked_seed_rows = tl.load(
                    temporal_row_map_ptr + row_map_offset
                ).to(tl.int32)
                # Preserve the structural anchor and local-neighbor slots.
                # Only the two dynamic basis seeds follow Q-space motion.
                active_seed_rows = tl.where(
                    seed_ids >= 2,
                    tracked_seed_rows,
                    active_seed_rows,
                )
            merge_query_outer = (
                (query_outer // MERGE_FRAME_GROUP) * MERGE_FRAME_GROUP
                + MERGE_FRAME_GROUP // 2
            )
            q_base = (
                (batch * Q_OUTER + merge_query_outer)
                * ROWS
                * COLS
                * TOTAL_HEADS
                * D_HEAD
            )
            seed_query_offsets = (
                q_base
                + (
                    (active_seed_rows[:, None] * COLS + merge_query_cols[:, None])
                    * TOTAL_HEADS
                    + global_head
                )
                * D_HEAD
                + dims[None, :]
            )
            if PACK_Q:
                seed_query_offsets = (((((batch * TOTAL_HEADS + global_head) * Q_OUTER + merge_query_outer)
                    * ROWS + active_seed_rows[:, None]) * COLS + merge_query_cols[:, None]) * D_HEAD + dims[None, :])
            seed_query = tl.load(
                q_ptr + seed_query_offsets,
                eviction_policy="evict_last",
            )
            if MEAN_SEED_ZERO:
                # Experimental analytic seed: current-frame row mean,
                # with no norm rescaling. Other seeds and route Q stay real.
                mean_offset = (
                    ((batch * Q_OUTER + merge_query_outer) * COLS
                     + merge_query_cols[:, None]) * TOTAL_HEADS + global_head
                ) * D_HEAD + dims[None, :]
                mean_query = tl.load(mean_seed_ptr + mean_offset)
                seed_query = tl.where(seed_ids[:, None] == 0, mean_query, seed_query)
            score_query = seed_query
            score_key_main = key_main
            score_key_tail = key_tail
            if FP8_SCORE_DOT:
                score_query = score_query.to(tl.float8e4nv)
                score_key_main = score_key_main.to(tl.float8e4nv)
                score_key_tail = score_key_tail.to(tl.float8e4nv)
            scores_main = tl.dot(
                score_query,
                tl.trans(score_key_main),
                input_precision=INPUT_PRECISION,
            )
            scores_tail = tl.dot(
                score_query,
                tl.trans(score_key_tail),
                input_precision=INPUT_PRECISION,
            )
            scores_main *= SCALE * 1.4426950408889634
            scores_tail *= SCALE * 1.4426950408889634

            if MEAN_CENTERED_ROUTING:
                log_norm = (
                    tl.sum(scores_main, axis=1)
                    + tl.sum(scores_tail, axis=1)
                ) / COLS
            elif GAUSSIAN_PARTITION_ROUTING:
                score_mean = (
                    tl.sum(scores_main, axis=1)
                    + tl.sum(scores_tail, axis=1)
                ) / COLS
                score_second_moment = (
                    tl.sum(scores_main * scores_main, axis=1)
                    + tl.sum(scores_tail * scores_tail, axis=1)
                ) / COLS
                score_variance = tl.maximum(
                    score_second_moment - score_mean * score_mean,
                    0.0,
                )
                # Scores are in log2 units. The shared log2(COLS) term
                # cancels when ownership is compared across seeds.
                log_norm = (
                    score_mean
                    + 0.34657359027997264 * score_variance
                )
            else:
                row_max = tl.maximum(
                    tl.max(scores_main, axis=1), tl.max(scores_tail, axis=1)
                )
                partition_main = tl.exp2(
                    scores_main - row_max[:, None]
                )
                partition_tail = tl.exp2(
                    scores_tail - row_max[:, None]
                )
                row_sum = tl.sum(partition_main, axis=1)
                row_sum += tl.sum(partition_tail, axis=1)
                log_norm = row_max + tl.log2(row_sum)
            log_probs_main = tl.reshape(
                scores_main - log_norm[:, None],
                (BLOCK_J, C, 64),
            )
            log_probs_tail = tl.reshape(
                scores_tail - log_norm[:, None],
                (BLOCK_J, C, 16),
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
            if REUSE_PARTITION_EXP:
                normalized_partition_main = tl.reshape(
                    partition_main / row_sum[:, None],
                    (BLOCK_J, C, 64),
                )
                normalized_partition_tail = tl.reshape(
                    partition_tail / row_sum[:, None],
                    (BLOCK_J, C, 16),
                )
                probs_main = tl.where(
                    route_main[:, None, :] == cluster_ids,
                    normalized_partition_main,
                    0.0,
                )
                probs_tail = tl.where(
                    route_tail[:, None, :] == cluster_ids,
                    normalized_partition_tail,
                    0.0,
                )
            else:
                probs_main = tl.where(
                    route_main[:, None, :] == cluster_ids,
                    tl.exp2(winner_main)[:, None, :],
                    0.0,
                )
                probs_tail = tl.where(
                    route_tail[:, None, :] == cluster_ids,
                    tl.exp2(winner_tail)[:, None, :],
                    0.0,
                )
            prob_sum = tl.sum(probs_main, axis=2) + tl.sum(probs_tail, axis=2)
            has_key = prob_sum > 0.0
            safe_sum = tl.where(has_key, prob_sum, 1.0)
            if DIRECT_BULK_NORMALIZATION:
                # The query-column singleton is removed below. Keep the
                # winner probabilities unnormalized until that removal so
                # the bulk distribution needs only one normalization.
                weights_main = probs_main
                weights_tail = probs_tail
            else:
                inverse_sum = 1.0 / safe_sum
                weights_main = probs_main * inverse_sum[:, :, None]
                weights_tail = probs_tail * inverse_sum[:, :, None]
            if USE_ENTROPY_BIAS:
                expected = tl.sum(
                    probs_main
                    * tl.where(
                        probs_main > 0.0,
                        winner_main[:, None, :],
                        0.0,
                    ),
                    axis=2,
                )
                expected += tl.sum(
                    probs_tail
                    * tl.where(
                        probs_tail > 0.0,
                        winner_tail[:, None, :],
                        0.0,
                    ),
                    axis=2,
                )
                if DIRECT_BULK_NORMALIZATION:
                    # Recomputed directly over the residual bulk after
                    # singleton removal below.
                    route_bias = tl.zeros((BLOCK_J, C), tl.float32)
                else:
                    expected_mean = expected * inverse_sum
                    route_bias = (
                        expected_mean - tl.log2(safe_sum)
                    ) * 0.6931471805599453
                if (
                    LOGIT_VARIANCE_BIAS_SCALE != 0.0
                    and not DIRECT_BULK_NORMALIZATION
                ):
                    second_moment = tl.sum(
                        probs_main
                        * tl.where(
                            probs_main > 0.0,
                            winner_main[:, None, :]
                            * winner_main[:, None, :],
                            0.0,
                        ),
                        axis=2,
                    )
                    second_moment += tl.sum(
                        probs_tail
                        * tl.where(
                            probs_tail > 0.0,
                            winner_tail[:, None, :]
                            * winner_tail[:, None, :],
                            0.0,
                        ),
                        axis=2,
                    )
                    logit_variance = tl.maximum(
                        second_moment * inverse_sum
                        - expected_mean * expected_mean,
                        0.0,
                    )
                    route_bias -= (
                        LOGIT_VARIANCE_BIAS_SCALE
                        * logit_variance
                        * 0.4804530139182014
                    )
            else:
                route_bias = tl.zeros((BLOCK_J, C), tl.float32)
            route_bias *= ROUTE_BIAS_SCALE
            route_bias = tl.where(has_key, route_bias, float("inf"))

            if GLOBAL_MASS_TOP_R > 0:
                tl.static_assert(GLOBAL_MASS_TOP_R <= 4)
                tl.static_assert(GLOBAL_MASS_TOP_R <= GLOBAL_MASS_EXTRA_BLOCK)
                global_extra_lanes = tl.arange(0, GLOBAL_MASS_EXTRA_BLOCK)
                if GLOBAL_QUERY_COLUMN_SELECTOR:
                    global_query_cols = (
                        query_col_start + tl.arange(0, BLOCK_J)
                    )
                    selected_main_global = tl.full(
                        (BLOCK_J, 64), False, tl.int1
                    )
                    selected_tail_global = tl.full(
                        (BLOCK_J, 16), False, tl.int1
                    )
                    selected_col_global = tl.zeros(
                        (BLOCK_J, GLOBAL_MASS_EXTRA_BLOCK), tl.int32
                    )
                    # Use a deterministic symmetric stencil around each
                    # query column. Every rank denotes a distinct raw K/V
                    # singleton and is peeled from the bulk clusters.
                    for rank in tl.static_range(0, GLOBAL_MASS_TOP_R):
                        if rank == 0:
                            rank_col = global_query_cols
                        elif rank == 1:
                            rank_col = (global_query_cols + COLS - 1) % COLS
                        elif rank == 2:
                            rank_col = (global_query_cols + 1) % COLS
                        else:
                            rank_col = (global_query_cols + COLS - 2) % COLS
                        rank_selected_main = (
                            tl.arange(0, 64)[None, :]
                            == rank_col[:, None]
                        )
                        rank_selected_tail = (
                            64 + tl.arange(0, 16)[None, :]
                            == rank_col[:, None]
                        )
                        if GLOBAL_QUERY_COLUMN_PRIMARY_ROW_MOD4_OMIT2:
                            primary_active = key_row % 4 != 2
                            rank_selected_main = (
                                rank_selected_main & primary_active
                            )
                            rank_selected_tail = (
                                rank_selected_tail & primary_active
                            )
                        if (
                            rank == 1
                            and GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE > 1
                        ):
                            secondary_active = (
                                key_outer
                                % GLOBAL_QUERY_COLUMN_SECONDARY_FRAME_STRIDE
                                == 0
                            )
                            rank_selected_main = (
                                rank_selected_main & secondary_active
                            )
                            rank_selected_tail = (
                                rank_selected_tail & secondary_active
                            )
                        selected_main_global = (
                            selected_main_global | rank_selected_main
                        )
                        selected_tail_global = (
                            selected_tail_global | rank_selected_tail
                        )
                        selected_col_global = tl.where(
                            global_extra_lanes[None, :] == rank,
                            rank_col[:, None],
                            selected_col_global,
                        )
                else:
                    key_mass_main = tl.max(weights_main, axis=1)
                    key_mass_tail = tl.max(weights_tail, axis=1)
                    selected_main_global = tl.full(
                        (BLOCK_J, 64), False, tl.int1
                    )
                    selected_tail_global = tl.full(
                        (BLOCK_J, 16), False, tl.int1
                    )
                    selected_col_global = tl.zeros(
                        (BLOCK_J, GLOBAL_MASS_EXTRA_BLOCK), tl.int32
                    )
                    for rank in tl.static_range(0, GLOBAL_MASS_TOP_R):
                        main_rank_value, main_rank_index = tl.max(
                            tl.where(
                                ~selected_main_global,
                                key_mass_main,
                                -float("inf"),
                            ),
                            axis=1,
                            return_indices=True,
                            return_indices_tie_break_left=True,
                        )
                        tail_rank_value, tail_rank_index = tl.max(
                            tl.where(
                                ~selected_tail_global,
                                key_mass_tail,
                                -float("inf"),
                            ),
                            axis=1,
                            return_indices=True,
                            return_indices_tie_break_left=True,
                        )
                        rank_col = tl.where(
                            tail_rank_value > main_rank_value,
                            64 + tail_rank_index,
                            main_rank_index,
                        )
                        selected_main_global = selected_main_global | (
                            tl.arange(0, 64)[None, :]
                            == rank_col[:, None]
                        )
                        selected_tail_global = selected_tail_global | (
                            64 + tl.arange(0, 16)[None, :]
                            == rank_col[:, None]
                        )
                        selected_col_global = tl.where(
                            global_extra_lanes[None, :] == rank,
                            rank_col[:, None],
                            selected_col_global,
                        )

                if DIRECT_BULK_NORMALIZATION:
                    selected_probability = tl.sum(
                        tl.where(
                            selected_main_global[:, None, :],
                            probs_main,
                            0.0,
                        ),
                        axis=2,
                    )
                    selected_probability += tl.sum(
                        tl.where(
                            selected_tail_global[:, None, :],
                            probs_tail,
                            0.0,
                        ),
                        axis=2,
                    )
                    selected_cluster_weight = (
                        selected_probability / safe_sum
                    )
                else:
                    selected_cluster_weight = tl.sum(
                        tl.where(
                            selected_main_global[:, None, :],
                            weights_main,
                            0.0,
                        ),
                        axis=2,
                    )
                    selected_cluster_weight += tl.sum(
                        tl.where(
                            selected_tail_global[:, None, :],
                            weights_tail,
                            0.0,
                        ),
                        axis=2,
                    )
                residual_mass = 1.0 - selected_cluster_weight
                bulk_has_key = has_key & (residual_mass > 1.0e-6)
                safe_residual_mass = tl.where(
                    bulk_has_key, residual_mass, 1.0
                )
                if GLOBAL_MASS_DEFER_CORRECTION:
                    tl.static_assert(GLOBAL_QUERY_COLUMN_SELECTOR)
                    tl.static_assert(GLOBAL_MASS_TOP_R == 1)
                    selected_weight_global, selected_owner_global = tl.max(
                        selected_cluster_weight,
                        axis=1,
                        return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    metadata_query_cols = (
                        query_col_start + tl.arange(0, BLOCK_J)
                    )
                    metadata_base = (
                        (((((batch_i64 * Q_OUTER + query_outer) * COLS
                            + metadata_query_cols)
                           * ACTIVE_HEADS
                           + head)
                          * K_OUTER
                          + key_outer)
                         * ROWS
                         + key_row)
                    )
                    packed_metadata = (
                        selected_owner_global.to(tl.float32) * 2.0
                        + selected_weight_global
                    )
                    tl.store(
                        selected_weight_ptr + metadata_base,
                        packed_metadata,
                        eviction_policy="evict_first",
                    )
                else:
                    if DIRECT_BULK_NORMALIZATION:
                        bulk_sum = prob_sum - selected_probability
                        safe_bulk_sum = tl.where(
                            bulk_has_key, bulk_sum, 1.0
                        )
                        if USE_ENTROPY_BIAS:
                            selected_expected = tl.sum(
                                tl.where(
                                    selected_main_global[:, None, :],
                                    probs_main
                                    * winner_main[:, None, :],
                                    0.0,
                                ),
                                axis=2,
                            )
                            selected_expected += tl.sum(
                                tl.where(
                                    selected_tail_global[:, None, :],
                                    probs_tail
                                    * winner_tail[:, None, :],
                                    0.0,
                                ),
                                axis=2,
                            )
                            bulk_expected = expected - selected_expected
                            route_bias = (
                                bulk_expected / safe_bulk_sum
                                - tl.log2(safe_bulk_sum)
                            ) * 0.6931471805599453
                        else:
                            route_bias = tl.zeros(
                                (BLOCK_J, C), tl.float32
                            )
                        route_bias = tl.where(
                            bulk_has_key, route_bias, float("inf")
                        )
                        weights_main = tl.where(
                            ~selected_main_global[:, None, :],
                            probs_main / safe_bulk_sum[:, :, None],
                            0.0,
                        )
                        weights_tail = tl.where(
                            ~selected_tail_global[:, None, :],
                            probs_tail / safe_bulk_sum[:, :, None],
                            0.0,
                        )
                    elif USE_ENTROPY_BIAS:
                        selected_w_log_w = (
                            selected_cluster_weight
                            * tl.log2(
                                tl.maximum(
                                    selected_cluster_weight, 1.0e-20
                                )
                            )
                            * 0.6931471805599453
                            * ROUTE_BIAS_SCALE
                        )
                        route_bias = (
                            (route_bias - selected_w_log_w)
                            / safe_residual_mass
                            - tl.log2(safe_residual_mass)
                            * 0.6931471805599453
                            * ROUTE_BIAS_SCALE
                        )
                    else:
                        route_bias = tl.zeros((BLOCK_J, C), tl.float32)
                    if not DIRECT_BULK_NORMALIZATION:
                        route_bias = tl.where(
                            bulk_has_key, route_bias, float("inf")
                        )
                        if not GLOBAL_MASS_BIAS_ONLY:
                            weights_main = tl.where(
                                ~selected_main_global[:, None, :],
                                weights_main,
                                0.0,
                            )
                            weights_tail = tl.where(
                                ~selected_tail_global[:, None, :],
                                weights_tail,
                                0.0,
                            )
                if (
                    not MATERIALIZE_GLOBAL_MASS
                    and not GLOBAL_QUERY_COLUMN_SELECTOR
                ):
                    global_query_cols = (
                        query_col_start + tl.arange(0, BLOCK_J)
                    )
                    global_selected_base = (
                        (((((batch_i64 * Q_OUTER + query_outer) * COLS
                            + global_query_cols[:, None])
                           * ACTIVE_HEADS
                           + head)
                          * K_OUTER
                          + key_outer)
                         * ROWS
                         + key_row)
                        * GLOBAL_MASS_TOP_R
                        + global_extra_lanes[None, :]
                    )
                    tl.store(
                        selected_col_ptr + global_selected_base,
                        selected_col_global,
                        mask=(
                            global_extra_lanes[None, :]
                            < GLOBAL_MASS_TOP_R
                        ),
                        eviction_policy="evict_first",
                    )

            if STORE_CLUSTER_MASS or FUSE_CLUSTER_MASS:
                selected_query_cols = query_col_start + tl.arange(0, BLOCK_J)
                selected_seed = tl.arange(0, C)
                main_weight, main_index = tl.max(
                    weights_main,
                    axis=2,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                tail_weight, tail_index = tl.max(
                    weights_tail,
                    axis=2,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                selected_col = tl.where(
                    tail_weight > main_weight,
                    64 + tail_index,
                    main_index,
                )
                selected_weight = tl.maximum(main_weight, tail_weight)
                selected_base = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + selected_query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + selected_seed[None, :]
                )
                if STORE_CLUSTER_MASS:
                    tl.store(selected_col_ptr + selected_base, selected_col)
                    tl.store(
                        selected_weight_ptr + selected_base,
                        selected_weight,
                    )

            weights_main = tl.reshape(weights_main, (BLOCK_J * C, 64))
            weights_tail = tl.reshape(weights_tail, (BLOCK_J * C, 16))
            merged_base = (
                ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                     + query_cols[:, None])
                    * ACTIVE_HEADS
                    + head)
                   * K_OUTER
                   + key_outer)
                  * ROWS
                  + key_row)
                 * OUT_C)
                + seed_ids[:, None]
            )
            merged_output = merged_base * D_HEAD + dims[None, :]
            if SPLIT_KV_ACCUM:
                residual_scale = tl.reshape(
                    safe_residual_mass, (BLOCK_J * C,)
                )
                if FP8_MERGE_K_DOT or FP8_MERGE_V_DOT:
                    fp8_weights_main = weights_main.to(tl.float8e4nv)
                    # Hopper FP8 dot requires K >= 32; keep the 16-token tail BF16.
                key_weights_main = weights_main.to(key_main.dtype)
                key_weights_tail = weights_tail.to(key_tail.dtype)
                key_operand_main = key_main
                key_operand_tail = key_tail
                if FP8_MERGE_K_DOT:
                    key_weights_main = fp8_weights_main
                    key_operand_main = key_main.to(tl.float8e4nv)
                merged_key = tl.dot(
                    key_weights_main,
                    key_operand_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_key += tl.dot(
                    key_weights_tail,
                    key_operand_tail,
                    input_precision=INPUT_PRECISION,
                )
                if not DIRECT_BULK_NORMALIZATION:
                    merged_key = _divide_residual(merged_key, residual_scale, RESIDUAL_DIV_MODE)
                if SCALED_FP8_KV:
                    key_scale = tl.maximum(tl.max(tl.abs(merged_key), axis=1) / 448.0, 1.0e-8)
                    tl.store(merged_k_scale_ptr + tl.reshape(merged_base, (BLOCK_J * C,)),
                             key_scale, eviction_policy="evict_first")
                    merged_key = (merged_key / key_scale[:, None]).to(tl.float8e4nv)
                tl.store(
                    merged_k_ptr + merged_output,
                    merged_key,
                    eviction_policy="evict_first",
                )
                value_weights_main = weights_main.to(value_main.dtype)
                value_weights_tail = weights_tail.to(value_tail.dtype)
                value_operand_main = value_main
                value_operand_tail = value_tail
                if FP8_MERGE_V_DOT:
                    value_weights_main = fp8_weights_main
                    value_operand_main = value_main.to(tl.float8e4nv)
                merged_value = tl.dot(
                    value_weights_main,
                    value_operand_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_value += tl.dot(
                    value_weights_tail,
                    value_operand_tail,
                    input_precision=INPUT_PRECISION,
                )
                if not DIRECT_BULK_NORMALIZATION:
                    merged_value = _divide_residual(merged_value, residual_scale, RESIDUAL_DIV_MODE)
                if SCALED_FP8_KV:
                    value_scale = tl.maximum(tl.max(tl.abs(merged_value), axis=1) / 448.0, 1.0e-8)
                    tl.store(merged_v_scale_ptr + tl.reshape(merged_base, (BLOCK_J * C,)),
                             value_scale, eviction_policy="evict_first")
                    merged_value = (merged_value / value_scale[:, None]).to(tl.float8e4nv)
                tl.store(
                    merged_v_ptr + merged_output,
                    merged_value,
                    eviction_policy="evict_first",
                )
            else:
                merged_pair = tl.dot(
                    weights_main.to(key_main.dtype),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_pair += tl.dot(
                    weights_tail.to(key_tail.dtype),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                merged_key, merged_value = tl.split(
                    tl.reshape(merged_pair, (BLOCK_J * C, 128, 2))
                )
                if (
                    GLOBAL_MASS_TOP_R > 0
                    and not GLOBAL_MASS_BIAS_ONLY
                    and not GLOBAL_MASS_DEFER_CORRECTION
                ):
                    residual_scale = tl.reshape(
                        safe_residual_mass, (BLOCK_J * C,)
                    )
                    merged_key = _divide_residual(merged_key, residual_scale, RESIDUAL_DIV_MODE)
                    merged_value = _divide_residual(merged_value, residual_scale, RESIDUAL_DIV_MODE)
                if FUSE_CLUSTER_MASS:
                    selected_col_flat = tl.reshape(
                        selected_col, (BLOCK_J * C,)
                    )
                    selected_weight_flat = tl.reshape(
                        selected_weight, (BLOCK_J * C,)
                    )
                    singleton_offsets = (
                        row_base
                        + (
                            selected_col_flat[:, None] * TOTAL_HEADS
                            + global_head
                        )
                        * D_HEAD
                        + dims[None, :]
                    )
                    singleton_key = tl.load(
                        k_ptr + singleton_offsets,
                        eviction_policy="evict_last",
                    )
                    singleton_value = tl.load(
                        v_ptr + singleton_offsets,
                        eviction_policy="evict_last",
                    )
                    residual_weight = 1.0 - selected_weight_flat
                    bulk_valid_flat = tl.reshape(
                        has_key, (BLOCK_J * C,)
                    ) & (residual_weight > 1.0e-6)
                    singleton_valid_flat = tl.reshape(
                        has_key, (BLOCK_J * C,)
                    )
                    safe_residual = tl.where(
                        bulk_valid_flat, residual_weight, 1.0
                    )
                    if CORRECT_CLUSTER_VECTORS:
                        merged_key = (
                            merged_key
                            - selected_weight_flat[:, None] * singleton_key
                        ) / safe_residual[:, None]
                        merged_value = (
                            merged_value
                            - selected_weight_flat[:, None] * singleton_value
                        ) / safe_residual[:, None]
                    selected_weight_matrix = tl.reshape(
                        selected_weight_flat, (BLOCK_J, C)
                    )
                    residual_matrix = 1.0 - selected_weight_matrix
                    safe_residual_matrix = tl.where(
                        has_key & (residual_matrix > 1.0e-6),
                        residual_matrix,
                        1.0,
                    )
                    selected_w_log_w = (
                        selected_weight_matrix
                        * tl.log2(tl.maximum(selected_weight_matrix, 1.0e-20))
                        * 0.6931471805599453
                    )
                    route_bias = (
                        (route_bias - selected_w_log_w)
                        / safe_residual_matrix
                        - tl.log2(safe_residual_matrix)
                        * 0.6931471805599453
                    )
                    route_bias = tl.where(
                        has_key & (residual_matrix > 1.0e-6),
                        route_bias,
                        float("inf"),
                    )
                if SCALED_FP8_KV:
                    key_scale = tl.maximum(
                        tl.max(tl.abs(merged_key), axis=1) / 448.0,
                        1.0e-8,
                    )
                    value_scale = tl.maximum(
                        tl.max(tl.abs(merged_value), axis=1) / 448.0,
                        1.0e-8,
                    )
                    tl.store(
                        merged_k_ptr + merged_output,
                        (merged_key / key_scale[:, None]).to(tl.float8e4nv),
                        eviction_policy="evict_first",
                    )
                    tl.store(
                        merged_v_ptr + merged_output,
                        (merged_value / value_scale[:, None]).to(tl.float8e4nv),
                        eviction_policy="evict_first",
                    )
                    scale_output = tl.reshape(merged_base, (BLOCK_J * C,))
                    tl.store(
                        merged_k_scale_ptr + scale_output,
                        key_scale,
                        eviction_policy="evict_first",
                    )
                    tl.store(
                        merged_v_scale_ptr + scale_output,
                        value_scale,
                        eviction_policy="evict_first",
                    )
                else:
                    tl.store(
                        merged_k_ptr + merged_output,
                        merged_key,
                        eviction_policy="evict_first",
                    )
                    tl.store(
                        merged_v_ptr + merged_output,
                        merged_value,
                        eviction_policy="evict_first",
                    )
            if GLOBAL_MASS_TOP_R > 0 and MATERIALIZE_GLOBAL_MASS:
                global_singleton_offsets = (
                    row_base
                    + (
                        selected_col_global[:, :, None] * TOTAL_HEADS
                        + global_head
                    )
                    * D_HEAD
                    + dims[None, None, :]
                )
                global_singleton_key = tl.load(
                    k_ptr + global_singleton_offsets,
                    eviction_policy="evict_last",
                )
                global_singleton_value = tl.load(
                    v_ptr + global_singleton_offsets,
                    eviction_policy="evict_last",
                )
                global_query_cols = (
                    query_col_start + tl.arange(0, BLOCK_J)
                )
                global_route_base = (
                    (((((batch_i64 * Q_OUTER + query_outer) * COLS
                        + global_query_cols)
                       * ACTIVE_HEADS
                       + head)
                      * K_OUTER
                      + key_outer)
                     * ROWS
                     + key_row)
                    * OUT_C
                )
                global_extra_token = (
                    global_route_base[:, None]
                    + C
                    + global_extra_lanes[None, :]
                )
                global_extra_output = (
                    global_extra_token[:, :, None] * D_HEAD
                    + dims[None, None, :]
                )
                global_extra_mask = (
                    global_extra_lanes[None, :, None] < GLOBAL_MASS_TOP_R
                )
                tl.store(
                    merged_k_ptr + global_extra_output,
                    global_singleton_key,
                    mask=global_extra_mask,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + global_extra_output,
                    global_singleton_value,
                    mask=global_extra_mask,
                    eviction_policy="evict_first",
                )
            if FUSE_CLUSTER_MASS:
                singleton_output = (merged_base + C) * D_HEAD + dims[None, :]
                tl.store(
                    merged_k_ptr + singleton_output,
                    singleton_key,
                    mask=singleton_valid_flat[:, None],
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + singleton_output,
                    singleton_value,
                    mask=singleton_valid_flat[:, None],
                    eviction_policy="evict_first",
                )
            query_col_ids = query_col_start + tl.arange(0, BLOCK_J)
            bias_seed = tl.arange(0, C)
            bias_base = (
                ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                     + query_col_ids[:, None])
                    * ACTIVE_HEADS
                    + head)
                   * K_OUTER
                   + key_outer)
                  * ROWS
                  + key_row)
                 * OUT_C)
                + bias_seed[None, :]
            )
            tl.store(
                bias_ptr + bias_base,
                route_bias,
                eviction_policy="evict_first",
            )
            if FUSE_CLUSTER_MASS:
                tl.store(
                    bias_ptr + bias_base + C,
                    tl.where(has_key, 0.0, float("inf")),
                    eviction_policy="evict_first",
                )
            if GLOBAL_MASS_TOP_R > 0 and MATERIALIZE_GLOBAL_MASS:
                global_bias_mask = (
                    global_extra_lanes[None, :] < GLOBAL_MASS_TOP_R
                )
                tl.store(
                    bias_ptr
                    + global_route_base[:, None]
                    + C
                    + global_extra_lanes[None, :],
                    0.0,
                    mask=global_bias_mask,
                    eviction_policy="evict_first",
                )
