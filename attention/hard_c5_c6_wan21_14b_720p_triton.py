"""Hard-C5/C6/C8 GridMergeAttention for Wan2.1-14B at 720p.

Both variants preserve the maintained two-stage algorithm:

1. Merge each 80-key row into C compressed K/V tokens.
2. Run full attention over all ``21 x 45 x C`` compressed tokens.

Triton requires power-of-two tensor extents, so the merge kernel represents
the seed axis with eight slots and masks slots ``C:8``. Eight query columns
are processed per iteration, keeping the score tile at ``64 x 80`` for both
C5, C6, and C8. The valid math still contains exactly C seeds.
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
QUERY_OUTER_GROUP = 21
SEED_SLOTS = 8

MERGE_BLOCK_J = 8
MERGE_NUM_WARPS = 4
MERGE_NUM_STAGES = 2
MERGE_LOOP_STAGES = 2
COMPACT_C5_BLOCK_J = 16

ROUTE_BLOCK_I = 64
ROUTE_BLOCK_TOKENS = 64
ROUTE_NUM_WARPS = 4
ROUTE_NUM_STAGES = 1
INPUT_PRECISION = "ieee"
CENTROID_CANDIDATES = 11
CENTROID_BLOCK_ROWS = 16
CENTROID_NUM_WARPS = 4
CENTROID_NUM_STAGES = 1


if triton is not None:

    @triton.jit
    def _hard_c5_720p_query_centroid_kernel(
        q_ptr,
        centroid_ptr,
        Q_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        CANDIDATES: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        """Run one local Lloyd update and preserve each cluster's mean norm."""

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)

        query_col = block_id % COLS
        block_id //= COLS
        key_row = block_id % ROWS
        query_outer = block_id // ROWS

        candidate_slot = tl.arange(0, BLOCK_ROWS)
        seed_slot = tl.arange(0, BLOCK_ROWS)
        dims = tl.arange(0, 128)
        valid_candidate = candidate_slot < CANDIDATES
        valid_seed = seed_slot < C

        candidate_start = tl.minimum(
            tl.maximum(key_row - CANDIDATES // 2, 0),
            ROWS - CANDIDATES,
        )
        candidate_rows = candidate_start + candidate_slot
        # round(linspace(0, 10, 5)) from the real-QKV screening definition.
        seed_position = tl.where(
            seed_slot == 0,
            0,
            tl.where(
                seed_slot == 1,
                2,
                tl.where(seed_slot == 2, 5, tl.where(seed_slot == 3, 8, 10)),
            ),
        )
        seed_rows = candidate_start + seed_position

        if HEAD_MAJOR:
            q_base = (
                (batch * TOTAL_HEADS + global_head)
                * Q_OUTER
                * ROWS
                * COLS
                * D_HEAD
                + query_outer * ROWS * COLS * D_HEAD
            )
            candidate_offsets = (
                q_base
                + (candidate_rows[:, None] * COLS + query_col) * D_HEAD
                + dims[None, :]
            )
            seed_offsets = (
                q_base
                + (seed_rows[:, None] * COLS + query_col) * D_HEAD
                + dims[None, :]
            )
        else:
            q_base = (
                (batch * Q_OUTER + query_outer)
                * ROWS
                * COLS
                * TOTAL_HEADS
                * D_HEAD
            )
            candidate_offsets = (
                q_base
                + (
                    (candidate_rows[:, None] * COLS + query_col) * TOTAL_HEADS
                    + global_head
                )
                * D_HEAD
                + dims[None, :]
            )
            seed_offsets = (
                q_base
                + (
                    (seed_rows[:, None] * COLS + query_col) * TOTAL_HEADS
                    + global_head
                )
                * D_HEAD
                + dims[None, :]
            )

        candidates = tl.load(
            q_ptr + candidate_offsets,
            mask=valid_candidate[:, None],
            other=0.0,
        )
        initial_centers = tl.load(
            q_ptr + seed_offsets,
            mask=valid_seed[:, None],
            other=0.0,
        )
        candidates_f32 = candidates.to(tl.float32)
        initial_centers_f32 = initial_centers.to(tl.float32)
        candidate_sq_norm = tl.sum(candidates_f32 * candidates_f32, axis=1)
        center_sq_norm = tl.sum(
            initial_centers_f32 * initial_centers_f32, axis=1
        )
        distance = (
            candidate_sq_norm[:, None]
            + center_sq_norm[None, :]
            - 2.0
            * tl.dot(
                candidates_f32,
                tl.trans(initial_centers_f32),
                input_precision=INPUT_PRECISION,
            )
        )
        distance = tl.where(valid_seed[None, :], distance, float("inf"))
        _, assignment = tl.max(
            -distance,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        membership = (
            (assignment[:, None] == seed_slot[None, :])
            & valid_candidate[:, None]
            & valid_seed[None, :]
        )
        membership_f32 = membership.to(tl.float32)
        count = tl.sum(membership_f32, axis=0)
        safe_count = tl.where(count > 0.0, count, 1.0)
        weights = membership_f32 / safe_count[None, :]
        centroids = tl.dot(
            tl.trans(weights),
            candidates_f32,
            input_precision=INPUT_PRECISION,
        )

        candidate_norm = tl.sqrt(tl.maximum(candidate_sq_norm, 0.0))
        target_norm = tl.sum(
            membership_f32 * candidate_norm[:, None], axis=0
        ) / safe_count
        centroid_norm = tl.sqrt(
            tl.maximum(tl.sum(centroids * centroids, axis=1), 1.0e-12)
        )
        centroids *= (target_norm / centroid_norm)[:, None]
        centroids = tl.where(
            (count > 0.0)[:, None], centroids, initial_centers_f32
        )

        centroid_base = (
            (((((batch_i64 * Q_OUTER + query_outer) * ROWS + key_row) * COLS
                + query_col)
               * ACTIVE_HEADS
               + head)
              * C)
            + seed_slot
        )
        centroid_offsets = centroid_base[:, None] * D_HEAD + dims[None, :]
        tl.store(
            centroid_ptr + centroid_offsets,
            centroids,
            mask=valid_seed[:, None],
        )

    @triton.jit
    def _hard_c56_merge_kernel(
        q_ptr,
        centroid_ptr,
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
        HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        C_PAD: tl.constexpr,
        BLOCK_J: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_QUERY_CENTROIDS: tl.constexpr,
        USE_EXTERNAL_SEEDS: tl.constexpr,
        GAUSSIAN_PARTITION: tl.constexpr,
        MAX_PARTITION: tl.constexpr,
        SPLIT_KV: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Compress one 80-key row for all query frames and one head."""

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

        flat = tl.arange(0, 64)
        seed_slot = flat % C_PAD
        query_col_offset = flat // C_PAD
        valid_seed = seed_slot < C
        main_cols = tl.arange(0, 64)
        tail_cols = 64 + tl.arange(0, 16)
        dims = tl.arange(0, 128)

        if HEAD_MAJOR:
            row_base = (
                ((batch * TOTAL_HEADS + global_head) * K_OUTER + key_outer)
                * ROWS
                * COLS
                * D_HEAD
                + key_row * COLS * D_HEAD
            )
            key_offsets = row_base + main_cols[:, None] * D_HEAD + dims[None, :]
            tail_offsets = row_base + tail_cols[:, None] * D_HEAD + dims[None, :]
        else:
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
            tail_offsets = (
                row_base
                + (tail_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
                + dims[None, :]
            )

        key_main = tl.load(k_ptr + key_offsets, eviction_policy="evict_last")
        key_tail = tl.load(k_ptr + tail_offsets, eviction_policy="evict_last")
        value_main = tl.load(v_ptr + key_offsets, eviction_policy="evict_last")
        value_tail = tl.load(v_ptr + tail_offsets, eviction_policy="evict_last")

        # A contiguous local window keeps seed loads coalesced and always
        # includes the query row whose index matches this key row.
        seed_start = tl.minimum(tl.maximum(key_row - C // 2, 0), ROWS - C)
        fixed_seed_rows = seed_start + seed_slot

        for q_offset in tl.range(0, Q_GROUP, num_stages=LOOP_STAGES):
            query_outer = query_group * Q_GROUP + q_offset
            if HEAD_MAJOR:
                q_base = (
                    (batch * TOTAL_HEADS + global_head)
                    * Q_OUTER
                    * ROWS
                    * COLS
                    * D_HEAD
                    + query_outer * ROWS * COLS * D_HEAD
                )
            else:
                q_base = (
                    (batch * Q_OUTER + query_outer)
                    * ROWS
                    * COLS
                    * TOTAL_HEADS
                    * D_HEAD
                )

            for query_col_start in tl.range(0, COLS, BLOCK_J):
                query_cols = query_col_start + query_col_offset
                if USE_QUERY_CENTROIDS:
                    centroid_base = (
                        (((((batch_i64 * Q_OUTER + query_outer) * ROWS + key_row)
                            * COLS
                            + query_cols)
                           * ACTIVE_HEADS
                           + head)
                          * C)
                        + seed_slot
                    )
                    seed_offsets = (
                        centroid_base[:, None] * D_HEAD + dims[None, :]
                    )
                    seed_query = tl.load(
                        centroid_ptr + seed_offsets,
                        mask=valid_seed[:, None],
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                else:
                    if USE_EXTERNAL_SEEDS:
                        router_slot = (query_outer * ROUTER_FRAMES) // Q_OUTER
                        seed_base = (
                            ((((((batch_i64 * ROUTER_FRAMES + router_slot)
                                  * K_OUTER + key_outer)
                                 * ROWS + key_row)
                                * COLS + query_cols)
                               * TOTAL_HEADS + global_head)
                              * C)
                            + seed_slot
                        )
                        seed_rows = tl.load(
                            seed_row_ptr + seed_base,
                            mask=valid_seed,
                            other=0,
                        ).to(tl.int32)
                    else:
                        seed_rows = fixed_seed_rows
                    if HEAD_MAJOR:
                        seed_offsets = (
                            q_base
                            + (seed_rows[:, None] * COLS + query_cols[:, None])
                            * D_HEAD
                            + dims[None, :]
                        )
                    else:
                        seed_offsets = (
                            q_base
                            + (
                                (seed_rows[:, None] * COLS + query_cols[:, None])
                                * TOTAL_HEADS
                                + global_head
                            )
                            * D_HEAD
                            + dims[None, :]
                        )
                    seed_query = tl.load(
                        q_ptr + seed_offsets,
                        mask=valid_seed[:, None],
                        other=0.0,
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

                if GAUSSIAN_PARTITION:
                    score_mean = (
                        tl.sum(scores_main, axis=1)
                        + tl.sum(scores_tail, axis=1)
                    ) / COLS
                    score_second_moment = (
                        tl.sum(scores_main * scores_main, axis=1)
                        + tl.sum(scores_tail * scores_tail, axis=1)
                    ) / COLS
                    score_variance = tl.maximum(
                        score_second_moment - score_mean * score_mean, 0.0
                    )
                    # Scores are in log2 units. The Gaussian cumulant term is
                    # ln(2) * variance / 2; log2(COLS) cancels across seeds.
                    log_norm = score_mean + 0.34657359027997264 * score_variance
                elif MAX_PARTITION:
                    log_norm = tl.maximum(
                        tl.max(scores_main, axis=1),
                        tl.max(scores_tail, axis=1),
                    )
                else:
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
                raw_scores_main = tl.reshape(
                    scores_main, (BLOCK_J, C_PAD, 64)
                )
                raw_scores_tail = tl.reshape(
                    scores_tail, (BLOCK_J, C_PAD, 16)
                )
                log_probs_main = tl.reshape(
                    scores_main - log_norm[:, None],
                    (BLOCK_J, C_PAD, 64),
                )
                log_probs_tail = tl.reshape(
                    scores_tail - log_norm[:, None],
                    (BLOCK_J, C_PAD, 16),
                )
                seed_mask = tl.arange(0, C_PAD)[None, :, None] < C
                log_probs_main = tl.where(
                    seed_mask, log_probs_main, -float("inf")
                )
                log_probs_tail = tl.where(
                    seed_mask, log_probs_tail, -float("inf")
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
                cluster = tl.arange(0, C_PAD)[None, :, None]
                mask_main = route_main[:, None, :] == cluster
                mask_tail = route_tail[:, None, :] == cluster
                if GAUSSIAN_PARTITION:
                    raw_winner_main = tl.sum(
                        tl.where(mask_main, raw_scores_main, 0.0), axis=1
                    )
                    raw_winner_tail = tl.sum(
                        tl.where(mask_tail, raw_scores_tail, 0.0), axis=1
                    )
                    assigned = (
                        tl.sum(mask_main.to(tl.int32), axis=2)
                        + tl.sum(mask_tail.to(tl.int32), axis=2)
                    ) > 0
                    cluster_max = tl.maximum(
                        tl.max(
                            tl.where(
                                mask_main,
                                raw_winner_main[:, None, :],
                                -float("inf"),
                            ),
                            axis=2,
                        ),
                        tl.max(
                            tl.where(
                                mask_tail,
                                raw_winner_tail[:, None, :],
                                -float("inf"),
                            ),
                            axis=2,
                        ),
                    )
                    safe_cluster_max = tl.where(assigned, cluster_max, 0.0)
                    probs_main = tl.where(
                        mask_main,
                        tl.exp2(
                            raw_winner_main[:, None, :]
                            - safe_cluster_max[:, :, None]
                        ),
                        0.0,
                    )
                    probs_tail = tl.where(
                        mask_tail,
                        tl.exp2(
                            raw_winner_tail[:, None, :]
                            - safe_cluster_max[:, :, None]
                        ),
                        0.0,
                    )
                else:
                    probs_main = tl.where(
                        mask_main, tl.exp2(winner_main)[:, None, :], 0.0
                    )
                    probs_tail = tl.where(
                        mask_tail, tl.exp2(winner_tail)[:, None, :], 0.0
                    )
                prob_sum = tl.sum(probs_main, axis=2) + tl.sum(probs_tail, axis=2)
                valid_clusters = tl.arange(0, C_PAD)[None, :] < C
                has_key = (prob_sum > 0.0) & valid_clusters
                safe_sum = tl.where(has_key, prob_sum, 1.0)
                if GAUSSIAN_PARTITION:
                    expected = tl.sum(
                        probs_main * raw_winner_main[:, None, :], axis=2
                    )
                    expected += tl.sum(
                        probs_tail * raw_winner_tail[:, None, :], axis=2
                    )
                else:
                    expected = tl.sum(
                        probs_main * winner_main[:, None, :], axis=2
                    )
                    expected += tl.sum(
                        probs_tail * winner_tail[:, None, :], axis=2
                    )
                inverse_sum = 1.0 / safe_sum
                expected *= inverse_sum
                weights_main = probs_main * inverse_sum[:, :, None]
                weights_tail = probs_tail * inverse_sum[:, :, None]
                if GAUSSIAN_PARTITION:
                    route_bias = tl.where(
                        has_key,
                        (
                            expected
                            - safe_cluster_max
                            - tl.log2(safe_sum)
                        )
                        * 0.6931471805599453,
                        float("inf"),
                    )
                else:
                    route_bias = tl.where(
                        has_key,
                        (expected - tl.log2(safe_sum)) * 0.6931471805599453,
                        float("inf"),
                    )

                weights_main = tl.reshape(weights_main, (64, 64))
                weights_tail = tl.reshape(weights_tail, (64, 16))
                if SPLIT_KV:
                    merged_key = tl.dot(
                        weights_main.to(key_main.dtype),
                        key_main,
                        input_precision=INPUT_PRECISION,
                    )
                    merged_key += tl.dot(
                        weights_tail.to(key_tail.dtype),
                        key_tail,
                        input_precision=INPUT_PRECISION,
                    )
                    merged_value = tl.dot(
                        weights_main.to(value_main.dtype),
                        value_main,
                        input_precision=INPUT_PRECISION,
                    )
                    merged_value += tl.dot(
                        weights_tail.to(value_tail.dtype),
                        value_tail,
                        input_precision=INPUT_PRECISION,
                    )
                else:
                    merged_pair = tl.dot(
                        weights_main.to(key_main.dtype),
                        tl.interleave(key_main, value_main),
                        input_precision=INPUT_PRECISION,
                    )
                    merged_pair += tl.dot(
                        weights_tail.to(key_tail.dtype),
                        tl.interleave(key_tail, value_tail),
                        input_precision=INPUT_PRECISION,
                    )
                    merged_key, merged_value = tl.split(
                        tl.reshape(merged_pair, (64, 128, 2))
                    )

                merged_base = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + seed_slot[:, None]
                )
                merged_output = merged_base * D_HEAD + dims[None, :]
                tl.store(
                    merged_k_ptr + merged_output,
                    merged_key,
                    mask=valid_seed[:, None],
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + merged_output,
                    merged_value,
                    mask=valid_seed[:, None],
                    eviction_policy="evict_first",
                )

                query_cols = query_col_start + tl.arange(0, BLOCK_J)
                seed_slots = tl.arange(0, C_PAD)
                bias_base = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + seed_slots[None, :]
                )
                tl.store(
                    bias_ptr + bias_base,
                    route_bias,
                    mask=seed_slots[None, :] < C,
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _hard_c6_compact_merge_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        BLOCK_J: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Compress fixed Hard-C6 seeds as one C4 tile plus one C2 tile."""

        tl.static_assert(C == 6)
        tl.static_assert(BLOCK_J == 16)

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        key_outer = block_id % K_OUTER
        key_row = block_id // K_OUTER

        flat4 = tl.arange(0, 64)
        seed4 = flat4 % 4
        col4_offset = flat4 // 4
        flat2 = tl.arange(0, 32)
        seed2 = flat2 % 2
        col2_offset = flat2 // 2
        col_offset = tl.arange(0, BLOCK_J)
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
        tail_offsets = (
            row_base
            + (tail_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
            + dims[None, :]
        )
        key_main = tl.load(k_ptr + key_offsets, eviction_policy="evict_last")
        key_tail = tl.load(k_ptr + tail_offsets, eviction_policy="evict_last")
        value_main = tl.load(v_ptr + key_offsets, eviction_policy="evict_last")
        value_tail = tl.load(v_ptr + tail_offsets, eviction_policy="evict_last")
        key_value_main = tl.interleave(key_main, value_main)
        key_value_tail = tl.interleave(key_tail, value_tail)
        seed_start = tl.minimum(tl.maximum(key_row - C // 2, 0), ROWS - C)
        seed_rows4 = seed_start + seed4
        seed_rows2 = seed_start + 4 + seed2
        log2e: tl.constexpr = 1.4426950408889634
        ln2: tl.constexpr = 0.6931471805599453

        for query_col_start in tl.range(0, COLS, BLOCK_J):
            query_cols4 = query_col_start + col4_offset
            query_cols2 = query_col_start + col2_offset
            query_cols = query_col_start + col_offset
            for query_outer in tl.range(0, Q_OUTER, num_stages=LOOP_STAGES):
                q_base = (
                    (batch * Q_OUTER + query_outer)
                    * ROWS
                    * COLS
                    * TOTAL_HEADS
                    * D_HEAD
                )
                seed_offsets4 = (
                    q_base
                    + (
                        (seed_rows4[:, None] * COLS + query_cols4[:, None])
                        * TOTAL_HEADS
                        + global_head
                    )
                    * D_HEAD
                    + dims[None, :]
                )
                seed_offsets2 = (
                    q_base
                    + (
                        (seed_rows2[:, None] * COLS + query_cols2[:, None])
                        * TOTAL_HEADS
                        + global_head
                    )
                    * D_HEAD
                    + dims[None, :]
                )
                seed_query4 = tl.load(
                    q_ptr + seed_offsets4, eviction_policy="evict_last"
                )
                seed_query2 = tl.load(
                    q_ptr + seed_offsets2, eviction_policy="evict_last"
                )

                scores4_main = tl.dot(
                    seed_query4,
                    tl.trans(key_main),
                    input_precision=INPUT_PRECISION,
                ) * (SCALE * log2e)
                scores4_tail = tl.dot(
                    seed_query4,
                    tl.trans(key_tail),
                    input_precision=INPUT_PRECISION,
                ) * (SCALE * log2e)
                scores2_main = tl.dot(
                    seed_query2,
                    tl.trans(key_main),
                    input_precision=INPUT_PRECISION,
                ) * (SCALE * log2e)
                scores2_tail = tl.dot(
                    seed_query2,
                    tl.trans(key_tail),
                    input_precision=INPUT_PRECISION,
                ) * (SCALE * log2e)

                norm4_max = tl.maximum(
                    tl.max(scores4_main, axis=1), tl.max(scores4_tail, axis=1)
                )
                norm4_sum = tl.sum(
                    tl.exp2(scores4_main - norm4_max[:, None]), axis=1
                ) + tl.sum(tl.exp2(scores4_tail - norm4_max[:, None]), axis=1)
                norm2_max = tl.maximum(
                    tl.max(scores2_main, axis=1), tl.max(scores2_tail, axis=1)
                )
                norm2_sum = tl.sum(
                    tl.exp2(scores2_main - norm2_max[:, None]), axis=1
                ) + tl.sum(tl.exp2(scores2_tail - norm2_max[:, None]), axis=1)
                logp4_main = tl.reshape(
                    scores4_main - (norm4_max + tl.log2(norm4_sum))[:, None],
                    (BLOCK_J, 4, 64),
                )
                logp4_tail = tl.reshape(
                    scores4_tail - (norm4_max + tl.log2(norm4_sum))[:, None],
                    (BLOCK_J, 4, 16),
                )
                logp2_main = tl.reshape(
                    scores2_main - (norm2_max + tl.log2(norm2_sum))[:, None],
                    (BLOCK_J, 2, 64),
                )
                logp2_tail = tl.reshape(
                    scores2_tail - (norm2_max + tl.log2(norm2_sum))[:, None],
                    (BLOCK_J, 2, 16),
                )

                winner4_main, route4_main = tl.max(
                    logp4_main,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                winner4_tail, route4_tail = tl.max(
                    logp4_tail,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                winner2_main, route2_main = tl.max(
                    logp2_main,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                winner2_tail, route2_tail = tl.max(
                    logp2_tail,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                choose2_main = winner2_main > winner4_main
                choose2_tail = winner2_tail > winner4_tail
                winner_main = tl.where(choose2_main, winner2_main, winner4_main)
                winner_tail = tl.where(choose2_tail, winner2_tail, winner4_tail)
                route_main = tl.where(choose2_main, route2_main + 4, route4_main)
                route_tail = tl.where(choose2_tail, route2_tail + 4, route4_tail)

                cluster4 = tl.arange(0, 4)[None, :, None]
                cluster2 = (4 + tl.arange(0, 2))[None, :, None]
                probability_main = tl.exp2(winner_main)
                probability_tail = tl.exp2(winner_tail)
                probs4_main = tl.where(
                    route_main[:, None, :] == cluster4,
                    probability_main[:, None, :],
                    0.0,
                )
                probs4_tail = tl.where(
                    route_tail[:, None, :] == cluster4,
                    probability_tail[:, None, :],
                    0.0,
                )
                probs2_main = tl.where(
                    route_main[:, None, :] == cluster2,
                    probability_main[:, None, :],
                    0.0,
                )
                probs2_tail = tl.where(
                    route_tail[:, None, :] == cluster2,
                    probability_tail[:, None, :],
                    0.0,
                )

                sum4 = tl.sum(probs4_main, axis=2) + tl.sum(probs4_tail, axis=2)
                sum2 = tl.sum(probs2_main, axis=2) + tl.sum(probs2_tail, axis=2)
                has4 = sum4 > 0.0
                has2 = sum2 > 0.0
                safe4 = tl.where(has4, sum4, 1.0)
                safe2 = tl.where(has2, sum2, 1.0)
                inv4 = 1.0 / safe4
                inv2 = 1.0 / safe2
                weights4_main = probs4_main * inv4[:, :, None]
                weights4_tail = probs4_tail * inv4[:, :, None]
                weights2_main = probs2_main * inv2[:, :, None]
                weights2_tail = probs2_tail * inv2[:, :, None]
                expected4 = tl.sum(
                    probs4_main * winner_main[:, None, :], axis=2
                ) + tl.sum(probs4_tail * winner_tail[:, None, :], axis=2)
                expected2 = tl.sum(
                    probs2_main * winner_main[:, None, :], axis=2
                ) + tl.sum(probs2_tail * winner_tail[:, None, :], axis=2)
                bias4 = tl.where(
                    has4, (expected4 * inv4 - tl.log2(safe4)) * ln2, float("inf")
                )
                bias2 = tl.where(
                    has2, (expected2 * inv2 - tl.log2(safe2)) * ln2, float("inf")
                )

                merged_pair4 = tl.dot(
                    tl.reshape(weights4_main, (64, 64)).to(key_main.dtype),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_pair4 += tl.dot(
                    tl.reshape(weights4_tail, (64, 16)).to(key_tail.dtype),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                merged_key4, merged_value4 = tl.split(
                    tl.reshape(merged_pair4, (64, 128, 2))
                )
                merged_pair2 = tl.dot(
                    tl.reshape(weights2_main, (32, 64)).to(key_main.dtype),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_pair2 += tl.dot(
                    tl.reshape(weights2_tail, (32, 16)).to(key_tail.dtype),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                merged_key2, merged_value2 = tl.split(
                    tl.reshape(merged_pair2, (32, 128, 2))
                )

                output_base4 = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols4[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + seed4[:, None]
                )
                output_base2 = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols2[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + 4
                    + seed2[:, None]
                )
                tl.store(
                    merged_k_ptr + output_base4 * D_HEAD + dims[None, :],
                    merged_key4,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + output_base4 * D_HEAD + dims[None, :],
                    merged_value4,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_k_ptr + output_base2 * D_HEAD + dims[None, :],
                    merged_key2,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + output_base2 * D_HEAD + dims[None, :],
                    merged_value2,
                    eviction_policy="evict_first",
                )

                bias_base4 = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + tl.arange(0, 4)[None, :]
                )
                bias_base2 = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + 4
                    + tl.arange(0, 2)[None, :]
                )
                tl.store(
                    bias_ptr + bias_base4,
                    bias4,
                    eviction_policy="evict_first",
                )
                tl.store(
                    bias_ptr + bias_base2,
                    bias2,
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _hard_c5_temporal_shared_merge_kernel(
        q_ptr,
        seed_row_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
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
        BLOCK_J: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Compact C5 merge with four-seed and singleton tensor-core tiles."""

        tl.static_assert(C == 5)
        tl.static_assert(BLOCK_J == 16)
        tl.static_assert(ROUTER_FRAMES >= 1)
        tl.static_assert(ROUTER_FRAMES <= 3)

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        key_outer = block_id % K_OUTER
        key_row = block_id // K_OUTER

        flat4 = tl.arange(0, 64)
        seed4 = flat4 % 4
        col4_offset = flat4 // 4
        col_offset = tl.arange(0, BLOCK_J)
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
        tail_offsets = (
            row_base
            + (tail_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
            + dims[None, :]
        )
        key_main = tl.load(k_ptr + key_offsets, eviction_policy="evict_last")
        key_tail = tl.load(k_ptr + tail_offsets, eviction_policy="evict_last")
        value_main = tl.load(v_ptr + key_offsets, eviction_policy="evict_last")
        value_tail = tl.load(v_ptr + tail_offsets, eviction_policy="evict_last")
        key_value_main = tl.interleave(key_main, value_main)
        key_value_tail = tl.interleave(key_tail, value_tail)
        log2e: tl.constexpr = 1.4426950408889634
        ln2: tl.constexpr = 0.6931471805599453

        for query_col_start in tl.range(0, COLS, BLOCK_J):
            query_cols4 = query_col_start + col4_offset
            query_cols = query_col_start + col_offset
            table_frame_base = batch_i64 * ROUTER_FRAMES

            seed_base4_0 = (table_frame_base * COLS + query_cols4)
            seed_base4_0 = (seed_base4_0 * ACTIVE_HEADS + head)
            seed_base4_0 = (seed_base4_0 * K_OUTER + key_outer)
            seed_base4_0 = (seed_base4_0 * ROWS + key_row) * C + seed4
            seed_rows4_0 = tl.load(seed_row_ptr + seed_base4_0).to(tl.int32)
            seed_base5_0 = (table_frame_base * COLS + query_cols)
            seed_base5_0 = (seed_base5_0 * ACTIVE_HEADS + head)
            seed_base5_0 = (seed_base5_0 * K_OUTER + key_outer)
            seed_base5_0 = (seed_base5_0 * ROWS + key_row) * C + 4
            seed_rows5_0 = tl.load(seed_row_ptr + seed_base5_0).to(tl.int32)

            if ROUTER_FRAMES > 1:
                seed_base4_1 = ((table_frame_base + 1) * COLS + query_cols4)
                seed_base4_1 = (seed_base4_1 * ACTIVE_HEADS + head)
                seed_base4_1 = (seed_base4_1 * K_OUTER + key_outer)
                seed_base4_1 = (seed_base4_1 * ROWS + key_row) * C + seed4
                seed_rows4_1 = tl.load(seed_row_ptr + seed_base4_1).to(tl.int32)
                seed_base5_1 = ((table_frame_base + 1) * COLS + query_cols)
                seed_base5_1 = (seed_base5_1 * ACTIVE_HEADS + head)
                seed_base5_1 = (seed_base5_1 * K_OUTER + key_outer)
                seed_base5_1 = (seed_base5_1 * ROWS + key_row) * C + 4
                seed_rows5_1 = tl.load(seed_row_ptr + seed_base5_1).to(tl.int32)
            if ROUTER_FRAMES > 2:
                seed_base4_2 = ((table_frame_base + 2) * COLS + query_cols4)
                seed_base4_2 = (seed_base4_2 * ACTIVE_HEADS + head)
                seed_base4_2 = (seed_base4_2 * K_OUTER + key_outer)
                seed_base4_2 = (seed_base4_2 * ROWS + key_row) * C + seed4
                seed_rows4_2 = tl.load(seed_row_ptr + seed_base4_2).to(tl.int32)
                seed_base5_2 = ((table_frame_base + 2) * COLS + query_cols)
                seed_base5_2 = (seed_base5_2 * ACTIVE_HEADS + head)
                seed_base5_2 = (seed_base5_2 * K_OUTER + key_outer)
                seed_base5_2 = (seed_base5_2 * ROWS + key_row) * C + 4
                seed_rows5_2 = tl.load(seed_row_ptr + seed_base5_2).to(tl.int32)

            for query_outer in tl.range(0, Q_OUTER, num_stages=LOOP_STAGES):
                router_slot = (query_outer * ROUTER_FRAMES) // Q_OUTER
                active_rows4 = seed_rows4_0
                active_rows5 = seed_rows5_0
                if ROUTER_FRAMES > 1:
                    active_rows4 = tl.where(
                        router_slot == 1, seed_rows4_1, active_rows4
                    )
                    active_rows5 = tl.where(
                        router_slot == 1, seed_rows5_1, active_rows5
                    )
                if ROUTER_FRAMES > 2:
                    active_rows4 = tl.where(
                        router_slot == 2, seed_rows4_2, active_rows4
                    )
                    active_rows5 = tl.where(
                        router_slot == 2, seed_rows5_2, active_rows5
                    )

                q_base = (
                    (batch * Q_OUTER + query_outer)
                    * ROWS
                    * COLS
                    * TOTAL_HEADS
                    * D_HEAD
                )
                seed_offsets4 = (
                    q_base
                    + (
                        (active_rows4[:, None] * COLS + query_cols4[:, None])
                        * TOTAL_HEADS
                        + global_head
                    )
                    * D_HEAD
                    + dims[None, :]
                )
                seed_offsets5 = (
                    q_base
                    + (
                        (active_rows5[:, None] * COLS + query_cols[:, None])
                        * TOTAL_HEADS
                        + global_head
                    )
                    * D_HEAD
                    + dims[None, :]
                )
                seed_query4 = tl.load(
                    q_ptr + seed_offsets4, eviction_policy="evict_last"
                )
                seed_query5 = tl.load(
                    q_ptr + seed_offsets5, eviction_policy="evict_last"
                )

                scores4_main = tl.dot(
                    seed_query4, tl.trans(key_main), input_precision=INPUT_PRECISION
                ) * (SCALE * log2e)
                scores4_tail = tl.dot(
                    seed_query4, tl.trans(key_tail), input_precision=INPUT_PRECISION
                ) * (SCALE * log2e)
                scores5_main = tl.dot(
                    seed_query5, tl.trans(key_main), input_precision=INPUT_PRECISION
                ) * (SCALE * log2e)
                scores5_tail = tl.dot(
                    seed_query5, tl.trans(key_tail), input_precision=INPUT_PRECISION
                ) * (SCALE * log2e)

                norm4_max = tl.maximum(
                    tl.max(scores4_main, axis=1), tl.max(scores4_tail, axis=1)
                )
                norm4_sum = tl.sum(
                    tl.exp2(scores4_main - norm4_max[:, None]), axis=1
                ) + tl.sum(tl.exp2(scores4_tail - norm4_max[:, None]), axis=1)
                norm5_max = tl.maximum(
                    tl.max(scores5_main, axis=1), tl.max(scores5_tail, axis=1)
                )
                norm5_sum = tl.sum(
                    tl.exp2(scores5_main - norm5_max[:, None]), axis=1
                ) + tl.sum(tl.exp2(scores5_tail - norm5_max[:, None]), axis=1)
                logp4_main = tl.reshape(
                    scores4_main - (norm4_max + tl.log2(norm4_sum))[:, None],
                    (BLOCK_J, 4, 64),
                )
                logp4_tail = tl.reshape(
                    scores4_tail - (norm4_max + tl.log2(norm4_sum))[:, None],
                    (BLOCK_J, 4, 16),
                )
                logp5_main = scores5_main - (
                    norm5_max + tl.log2(norm5_sum)
                )[:, None]
                logp5_tail = scores5_tail - (
                    norm5_max + tl.log2(norm5_sum)
                )[:, None]

                winner4_main, route4_main = tl.max(
                    logp4_main,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                winner4_tail, route4_tail = tl.max(
                    logp4_tail,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                choose5_main = logp5_main > winner4_main
                choose5_tail = logp5_tail > winner4_tail
                winner_main = tl.where(choose5_main, logp5_main, winner4_main)
                winner_tail = tl.where(choose5_tail, logp5_tail, winner4_tail)
                route_main = tl.where(choose5_main, 4, route4_main)
                route_tail = tl.where(choose5_tail, 4, route4_tail)

                cluster4 = tl.arange(0, 4)[None, :, None]
                probs4_main = tl.where(
                    route_main[:, None, :] == cluster4,
                    tl.exp2(winner_main)[:, None, :],
                    0.0,
                )
                probs4_tail = tl.where(
                    route_tail[:, None, :] == cluster4,
                    tl.exp2(winner_tail)[:, None, :],
                    0.0,
                )
                probs5_main = tl.where(
                    route_main == 4, tl.exp2(winner_main), 0.0
                )
                probs5_tail = tl.where(
                    route_tail == 4, tl.exp2(winner_tail), 0.0
                )

                sum4 = tl.sum(probs4_main, axis=2) + tl.sum(probs4_tail, axis=2)
                sum5 = tl.sum(probs5_main, axis=1) + tl.sum(probs5_tail, axis=1)
                has4 = sum4 > 0.0
                has5 = sum5 > 0.0
                safe4 = tl.where(has4, sum4, 1.0)
                safe5 = tl.where(has5, sum5, 1.0)
                inv4 = 1.0 / safe4
                inv5 = 1.0 / safe5
                weights4_main = probs4_main * inv4[:, :, None]
                weights4_tail = probs4_tail * inv4[:, :, None]
                weights5_main = probs5_main * inv5[:, None]
                weights5_tail = probs5_tail * inv5[:, None]

                expected4 = tl.sum(
                    probs4_main * winner_main[:, None, :], axis=2
                ) + tl.sum(probs4_tail * winner_tail[:, None, :], axis=2)
                expected5 = tl.sum(probs5_main * winner_main, axis=1) + tl.sum(
                    probs5_tail * winner_tail, axis=1
                )
                bias4 = tl.where(
                    has4, (expected4 * inv4 - tl.log2(safe4)) * ln2, float("inf")
                )
                bias5 = tl.where(
                    has5, (expected5 * inv5 - tl.log2(safe5)) * ln2, float("inf")
                )

                merged_pair4 = tl.dot(
                    tl.reshape(weights4_main, (64, 64)).to(key_main.dtype),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_pair4 += tl.dot(
                    tl.reshape(weights4_tail, (64, 16)).to(key_tail.dtype),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                merged_key4, merged_value4 = tl.split(
                    tl.reshape(merged_pair4, (64, 128, 2))
                )
                merged_pair5 = tl.dot(
                    weights5_main.to(key_main.dtype),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_pair5 += tl.dot(
                    weights5_tail.to(key_tail.dtype),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                merged_key5, merged_value5 = tl.split(
                    tl.reshape(merged_pair5, (BLOCK_J, 128, 2))
                )

                output_base4 = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols4[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + seed4[:, None]
                )
                output_base5 = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols)
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + 4
                )
                tl.store(
                    merged_k_ptr + output_base4 * D_HEAD + dims[None, :],
                    merged_key4,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + output_base4 * D_HEAD + dims[None, :],
                    merged_value4,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_k_ptr + output_base5[:, None] * D_HEAD + dims[None, :],
                    merged_key5,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr + output_base5[:, None] * D_HEAD + dims[None, :],
                    merged_value5,
                    eviction_policy="evict_first",
                )

                bias_base4 = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + tl.arange(0, 4)[None, :]
                )
                tl.store(
                    bias_ptr + bias_base4,
                    bias4,
                    eviction_policy="evict_first",
                )
                tl.store(
                    bias_ptr + output_base5,
                    bias5,
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _hard_c56_route_kernel(
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
        HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        STORE_LOG_Z: tl.constexpr,
        KEY_FP8: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Full attention over all ``K_OUTER x 45 x C`` compressed tokens."""

        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        query_outer = query_group // COLS
        query_col = query_group % COLS

        query_rows = tl.arange(0, BLOCK_I)
        tile_tokens = tl.arange(0, BLOCK_TOKENS)
        dims = tl.arange(0, 128)
        query_mask = query_rows < ROWS
        q_token = (query_outer * ROWS + query_rows[:, None]) * COLS + query_col
        if HEAD_MAJOR:
            q_address = (
                ((batch * TOTAL_HEADS + global_head) * Q_OUTER * ROWS * COLS + q_token)
                * D_HEAD
                + dims[None, :]
            )
        else:
            q_address = (
                (((batch * Q_OUTER * ROWS * COLS + q_token) * TOTAL_HEADS + global_head)
                 * D_HEAD)
                + dims[None, :]
            )
        query = tl.load(
            q_ptr + q_address,
            mask=query_mask[:, None],
            other=0.0,
        )
        if KEY_FP8:
            query_for_dot = query.to(tl.float8e4nv)
        else:
            query_for_dot = query

        total_tokens = K_OUTER * ROWS * C
        route_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * total_tokens
        )
        row_max = tl.full((BLOCK_I,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_I,), tl.float32)
        accumulator = tl.zeros((BLOCK_I, 128), tl.float32)

        for token_start in tl.range(
            0,
            total_tokens,
            BLOCK_TOKENS,
            num_stages=LOOP_STAGES,
        ):
            tokens = token_start + tile_tokens
            token_mask = tokens < total_tokens
            merged_base = route_base + tokens[:, None]
            merged_key = tl.load(
                merged_k_ptr + merged_base * D_HEAD + dims[None, :],
                mask=token_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            merged_value = tl.load(
                merged_v_ptr + merged_base * D_HEAD + dims[None, :],
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
                query_for_dot,
                tl.trans(merged_key),
                input_precision=INPUT_PRECISION,
            )
            logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
            logits = tl.where(token_mask[None, :], logits, -float("inf"))
            next_max = tl.maximum(row_max, tl.max(logits, axis=1))
            probabilities = tl.exp2(logits - next_max[:, None])
            old_scale = tl.exp2(row_max - next_max)
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(query.dtype),
                merged_value,
                input_precision=INPUT_PRECISION,
            )
            row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
            row_max = next_max

        tl.store(
            out_ptr + q_address,
            accumulator / row_sum[:, None],
            mask=query_mask[:, None],
        )
        if STORE_LOG_Z:
            log_z_address = (
                ((((batch * Q_OUTER + query_outer) * ROWS + query_rows)
                  * COLS
                  + query_col)
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


def _hard_c_attention(
    q,
    k,
    v,
    c,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    head_major=True,
    query_centroid_norm_window=0,
    temporal_shared_seed_rows=None,
    native_temporal_seed_rows=None,
    return_log_partition=False,
    compact_c6=False,
    compressed_key_fp8=False,
    gaussian_partition=False,
    max_partition=False,
    split_merge_kv=False,
    merge_num_warps=MERGE_NUM_WARPS,
    merge_num_stages=MERGE_NUM_STAGES,
    merge_loop_stages=MERGE_LOOP_STAGES,
    route_block_tokens=ROUTE_BLOCK_TOKENS,
    route_num_warps=ROUTE_NUM_WARPS,
    route_num_stages=ROUTE_NUM_STAGES,
    route_loop_stages=None,
    query_outer_group=QUERY_OUTER_GROUP,
):
    if triton is None:
        raise RuntimeError("Triton is required for hard C5/C6/C8.")
    scale = q.size(-1) ** -0.5 if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if head_major:
        batch, total_heads, query_length, dim = q.shape
        query_outer = query_length // (ROWS * COLS)
        key_outer = k.size(2) // (ROWS * COLS)
    else:
        batch, query_outer, _, _, total_heads, dim = q.shape
        key_outer = k.size(1)
    active_heads = int(head_count or total_heads)
    query_outer_group = int(query_outer_group)
    if query_outer % query_outer_group:
        raise ValueError("query_outer must be divisible by query_outer_group")
    query_groups = query_outer // query_outer_group
    query_centroid_norm_window = int(query_centroid_norm_window or 0)
    use_temporal_shared_seeds = temporal_shared_seed_rows is not None
    use_native_temporal_seeds = native_temporal_seed_rows is not None
    if use_temporal_shared_seeds and use_native_temporal_seeds:
        raise ValueError("only one external seed-row layout may be supplied")
    if query_centroid_norm_window not in {0, CENTROID_CANDIDATES}:
        raise ValueError(
            f"query-centroid C5 supports window 0 or {CENTROID_CANDIDATES}."
        )
    if query_centroid_norm_window and c != 5:
        raise ValueError("query-centroid seeds are implemented only for Hard C5.")
    if query_centroid_norm_window and (ROWS, COLS, dim) != (45, 80, 128):
        raise ValueError(
            "query-centroid C5 is specialized for the Wan 45x80 grid and D=128."
        )
    if compact_c6:
        if c != 6 or head_major:
            raise ValueError("compact C6 requires native-layout Hard C6")
    if use_temporal_shared_seeds:
        if c != 5 or head_major or query_outer != QUERY_OUTER_GROUP:
            raise ValueError(
                "temporally shared external seeds require native 21-frame Hard C5"
            )
        if query_centroid_norm_window:
            raise ValueError(
                "temporally shared seeds and query centroids are mutually exclusive"
            )
        if temporal_shared_seed_rows.device != q.device:
            raise ValueError("temporally shared C5 seed rows must be on the QKV device")
        if temporal_shared_seed_rows.ndim != 7:
            raise ValueError(
                "temporally shared C5 rows must be [B, R, column, active_head, K, row, C]"
            )
        router_frames = int(temporal_shared_seed_rows.size(1))
        if not 1 <= router_frames <= 3:
            raise ValueError("temporally shared C5 supports 1 to 3 router frames")
        expected_seed_shape = (
            batch,
            router_frames,
            COLS,
            active_heads,
            key_outer,
            ROWS,
            c,
        )
        if tuple(temporal_shared_seed_rows.shape) != expected_seed_shape:
            raise ValueError(
                f"temporally shared C5 rows must be {expected_seed_shape}, "
                f"got {tuple(temporal_shared_seed_rows.shape)}"
            )
        temporal_shared_seed_rows = (
            temporal_shared_seed_rows.to(torch.uint8).contiguous()
        )
    elif use_native_temporal_seeds:
        if head_major or query_outer != QUERY_OUTER_GROUP or c not in {4, 6, 8}:
            raise ValueError(
                "native external seeds require 21-frame native-layout Hard C4/C6/C8"
            )
        if query_centroid_norm_window:
            raise ValueError("external seed rows cannot be combined with query centroids")
        if native_temporal_seed_rows.device != q.device:
            raise ValueError("native external seed rows must be on the QKV device")
        if native_temporal_seed_rows.ndim != 7:
            raise ValueError(
                "native external rows must be [B, R, K, row, column, head, C]"
            )
        router_frames = int(native_temporal_seed_rows.size(1))
        if not 1 <= router_frames <= 4:
            raise ValueError("native external seeds support one to four router frames")
        expected_seed_shape = (
            batch,
            router_frames,
            key_outer,
            ROWS,
            COLS,
            total_heads,
            c,
        )
        if tuple(native_temporal_seed_rows.shape) != expected_seed_shape:
            raise ValueError(
                f"native external rows must be {expected_seed_shape}, "
                f"got {tuple(native_temporal_seed_rows.shape)}"
            )
        native_temporal_seed_rows = (
            native_temporal_seed_rows.to(torch.uint8).contiguous()
        )
    else:
        router_frames = 1

    compressed_tokens = (
        batch * query_outer * COLS * active_heads * key_outer * ROWS * c
    )
    merged_key_dtype = torch.float8_e4m3fn if compressed_key_fp8 else q.dtype
    merged_k = torch.empty(
        (compressed_tokens, dim), device=q.device, dtype=merged_key_dtype
    )
    merged_v = torch.empty(
        (compressed_tokens, dim), device=q.device, dtype=q.dtype
    )
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

    if query_centroid_norm_window:
        centroid_tokens = (
            batch * query_outer * ROWS * COLS * active_heads * c
        )
        centroids = torch.empty(
            (centroid_tokens, dim), device=q.device, dtype=q.dtype
        )
        centroid_grid = (
            query_outer * ROWS * COLS,
            batch * active_heads,
        )
        _hard_c5_720p_query_centroid_kernel[centroid_grid](
            q,
            centroids,
            query_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            ROWS,
            COLS,
            c,
            query_centroid_norm_window,
            CENTROID_BLOCK_ROWS,
            INPUT_PRECISION,
            num_warps=CENTROID_NUM_WARPS,
            num_stages=CENTROID_NUM_STAGES,
        )
    else:
        centroids = q

    if compact_c6:
        merge_grid = (key_outer * ROWS, batch * active_heads)
        _hard_c6_compact_merge_kernel[merge_grid](
            q,
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
            ROWS,
            COLS,
            c,
            COMPACT_C5_BLOCK_J,
            merge_loop_stages,
            INPUT_PRECISION,
            SCALE=float(scale),
            num_warps=merge_num_warps,
            num_stages=merge_num_stages,
        )
    elif use_temporal_shared_seeds:
        merge_grid = (key_outer * ROWS, batch * active_heads)
        _hard_c5_temporal_shared_merge_kernel[merge_grid](
            q,
            temporal_shared_seed_rows,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            query_outer,
            router_frames,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            ROWS,
            COLS,
            c,
            COMPACT_C5_BLOCK_J,
            merge_loop_stages,
            INPUT_PRECISION,
            SCALE=float(scale),
            num_warps=merge_num_warps,
            num_stages=merge_num_stages,
        )
    else:
        merge_grid = (query_groups * key_outer * ROWS, batch * active_heads)
        _hard_c56_merge_kernel[merge_grid](
            q,
            centroids,
            native_temporal_seed_rows if use_native_temporal_seeds else q,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            query_outer,
            query_groups,
            query_outer_group,
            router_frames,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            ROWS,
            COLS,
            c,
            SEED_SLOTS,
            MERGE_BLOCK_J,
            merge_loop_stages,
            INPUT_PRECISION,
            bool(query_centroid_norm_window),
            bool(use_native_temporal_seeds),
            bool(gaussian_partition),
            bool(max_partition),
            bool(split_merge_kv),
            SCALE=float(scale),
            num_warps=merge_num_warps,
            num_stages=merge_num_stages,
        )

    route_grid = (1, batch * active_heads, query_outer * COLS)
    route_loop_stages = (
        1 if torch.version.hip is not None else 3
    ) if route_loop_stages is None else int(route_loop_stages)
    _hard_c56_route_kernel[route_grid](
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
        head_major,
        ROWS,
        COLS,
        c,
        ROUTE_BLOCK_I,
        route_block_tokens,
        route_loop_stages,
        INPUT_PRECISION,
        bool(return_log_partition),
        bool(compressed_key_fp8),
        SCALE=float(scale),
        num_warps=route_num_warps,
        num_stages=route_num_stages,
    )
    if return_log_partition:
        return output, log_partition
    return output


def hard_c_720p_native_temporal_seed_rows_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    c,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run native-layout Hard C4/C6/C8 from shared external seed rows."""

    c = int(c)
    if c not in {4, 6, 8}:
        raise ValueError("external Hellinger seeds support C=4, C=6, or C=8")
    return _hard_c_attention(
        q,
        k,
        v,
        c,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        native_temporal_seed_rows=seed_rows,
    )


def hard_c3_720p_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    return_log_partition=False,
):
    """Run symmetric local-seed Hard C3 on native Wan Q/K/V blocks."""

    return _hard_c_attention(
        q,
        k,
        v,
        3,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        return_log_partition=return_log_partition,
    )


def hard_c5_720p_attention_triton_head_major(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run hard C5 on Diffusers ``[B, H, 21*45*80, 128]`` Q/K/V."""

    return _hard_c_attention(
        q, k, v, 5, scale, output, head_start, head_count
    )


def hard_c5_720p_query_centroid_norm_attention_triton_head_major(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run Hard C5 with one P11 norm-preserving local Lloyd update."""

    return _hard_c_attention(
        q,
        k,
        v,
        5,
        scale,
        output,
        head_start,
        head_count,
        query_centroid_norm_window=CENTROID_CANDIDATES,
    )


def hard_c6_720p_attention_triton_head_major(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run hard C6 on Diffusers ``[B, H, 21*45*80, 128]`` Q/K/V."""

    return _hard_c_attention(
        q, k, v, 6, scale, output, head_start, head_count
    )


def hard_c8_720p_attention_triton_head_major(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run hard C8 on Diffusers ``[B, H, 21*45*80, 128]`` Q/K/V."""

    return _hard_c_attention(
        q, k, v, 8, scale, output, head_start, head_count
    )


def hard_c6_720p_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run hard C6 on native Wan ``[B, 21, 45, 80, H, 128]`` Q/K/V."""

    return _hard_c_attention(
        q, k, v, 6, scale, output, head_start, head_count, head_major=False
    )


def hard_c6_720p_compact_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed-seed Hard C6 with a compact four-plus-two merge kernel."""

    return _hard_c_attention(
        q,
        k,
        v,
        6,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        compact_c6=True,
    )


def hard_c6_720p_fp8_key_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C6 with native FP8 storage and QK for merged keys."""

    return _hard_c_attention(
        q,
        k,
        v,
        6,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        compressed_key_fp8=True,
    )


def hard_c6_720p_gaussian_partition_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C6 with a moment approximation to seed log-partitions."""

    return _hard_c_attention(
        q,
        k,
        v,
        6,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        gaussian_partition=True,
    )


def hard_c6_720p_split_kv_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C6 with sequential K and V merge accumulators."""

    return _hard_c_attention(
        q,
        k,
        v,
        6,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        split_merge_kv=True,
    )


def hard_c6_720p_max_partition_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C6 with max-normalized seed assignment."""

    return _hard_c_attention(
        q,
        k,
        v,
        6,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        max_partition=True,
    )


def hard_c5_720p_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed-seed Hard C5 on native Wan Q/K/V blocks."""

    return _hard_c_attention(
        q, k, v, 5, scale, output, head_start, head_count, head_major=False
    )


def hard_c5_720p_temporal_shared_seed_rows_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run compact Hard C5 from one to three temporally shared seed tables."""

    return _hard_c_attention(
        q,
        k,
        v,
        5,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        temporal_shared_seed_rows=seed_rows,
    )


def hard_c5_720p_query_centroid_norm_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run P11 norm-centroid Hard C5 on native Wan Q/K/V blocks."""

    return _hard_c_attention(
        q,
        k,
        v,
        5,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        query_centroid_norm_window=CENTROID_CANDIDATES,
    )


def hard_c8_720p_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run hard C8 on native Wan ``[B, 21, 45, 80, H, 128]`` Q/K/V."""

    return _hard_c_attention(
        q, k, v, 8, scale, output, head_start, head_count, head_major=False
    )


def hard_c8_720p_fp8_key_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C8 with native FP8 storage and QK for merged keys."""

    return _hard_c_attention(
        q,
        k,
        v,
        8,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        compressed_key_fp8=True,
    )


def hard_c8_720p_gaussian_partition_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C8 with a moment approximation to seed log-partitions."""

    return _hard_c_attention(
        q,
        k,
        v,
        8,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        gaussian_partition=True,
    )


def hard_c8_720p_split_kv_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C8 with sequential K and V merge accumulators."""

    return _hard_c_attention(
        q,
        k,
        v,
        8,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        split_merge_kv=True,
    )


def hard_c8_720p_max_partition_attention_triton_blocks(
    q, k, v, scale=None, output=None, head_start=0, head_count=None
):
    """Run fixed Hard C8 with max-normalized seed assignment."""

    return _hard_c_attention(
        q,
        k,
        v,
        8,
        scale,
        output,
        head_start,
        head_count,
        head_major=False,
        max_partition=True,
    )


__all__ = [
    "hard_c_720p_native_temporal_seed_rows_attention_triton_blocks",
    "hard_c3_720p_attention_triton_blocks",
    "hard_c5_720p_attention_triton_head_major",
    "hard_c5_720p_query_centroid_norm_attention_triton_head_major",
    "hard_c6_720p_attention_triton_head_major",
    "hard_c8_720p_attention_triton_head_major",
    "hard_c5_720p_attention_triton_blocks",
    "hard_c5_720p_temporal_shared_seed_rows_attention_triton_blocks",
    "hard_c5_720p_query_centroid_norm_attention_triton_blocks",
    "hard_c6_720p_attention_triton_blocks",
    "hard_c6_720p_compact_attention_triton_blocks",
    "hard_c6_720p_fp8_key_attention_triton_blocks",
    "hard_c6_720p_gaussian_partition_attention_triton_blocks",
    "hard_c6_720p_split_kv_attention_triton_blocks",
    "hard_c6_720p_max_partition_attention_triton_blocks",
    "hard_c8_720p_attention_triton_blocks",
    "hard_c8_720p_fp8_key_attention_triton_blocks",
    "hard_c8_720p_gaussian_partition_attention_triton_blocks",
    "hard_c8_720p_split_kv_attention_triton_blocks",
    "hard_c8_720p_max_partition_attention_triton_blocks",
]
