"""Exact hard-C4 for Wan2.1-T2V-14B at 720p (21 x 45 x 80 tokens).

Validated configs:
``configs/token_merging_wan_14b_hard_c4_triton_720p_fewstep.yaml`` and
``configs/token_merging_wan_14b_hard_c4_triton_720p_50step.yaml``.

The merge kernel reuses each 80-token K/V row across all 21 query frames and
stores route-contiguous compressed K/V. The route kernel then performs full
attention over all 21 x 45 x 4 compressed tokens.

For an interior key row ``i``, the four local seed query rows are
``[i - 2, i, i + 1, i + 3]``. Boundary rows retain the original contiguous
window so every block still includes the index-matched query ``q_i``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


HARD_720_ROWS = 45
HARD_720_COLS = 80
HARD_720_C = 4
HARD_720_QUERY_OUTER_GROUP = 21

HARD_720_MERGE_BLOCK_J = 16
HARD_720_MERGE_BLOCK_L = 128
HARD_720_MERGE_NUM_WARPS = 4
HARD_720_MERGE_NUM_STAGES = 1
HARD_720_MERGE_LOOP_STAGES = 2
HARD_720_MERGE_DISABLE_LICM = True

HARD_720_ROUTE_BLOCK_I = 64
HARD_720_ROUTE_BLOCK_TOKENS = 64
HARD_720_ROUTE_NUM_WARPS = 4
HARD_720_ROUTE_NUM_STAGES = 1
HARD_720_INPUT_PRECISION = "ieee"
HARD_720_CENTROID_CANDIDATES = 11
HARD_720_CENTROID_BLOCK_ROWS = 16
HARD_720_CENTROID_NUM_WARPS = 4
HARD_720_CENTROID_NUM_STAGES = 1
HARD_720_RESPONSIBILITY_TOP_R = 2
HARD_720_RESPONSIBILITY_BLOCK_TOKENS = 32
HARD_720_RESPONSIBILITY_NUM_WARPS = 4
HARD_720_RESPONSIBILITY_NUM_STAGES = 1
HARD_720_DISTORTION_BLOCK_CLUSTERS = 64
HARD_720_RESPONSE_CANDIDATES = 15
HARD_720_RESPONSE_PROXIES = 8
HARD_720_RESPONSE_BLOCK_J = 16
HARD_720_RESPONSE_BLOCK_ROWS = 16
HARD_720_RESPONSE_BLOCK_PROXIES = 16
HARD_720_RESPONSE_NUM_WARPS = 4
HARD_720_RESPONSE_NUM_STAGES = 2


if triton is not None:

    @triton.jit
    def _hard_c4_720p_query_response_seed_kernel(
        q_ptr,
        k_ptr,
        seed_row_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        CANDIDATES: tl.constexpr,
        PROXIES: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_PROXIES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Select center, left, and two response-span query rows per block."""

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)

        query_col_block = block_id % tl.cdiv(COLS, BLOCK_J)
        block_id //= tl.cdiv(COLS, BLOCK_J)
        key_row = block_id % ROWS
        block_id //= ROWS
        key_outer = block_id % K_OUTER
        query_outer = block_id // K_OUTER

        flat_rows = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        query_slots = flat_rows // BLOCK_ROWS
        candidate_slots = flat_rows % BLOCK_ROWS
        query_cols = query_col_block * BLOCK_J + query_slots
        candidate_start = tl.minimum(
            tl.maximum(key_row - CANDIDATES // 2, 0),
            ROWS - CANDIDATES,
        )
        candidate_rows = candidate_start + candidate_slots
        candidate_mask = (
            (query_cols < COLS) & (candidate_slots < CANDIDATES)
        )
        dims = tl.arange(0, 128)
        q_base = (
            (batch * Q_OUTER + query_outer)
            * ROWS
            * COLS
            * TOTAL_HEADS
            * D_HEAD
        )
        query_offsets = (
            q_base
            + (
                (candidate_rows[:, None] * COLS + query_cols[:, None])
                * TOTAL_HEADS
                + global_head
            )
            * D_HEAD
            + dims[None, :]
        )
        candidates = tl.load(
            q_ptr + query_offsets,
            mask=candidate_mask[:, None],
            other=0.0,
        )

        proxy_slots = tl.arange(0, BLOCK_PROXIES)
        proxy_cols = tl.where(
            proxy_slots == 0,
            0,
            tl.where(
                proxy_slots == 1,
                11,
                tl.where(
                    proxy_slots == 2,
                    23,
                    tl.where(
                        proxy_slots == 3,
                        34,
                        tl.where(
                            proxy_slots == 4,
                            45,
                            tl.where(
                                proxy_slots == 5,
                                56,
                                tl.where(proxy_slots == 6, 68, 79),
                            ),
                        ),
                    ),
                ),
            ),
        )
        proxy_mask = proxy_slots < PROXIES
        k_base = (
            (batch * K_OUTER + key_outer)
            * ROWS
            * COLS
            * TOTAL_HEADS
            * D_HEAD
            + key_row * COLS * TOTAL_HEADS * D_HEAD
        )
        key_offsets = (
            k_base
            + (proxy_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
            + dims[None, :]
        )
        proxy_keys = tl.load(
            k_ptr + key_offsets,
            mask=proxy_mask[:, None],
            other=0.0,
        )

        response_logits = tl.dot(
            candidates,
            tl.trans(proxy_keys),
            input_precision=INPUT_PRECISION,
        ) * (SCALE * 1.4426950408889634)
        response_logits = tl.reshape(
            response_logits,
            (BLOCK_J, BLOCK_ROWS, BLOCK_PROXIES),
        )
        response_logits = tl.where(
            proxy_mask[None, None, :],
            response_logits,
            -float("inf"),
        )
        response_max = tl.max(response_logits, axis=2)
        response_probability = tl.exp2(
            response_logits - response_max[:, :, None]
        )
        response_probability = tl.where(
            proxy_mask[None, None, :], response_probability, 0.0
        )
        response_probability /= tl.sum(
            response_probability, axis=2
        )[:, :, None]
        residual = tl.where(
            proxy_mask[None, None, :],
            response_probability - 1.0 / PROXIES,
            0.0,
        )

        candidate_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        center_index = key_row - candidate_start
        left_row = tl.where(key_row > 0, key_row - 1, key_row + 1)
        left_index = left_row - candidate_start
        used = (candidate_ids == center_index) | (candidate_ids == left_index)

        center_vector = tl.sum(
            tl.where(candidate_ids[:, :, None] == center_index, residual, 0.0),
            axis=1,
        )
        center_norm = tl.sqrt(
            tl.sum(center_vector * center_vector, axis=1) + 1.0e-12
        )
        center_basis = center_vector / center_norm[:, None]
        center_projection = tl.sum(
            residual * center_basis[:, None, :], axis=2
        )
        residual -= center_projection[:, :, None] * center_basis[:, None, :]

        left_vector = tl.sum(
            tl.where(candidate_ids[:, :, None] == left_index, residual, 0.0),
            axis=1,
        )
        left_norm = tl.sqrt(
            tl.sum(left_vector * left_vector, axis=1) + 1.0e-12
        )
        left_basis = left_vector / left_norm[:, None]
        left_projection = tl.sum(
            residual * left_basis[:, None, :], axis=2
        )
        residual -= left_projection[:, :, None] * left_basis[:, None, :]

        valid_candidates = (
            tl.arange(0, BLOCK_ROWS)[None, :] < CANDIDATES
        )
        objective = tl.sum(residual * residual, axis=2)
        objective = tl.where(
            valid_candidates & ~used, objective, -1.0
        )
        _, third_index = tl.max(
            objective,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        used |= candidate_ids == third_index[:, None]
        third_vector = tl.sum(
            tl.where(
                candidate_ids[:, :, None] == third_index[:, None, None],
                residual,
                0.0,
            ),
            axis=1,
        )
        third_norm = tl.sqrt(
            tl.sum(third_vector * third_vector, axis=1) + 1.0e-12
        )
        third_basis = third_vector / third_norm[:, None]
        third_projection = tl.sum(
            residual * third_basis[:, None, :], axis=2
        )
        residual -= third_projection[:, :, None] * third_basis[:, None, :]

        objective = tl.sum(residual * residual, axis=2)
        objective = tl.where(
            valid_candidates & ~used, objective, -1.0
        )
        _, fourth_index = tl.max(
            objective,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        output_query_cols = query_col_block * BLOCK_J + tl.arange(0, BLOCK_J)
        output_mask = output_query_cols < COLS
        seed_base = (
            ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                  + output_query_cols)
                 * ACTIVE_HEADS
                 + head)
                * K_OUTER
                + key_outer)
               * ROWS
               + key_row)
              * C)
        )
        tl.store(
            seed_row_ptr + seed_base,
            center_index + candidate_start,
            mask=output_mask,
        )
        tl.store(
            seed_row_ptr + seed_base + 1,
            left_index + candidate_start,
            mask=output_mask,
        )
        tl.store(
            seed_row_ptr + seed_base + 2,
            third_index + candidate_start,
            mask=output_mask,
        )
        tl.store(
            seed_row_ptr + seed_base + 3,
            fourth_index + candidate_start,
            mask=output_mask,
        )

    @triton.jit
    def _hard_c4_720p_query_centroid_kernel(
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
        """Run one local Lloyd update from the maintained wide C4 seeds."""

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

        seed_start = tl.minimum(tl.maximum(key_row - 1, 0), ROWS - C)
        seed_rows = seed_start + seed_slot
        wide_offset = tl.where(
            seed_slot == 0,
            -2,
            tl.where(seed_slot == 1, 0, tl.where(seed_slot == 2, 1, 3)),
        )
        use_wide_seeds = (key_row >= 2) & (key_row <= ROWS - 4)
        seed_rows = tl.where(use_wide_seeds, key_row + wide_offset, seed_rows)

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
                    (candidate_rows[:, None] * COLS + query_col)
                    * TOTAL_HEADS
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
        candidate_norm = tl.sum(candidates * candidates, axis=1)
        center_norm = tl.sum(initial_centers * initial_centers, axis=1)
        distance = (
            candidate_norm[:, None]
            + center_norm[None, :]
            - 2.0
            * tl.dot(
                candidates,
                tl.trans(initial_centers),
                input_precision=INPUT_PRECISION,
            )
        )
        distance = tl.where(valid_seed[None, :], distance, float("inf"))
        _, assignment = tl.min(
            distance,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        membership = (
            (assignment[:, None] == seed_slot[None, :])
            & valid_candidate[:, None]
            & valid_seed[None, :]
        )
        count = tl.sum(membership.to(tl.float32), axis=0)
        safe_count = tl.where(count > 0.0, count, 1.0)
        weights = membership.to(tl.float32) / safe_count[None, :]
        centroids = tl.dot(
            tl.trans(weights).to(candidates.dtype),
            candidates,
            input_precision=INPUT_PRECISION,
        )
        centroids = tl.where(
            (count > 0.0)[:, None], centroids, initial_centers
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
    def _temporal_compact_refine_rows(
        q_ptr,
        coarse_seed_row_ptr,
        probe_key,
        batch_i64,
        query_outer,
        key_outer,
        key_row,
        global_head,
        query_col_start,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        ALPHA: tl.constexpr,
        SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        """Return 8xC refined rows without materializing a temporal table."""

        BLOCK_COLUMNS: tl.constexpr = 8
        CANDIDATES: tl.constexpr = 8
        BLOCK_PROBES: tl.constexpr = 16
        candidate_flat = tl.arange(0, BLOCK_COLUMNS * CANDIDATES)
        column_slot = candidate_flat // CANDIDATES
        candidate_slot = candidate_flat % CANDIDATES
        query_columns = query_col_start + column_slot

        # Eight candidates: the aligned row, a radius-two neighborhood around
        # the strongest coarse dynamic row, and the other two coarse rows.
        coarse_seed_slot = tl.where(
            candidate_slot <= 5,
            1,
            candidate_slot - 4,
        )
        neighbor_offset = tl.where(
            candidate_slot <= 5,
            candidate_slot - 3,
            0,
        )
        coarse_base = (
            (((((batch_i64 * K_OUTER + key_outer) * ROWS + key_row) * COLS
                + query_columns)
               * TOTAL_HEADS
               + global_head)
              * C)
        )
        coarse_row = tl.load(
            coarse_seed_row_ptr + coarse_base + coarse_seed_slot,
            mask=candidate_slot > 0,
            other=key_row,
        ).to(tl.int32)
        candidate_row = tl.where(
            candidate_slot == 0,
            key_row,
            tl.minimum(tl.maximum(coarse_row + neighbor_offset, 0), ROWS - 1),
        )

        dims = tl.arange(0, 128)
        q_base = (
            (batch_i64 * Q_OUTER + query_outer)
            * ROWS
            * COLS
            * TOTAL_HEADS
            * D_HEAD
        )
        candidate_q = tl.load(
            q_ptr
            + q_base
            + (
                (candidate_row[:, None] * COLS + query_columns[:, None])
                * TOTAL_HEADS
                + global_head
            )
            * D_HEAD
            + dims[None, :],
        )

        probe_slot = tl.arange(0, BLOCK_PROBES)
        probe_valid = probe_slot < 8
        logits = tl.dot(
            candidate_q,
            tl.trans(probe_key),
            input_precision=INPUT_PRECISION,
        ) * (SCALE * 1.4426950408889634)
        logits = tl.where(probe_valid[None, :], logits, -float("inf"))
        row_max = tl.max(logits, axis=1)
        exponential = tl.exp2(logits - row_max[:, None])
        denominator = tl.sum(exponential, axis=1)
        log_mass = row_max + tl.log2(denominator)
        probability = exponential / denominator[:, None]
        profile = tl.where(
            probe_valid[None, :], tl.sqrt(probability), 0.0
        )
        profile = tl.reshape(
            profile, (BLOCK_COLUMNS, CANDIDATES, BLOCK_PROBES)
        ).to(tl.float32)
        profile_mean = tl.sum(profile, axis=2) * 0.125
        residual = tl.where(
            probe_valid[None, None, :],
            profile - profile_mean[:, :, None],
            0.0,
        )

        candidate_rows = tl.reshape(
            candidate_row, (BLOCK_COLUMNS, CANDIDATES)
        )
        candidate_ids = tl.arange(0, CANDIDATES)[None, :]
        valid = tl.full((BLOCK_COLUMNS, CANDIDATES), 1, tl.int1)
        for previous in tl.static_range(0, CANDIDATES):
            previous_row = tl.sum(
                tl.where(candidate_ids == previous, candidate_rows, 0), axis=1
            )
            valid = valid & (
                (candidate_ids <= previous)
                | (candidate_rows != previous_row[:, None])
            )

        mass = tl.reshape(log_mass, (BLOCK_COLUMNS, CANDIDATES))
        valid_count = tl.sum(valid.to(tl.float32), axis=1)
        mean = tl.sum(tl.where(valid, mass, 0.0), axis=1) / valid_count
        deviation = mass - mean[:, None]
        variance = (
            tl.sum(tl.where(valid, deviation * deviation, 0.0), axis=1)
            / valid_count
        )
        weight = tl.exp2(
            (2.0 * ALPHA * 1.4426950408889634)
            * deviation
            * tl.rsqrt(variance[:, None] + 1e-6)
        )

        used = candidate_rows == key_row
        basis_source = tl.sum(
            tl.where(candidate_ids[:, :, None] == 0, residual, 0.0), axis=1
        )
        pick1 = tl.full((BLOCK_COLUMNS,), key_row, tl.int32)
        pick2 = tl.full((BLOCK_COLUMNS,), key_row, tl.int32)
        pick3 = tl.full((BLOCK_COLUMNS,), key_row, tl.int32)
        for step in tl.static_range(0, 3):
            basis = basis_source * tl.rsqrt(
                tl.sum(basis_source * basis_source, axis=1) + 1e-12
            )[:, None]
            projection = tl.sum(residual * basis[:, None, :], axis=2)
            residual = residual - projection[:, :, None] * basis[:, None, :]
            objective = tl.sum(residual * residual, axis=2) * weight
            objective = tl.where(valid & ~used, objective, -1.0)
            _, position = tl.max(
                objective,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            hit = candidate_ids == position[:, None]
            picked_row = tl.sum(tl.where(hit, candidate_rows, 0), axis=1)
            used = used | (candidate_rows == picked_row[:, None])
            basis_source = tl.sum(
                tl.where(hit[:, :, None], residual, 0.0), axis=1
            )
            pick1 = tl.where(step == 0, picked_row, pick1)
            pick2 = tl.where(step == 1, picked_row, pick2)
            pick3 = tl.where(step == 2, picked_row, pick3)

        local_row = tl.maximum(key_row - 1, 0)
        dynamic0 = tl.where(pick1 != local_row, pick1, pick2)
        dynamic1 = tl.where(
            pick1 == local_row,
            pick3,
            tl.where(pick2 != local_row, pick2, pick3),
        )
        seed_slot = tl.arange(0, C)[None, :]
        local_seeds = tl.where(
            seed_slot == 0,
            key_row,
            tl.where(
                seed_slot == 1,
                local_row,
                tl.where(seed_slot == 2, dynamic0[:, None], dynamic1[:, None]),
            ),
        )
        boundary_seeds = tl.where(
            seed_slot == 0,
            key_row,
            tl.where(
                seed_slot == 1,
                pick1[:, None],
                tl.where(seed_slot == 2, pick2[:, None], pick3[:, None]),
            ),
        )
        return tl.where(key_row > 0, local_seeds, boundary_seeds).to(tl.int32)

    @triton.jit
    def _hard_c4_720p_merge_kernel(
        q_ptr,
        centroid_ptr,
        response_seed_row_ptr,
        temporal_seed_row_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        assignment_ptr,
        Q_OUTER: tl.constexpr,
        Q_GROUPS: tl.constexpr,
        Q_GROUP: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        BLOCK_J: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        FLATTEN_LOOPS: tl.constexpr,
        DISABLE_LICM: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        USE_QUERY_CENTROIDS: tl.constexpr,
        USE_RESPONSE_SEEDS: tl.constexpr,
        RESPONSE_SEEDS_NATIVE: tl.constexpr,
        RESPONSE_SEEDS_SHARED: tl.constexpr,
        RESPONSE_SEED_FRAMES: tl.constexpr,
        USE_TEMPORAL_COMPACT_REFINE: tl.constexpr,
        USE_ENTROPY_BIAS: tl.constexpr,
        STORE_ASSIGNMENT: tl.constexpr,
        EXACT_NEIGHBOR_WINDOW_COLS: tl.constexpr,
        QUERY_OUTER_OFFSET: tl.constexpr,
        ROUTE_BIAS_SCALE: tl.constexpr,
        TEMPORAL_ALPHA: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Compress one key row for all query frames and one active head."""

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

        flat_rows = tl.arange(0, BLOCK_J * C)
        seed_ids = flat_rows % C
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
            key_tail_offsets = row_base + tail_cols[:, None] * D_HEAD + dims[None, :]
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
            key_tail_offsets = (
                row_base
                + (tail_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
                + dims[None, :]
            )

        key_main = tl.load(
            k_ptr + key_offsets,
            eviction_policy="evict_last",
        )
        key_tail = tl.load(
            k_ptr + key_tail_offsets,
            eviction_policy="evict_last",
        )
        value_main = tl.load(
            v_ptr + key_offsets,
            eviction_policy="evict_last",
        )
        value_tail = tl.load(
            v_ptr + key_tail_offsets,
            eviction_policy="evict_last",
        )
        probe_slots = tl.arange(0, 16)
        probe_mask = probe_slots < 8
        probe_columns = probe_slots * (COLS // 8)
        if HEAD_MAJOR:
            probe_offsets = row_base + probe_columns[:, None] * D_HEAD + dims[None, :]
        else:
            probe_offsets = (
                row_base
                + (probe_columns[:, None] * TOTAL_HEADS + global_head) * D_HEAD
                + dims[None, :]
            )
        probe_key = tl.load(
            k_ptr + probe_offsets,
            mask=probe_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        seed_start = tl.minimum(
            tl.maximum(key_row - 1, 0), ROWS - C
        )
        seed_rows = seed_start + seed_ids
        wide_offset = tl.where(
            seed_ids == 0,
            -2,
            tl.where(seed_ids == 1, 0, tl.where(seed_ids == 2, 1, 3)),
        )
        use_wide_seeds = (key_row >= 2) & (key_row <= ROWS - 4)
        seed_rows = tl.where(use_wide_seeds, key_row + wide_offset, seed_rows)

        for q_offset in tl.range(
            0,
            Q_GROUP,
            num_stages=LOOP_STAGES,
            flatten=FLATTEN_LOOPS,
            disable_licm=DISABLE_LICM,
        ):
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

            for query_col_start in tl.range(
                0, COLS, BLOCK_J
            ):
                query_cols = query_col_start + flat_rows // C
                if USE_QUERY_CENTROIDS:
                    centroid_base = (
                        (((((batch_i64 * Q_OUTER + query_outer) * ROWS
                            + key_row)
                           * COLS
                           + query_cols[:, None])
                          * ACTIVE_HEADS
                          + head)
                         * C)
                        + seed_ids[:, None]
                    )
                    seed_query_offsets = (
                        centroid_base * D_HEAD + dims[None, :]
                    )
                    seed_query = tl.load(
                        centroid_ptr + seed_query_offsets,
                        eviction_policy="evict_last",
                    )
                else:
                    active_seed_rows = seed_rows
                    if USE_TEMPORAL_COMPACT_REFINE:
                        first_rows = _temporal_compact_refine_rows(
                            q_ptr,
                            temporal_seed_row_ptr,
                            probe_key,
                            batch_i64,
                            query_outer,
                            key_outer,
                            key_row,
                            global_head,
                            query_col_start,
                            Q_OUTER,
                            K_OUTER,
                            TOTAL_HEADS,
                            D_HEAD,
                            ROWS,
                            COLS,
                            C,
                            TEMPORAL_ALPHA,
                            SCALE,
                            INPUT_PRECISION,
                        )
                        second_rows = _temporal_compact_refine_rows(
                            q_ptr,
                            temporal_seed_row_ptr,
                            probe_key,
                            batch_i64,
                            query_outer,
                            key_outer,
                            key_row,
                            global_head,
                            query_col_start + 8,
                            Q_OUTER,
                            K_OUTER,
                            TOTAL_HEADS,
                            D_HEAD,
                            ROWS,
                            COLS,
                            C,
                            TEMPORAL_ALPHA,
                            SCALE,
                            INPUT_PRECISION,
                        )
                        active_seed_rows = tl.reshape(
                            tl.permute(
                                tl.join(first_rows, second_rows),
                                (2, 0, 1),
                            ),
                            (64,),
                        )
                    elif USE_RESPONSE_SEEDS:
                        if RESPONSE_SEEDS_SHARED:
                            response_seed_base = batch_i64 * COLS + query_cols
                            response_seed_base = (
                                response_seed_base * ACTIVE_HEADS + head
                            )
                            response_seed_base = (
                                response_seed_base * K_OUTER + key_outer
                            )
                            response_seed_base = response_seed_base * ROWS + key_row
                            response_seed_base = response_seed_base * C + seed_ids
                        elif RESPONSE_SEEDS_NATIVE:
                            response_query_outer = tl.where(
                                RESPONSE_SEED_FRAMES == 1, 0, query_outer
                            )
                            response_seed_base = (
                                batch_i64 * RESPONSE_SEED_FRAMES
                                + response_query_outer
                            )
                            response_seed_base = (
                                response_seed_base * K_OUTER + key_outer
                            )
                            response_seed_base = response_seed_base * ROWS + key_row
                            response_seed_base = response_seed_base * COLS + query_cols
                            response_seed_base = (
                                response_seed_base * TOTAL_HEADS + global_head
                            )
                            response_seed_base = response_seed_base * C + seed_ids
                        else:
                            response_seed_base = (
                                ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                                      + query_cols)
                                     * ACTIVE_HEADS
                                     + head)
                                    * K_OUTER
                                    + key_outer)
                                   * ROWS
                                   + key_row)
                                  * C)
                                + seed_ids
                            )
                        active_seed_rows = tl.load(
                            response_seed_row_ptr + response_seed_base
                        ).to(tl.int32)
                    if HEAD_MAJOR:
                        seed_query_offsets = (
                            q_base
                            + (
                                active_seed_rows[:, None] * COLS
                                + query_cols[:, None]
                            )
                            * D_HEAD
                            + dims[None, :]
                        )
                    else:
                        seed_query_offsets = (
                            q_base
                            + (
                                (
                                    active_seed_rows[:, None] * COLS
                                    + query_cols[:, None]
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

                if EXACT_NEIGHBOR_WINDOW_COLS:
                    global_query_outer = QUERY_OUTER_OFFSET + query_outer
                    neighbor_0 = tl.where(
                        global_query_outer == 0,
                        1,
                        global_query_outer - 1,
                    )
                    neighbor_1 = tl.where(
                        global_query_outer == 0,
                        2,
                        tl.where(
                            global_query_outer == K_OUTER - 1,
                            global_query_outer - 2,
                            global_query_outer + 1,
                        ),
                    )
                    is_exact_neighbor = (key_outer == neighbor_0) | (
                        key_outer == neighbor_1
                    )
                    window_start = tl.minimum(
                        tl.maximum(
                            query_cols - EXACT_NEIGHBOR_WINDOW_COLS // 2,
                            0,
                        ),
                        COLS - EXACT_NEIGHBOR_WINDOW_COLS,
                    )
                    exact_main = is_exact_neighbor & (
                        main_cols[None, :] >= window_start[:, None]
                    ) & (
                        main_cols[None, :]
                        < window_start[:, None] + EXACT_NEIGHBOR_WINDOW_COLS
                    )
                    exact_tail = is_exact_neighbor & (
                        tail_cols[None, :] >= window_start[:, None]
                    ) & (
                        tail_cols[None, :]
                        < window_start[:, None] + EXACT_NEIGHBOR_WINDOW_COLS
                    )
                    scores_main = tl.where(
                        exact_main, -float("inf"), scores_main
                    )
                    scores_tail = tl.where(
                        exact_tail, -float("inf"), scores_tail
                    )

                row_max = tl.maximum(
                    tl.max(scores_main, axis=1), tl.max(scores_tail, axis=1)
                )
                row_sum = tl.sum(tl.exp2(scores_main - row_max[:, None]), axis=1)
                row_sum += tl.sum(tl.exp2(scores_tail - row_max[:, None]), axis=1)
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
                if STORE_ASSIGNMENT:
                    assignment_query_cols = query_col_start + tl.arange(
                        0, BLOCK_J
                    )
                    assignment_base = (
                        ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                             + assignment_query_cols[:, None])
                            * ACTIVE_HEADS
                            + head)
                           * K_OUTER
                           + key_outer)
                          * ROWS
                          + key_row)
                         * COLS)
                    )
                    tl.store(
                        assignment_ptr
                        + assignment_base
                        + main_cols[None, :],
                        route_main.to(tl.uint8),
                    )
                    tl.store(
                        assignment_ptr
                        + assignment_base
                        + tail_cols[None, :],
                        route_tail.to(tl.uint8),
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
                prob_sum = tl.sum(probs_main, axis=2) + tl.sum(probs_tail, axis=2)
                has_key = prob_sum > 0.0
                safe_sum = tl.where(has_key, prob_sum, 1.0)
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
                    route_bias = (
                        expected * inverse_sum - tl.log2(safe_sum)
                    ) * 0.6931471805599453
                else:
                    route_bias = tl.zeros((BLOCK_J, C), tl.float32)
                route_bias *= ROUTE_BIAS_SCALE
                route_bias = tl.where(has_key, route_bias, float("inf"))

                weights_main = tl.reshape(weights_main, (BLOCK_J * C, 64))
                weights_tail = tl.reshape(weights_tail, (BLOCK_J * C, 16))
                key_value_main = tl.interleave(key_main, value_main)
                key_value_tail = tl.interleave(key_tail, value_tail)
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
                    + seed_ids[:, None]
                )
                merged_output = merged_base * D_HEAD + dims[None, :]
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

                query_col_ids = query_col_start + tl.arange(
                    0, BLOCK_J
                )
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
                     * C)
                    + bias_seed[None, :]
                )
                tl.store(
                    bias_ptr + bias_base,
                    route_bias,
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _hard_c4_720p_temporal_shared_merge_kernel(
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
            tl.static_assert(not STORE_PACKED_SINGLETON_KV)
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

        for query_col_start in tl.range(0, COLS, BLOCK_J):
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
                0,
                Q_OUTER,
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
                        merged_key /= residual_scale[:, None]
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
                        merged_value /= residual_scale[:, None]
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
                        merged_key /= residual_scale[:, None]
                        merged_value /= residual_scale[:, None]
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

    @triton.jit
    def _hard_c4_720p_query_column_grouped_merge_kernel(
        q_ptr,
        response_seed_row_ptr,
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
        QUERY_FRAME_GROUP: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Exact query-column R1 merge with query-frame parallelism."""

        tl.static_assert(C == 4)
        tl.static_assert(COLS == 80)
        tl.static_assert(D_HEAD == 128)
        tl.static_assert(Q_OUTER % QUERY_FRAME_GROUP == 0)

        query_groups = Q_OUTER // QUERY_FRAME_GROUP
        block_id = tl.program_id(0)
        query_group = block_id % query_groups
        key_block = block_id // query_groups
        key_outer = key_block % K_OUTER
        key_row = key_block // K_OUTER
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)

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

        flat_rows = tl.arange(0, BLOCK_J * C)
        seed_ids = flat_rows % C
        cluster_ids = tl.arange(0, C)[None, :, None]
        ln2 = 0.6931471805599453

        for query_col_start in tl.range(0, COLS, BLOCK_J):
            query_cols = query_col_start + flat_rows // C
            seed_base = batch_i64 * COLS + query_cols
            seed_base = seed_base * ACTIVE_HEADS + head
            seed_base = seed_base * K_OUTER + key_outer
            seed_base = seed_base * ROWS + key_row
            seed_base = seed_base * C + seed_ids
            active_seed_rows = tl.load(
                response_seed_row_ptr + seed_base
            ).to(tl.int32)

            for local_frame in tl.static_range(0, QUERY_FRAME_GROUP):
                query_outer = query_group * QUERY_FRAME_GROUP + local_frame
                q_base = (
                    (batch * Q_OUTER + query_outer)
                    * ROWS
                    * COLS
                    * TOTAL_HEADS
                    * D_HEAD
                )
                seed_query_offsets = (
                    q_base
                    + (
                        (active_seed_rows[:, None] * COLS + query_cols[:, None])
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
                    tl.max(scores_main, axis=1),
                    tl.max(scores_tail, axis=1),
                )
                row_sum = tl.sum(
                    tl.exp2(scores_main - row_max[:, None]), axis=1
                )
                row_sum += tl.sum(
                    tl.exp2(scores_tail - row_max[:, None]), axis=1
                )
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
                prob_sum = (
                    tl.sum(probs_main, axis=2)
                    + tl.sum(probs_tail, axis=2)
                )
                has_key = prob_sum > 0.0
                safe_sum = tl.where(has_key, prob_sum, 1.0)
                inverse_sum = 1.0 / safe_sum
                weights_main = probs_main * inverse_sum[:, :, None]
                weights_tail = probs_tail * inverse_sum[:, :, None]

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
                expected_mean = expected * inverse_sum
                route_bias = (expected_mean - tl.log2(safe_sum)) * ln2

                selected_query_cols = query_col_start + tl.arange(0, BLOCK_J)
                selected_main = (
                    main_cols[None, :] == selected_query_cols[:, None]
                )
                selected_tail = (
                    tail_cols[None, :] == selected_query_cols[:, None]
                )
                selected_cluster_weight = tl.sum(
                    tl.where(selected_main[:, None, :], weights_main, 0.0),
                    axis=2,
                )
                selected_cluster_weight += tl.sum(
                    tl.where(selected_tail[:, None, :], weights_tail, 0.0),
                    axis=2,
                )
                residual_mass = 1.0 - selected_cluster_weight
                bulk_has_key = has_key & (residual_mass > 1.0e-6)
                safe_residual = tl.where(bulk_has_key, residual_mass, 1.0)
                selected_w_log_w = (
                    selected_cluster_weight
                    * tl.log2(tl.maximum(selected_cluster_weight, 1.0e-20))
                    * ln2
                )
                route_bias = (
                    (route_bias - selected_w_log_w) / safe_residual
                    - tl.log2(safe_residual) * ln2
                )
                route_bias = tl.where(
                    bulk_has_key, route_bias, float("inf")
                )
                weights_main = tl.where(
                    ~selected_main[:, None, :], weights_main, 0.0
                )
                weights_tail = tl.where(
                    ~selected_tail[:, None, :], weights_tail, 0.0
                )

                merged_pair = tl.dot(
                    tl.reshape(weights_main, (BLOCK_J * C, 64)).to(
                        key_main.dtype
                    ),
                    key_value_main,
                    input_precision=INPUT_PRECISION,
                )
                merged_pair += tl.dot(
                    tl.reshape(weights_tail, (BLOCK_J * C, 16)).to(
                        key_tail.dtype
                    ),
                    key_value_tail,
                    input_precision=INPUT_PRECISION,
                )
                merged_key, merged_value = tl.split(
                    tl.reshape(merged_pair, (BLOCK_J * C, 128, 2))
                )
                residual_flat = tl.reshape(safe_residual, (BLOCK_J * C,))
                merged_key /= residual_flat[:, None]
                merged_value /= residual_flat[:, None]

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
                    + seed_ids[:, None]
                )
                merged_output = merged_base * D_HEAD + dims[None, :]
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
                bias_seed = tl.arange(0, C)
                bias_base = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + selected_query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * C)
                    + bias_seed[None, :]
                )
                tl.store(
                    bias_ptr + bias_base,
                    route_bias,
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _hard_c4_720p_query_column_weight_kernel(
        q_ptr,
        response_seed_row_ptr,
        k_ptr,
        weight_ptr,
        bias_ptr,
        query_outer_index,
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
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Materialize exact query-column R1 bulk weights for one Q frame."""

        tl.static_assert(C == 4)
        tl.static_assert(COLS == 80)
        tl.static_assert(D_HEAD == 128)
        column_blocks: tl.constexpr = COLS // BLOCK_J

        block_id = tl.program_id(0)
        query_col_block = block_id % column_blocks
        key_block = block_id // column_blocks
        key_outer = key_block % K_OUTER
        key_row = key_block // K_OUTER
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)

        main_cols = tl.arange(0, 64)
        tail_cols = 64 + tl.arange(0, 16)
        dims = tl.arange(0, 128)
        flat_rows = tl.arange(0, BLOCK_J * C)
        seed_ids = flat_rows % C
        query_col_start = query_col_block * BLOCK_J
        query_cols = query_col_start + flat_rows // C
        cluster_ids = tl.arange(0, C)[None, :, None]
        ln2 = 0.6931471805599453

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
        key_tail = tl.load(
            k_ptr + key_tail_offsets, eviction_policy="evict_last"
        )

        seed_base = batch_i64 * COLS + query_cols
        seed_base = seed_base * ACTIVE_HEADS + head
        seed_base = seed_base * K_OUTER + key_outer
        seed_base = seed_base * ROWS + key_row
        seed_base = seed_base * C + seed_ids
        active_seed_rows = tl.load(
            response_seed_row_ptr + seed_base
        ).to(tl.int32)
        q_base = (
            (batch * Q_OUTER + query_outer_index)
            * ROWS
            * COLS
            * TOTAL_HEADS
            * D_HEAD
        )
        seed_query_offsets = (
            q_base
            + (
                (active_seed_rows[:, None] * COLS + query_cols[:, None])
                * TOTAL_HEADS
                + global_head
            )
            * D_HEAD
            + dims[None, :]
        )
        seed_query = tl.load(
            q_ptr + seed_query_offsets, eviction_policy="evict_last"
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
        inverse_sum = 1.0 / safe_sum
        weights_main = probs_main * inverse_sum[:, :, None]
        weights_tail = probs_tail * inverse_sum[:, :, None]

        expected = tl.sum(
            probs_main
            * tl.where(
                probs_main > 0.0, winner_main[:, None, :], 0.0
            ),
            axis=2,
        )
        expected += tl.sum(
            probs_tail
            * tl.where(
                probs_tail > 0.0, winner_tail[:, None, :], 0.0
            ),
            axis=2,
        )
        route_bias = (expected * inverse_sum - tl.log2(safe_sum)) * ln2

        selected_query_cols = query_col_start + tl.arange(0, BLOCK_J)
        selected_main = main_cols[None, :] == selected_query_cols[:, None]
        selected_tail = tail_cols[None, :] == selected_query_cols[:, None]
        selected_weight = tl.sum(
            tl.where(selected_main[:, None, :], weights_main, 0.0), axis=2
        )
        selected_weight += tl.sum(
            tl.where(selected_tail[:, None, :], weights_tail, 0.0), axis=2
        )
        residual_mass = 1.0 - selected_weight
        bulk_has_key = has_key & (residual_mass > 1.0e-6)
        safe_residual = tl.where(bulk_has_key, residual_mass, 1.0)
        selected_w_log_w = (
            selected_weight
            * tl.log2(tl.maximum(selected_weight, 1.0e-20))
            * ln2
        )
        route_bias = (
            (route_bias - selected_w_log_w) / safe_residual
            - tl.log2(safe_residual) * ln2
        )
        route_bias = tl.where(bulk_has_key, route_bias, float("inf"))
        weights_main = tl.where(
            ~selected_main[:, None, :], weights_main, 0.0
        ) / safe_residual[:, :, None]
        weights_tail = tl.where(
            ~selected_tail[:, None, :], weights_tail, 0.0
        ) / safe_residual[:, :, None]

        weight_token = (
            (((((batch_i64 * COLS + query_cols[:, None])
                * ACTIVE_HEADS
                + head)
               * K_OUTER
               + key_outer)
              * ROWS
              + key_row)
             * C)
            + seed_ids[:, None]
        )
        tl.store(
            weight_ptr + weight_token * COLS + main_cols[None, :],
            tl.reshape(weights_main, (BLOCK_J * C, 64)),
            eviction_policy="evict_first",
        )
        tl.store(
            weight_ptr + weight_token * COLS + tail_cols[None, :],
            tl.reshape(weights_tail, (BLOCK_J * C, 16)),
            eviction_policy="evict_first",
        )
        bias_seed = tl.arange(0, C)
        bias_base = (
            ((((((batch_i64 * Q_OUTER + query_outer_index) * COLS
                 + selected_query_cols[:, None])
                * ACTIVE_HEADS
                + head)
               * K_OUTER
               + key_outer)
              * ROWS
              + key_row)
             * C)
            + bias_seed[None, :]
        )
        tl.store(
            bias_ptr + bias_base,
            route_bias,
            eviction_policy="evict_first",
        )

    @triton.jit
    def _hard_c4_720p_query_column_weighted_merge_kernel(
        weight_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        query_outer_index,
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
        INPUT_PRECISION: tl.constexpr,
    ):
        """Apply one frame of materialized C4 weights to K and V."""

        tl.static_assert(C == 4)
        tl.static_assert(COLS == 80)
        tl.static_assert(D_HEAD == 128)
        column_blocks: tl.constexpr = COLS // BLOCK_J

        block_id = tl.program_id(0)
        query_col_block = block_id % column_blocks
        key_block = block_id // column_blocks
        key_outer = key_block % K_OUTER
        key_row = key_block // K_OUTER
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)

        query_col_start = query_col_block * BLOCK_J
        flat_rows = tl.arange(0, BLOCK_J * C)
        query_cols = query_col_start + flat_rows // C
        seed_ids = flat_rows % C
        main_cols = tl.arange(0, 64)
        tail_cols = 64 + tl.arange(0, 16)
        dims = tl.arange(0, 128)
        weight_token = (
            (((((batch_i64 * COLS + query_cols[:, None])
                * ACTIVE_HEADS
                + head)
               * K_OUTER
               + key_outer)
              * ROWS
              + key_row)
             * C)
            + seed_ids[:, None]
        )
        weights_main = tl.load(
            weight_ptr + weight_token * COLS + main_cols[None, :],
            eviction_policy="evict_first",
        )
        weights_tail = tl.load(
            weight_ptr + weight_token * COLS + tail_cols[None, :],
            eviction_policy="evict_first",
        )

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
        key_tail = tl.load(
            k_ptr + key_tail_offsets, eviction_policy="evict_last"
        )
        value_main = tl.load(v_ptr + key_offsets, eviction_policy="evict_last")
        value_tail = tl.load(
            v_ptr + key_tail_offsets, eviction_policy="evict_last"
        )
        key_value_main = tl.interleave(key_main, value_main)
        key_value_tail = tl.interleave(key_tail, value_tail)
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
        merged_base = (
            ((((((batch_i64 * Q_OUTER + query_outer_index) * COLS
                 + query_cols[:, None])
                * ACTIVE_HEADS
                + head)
               * K_OUTER
               + key_outer)
              * ROWS
              + key_row)
             * C)
            + seed_ids[:, None]
        )
        merged_output = merged_base * D_HEAD + dims[None, :]
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

    @triton.jit
    def _hard_c4_720p_responsibility_lse_kernel(
        q_ptr,
        merged_k_ptr,
        bias_ptr,
        lse_ptr,
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
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Compute the initial compressed-attention log normalizer."""

        query_group = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        query_outer = query_group // COLS
        query_col = query_group % COLS

        query_rows = tl.arange(0, BLOCK_I)
        dims = tl.arange(0, 128)
        token_offsets = tl.arange(0, BLOCK_TOKENS)
        query_mask = query_rows < ROWS
        q_token = (query_outer * ROWS + query_rows[:, None]) * COLS + query_col
        if HEAD_MAJOR:
            q_address = (
                ((batch * TOTAL_HEADS + global_head) * Q_OUTER * ROWS * COLS
                 + q_token)
                * D_HEAD
                + dims[None, :]
            )
        else:
            q_address = (
                (((batch * Q_OUTER * ROWS * COLS + q_token) * TOTAL_HEADS
                  + global_head)
                 * D_HEAD)
                + dims[None, :]
            )
        query = tl.load(
            q_ptr + q_address,
            mask=query_mask[:, None],
            other=0.0,
        )
        total_tokens = K_OUTER * ROWS * C
        route_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * total_tokens
        )
        row_max = tl.full((BLOCK_I,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_I,), tl.float32)
        for token_start in tl.range(0, total_tokens, BLOCK_TOKENS):
            tokens = token_start + token_offsets
            token_mask = tokens < total_tokens
            merged_key = tl.load(
                merged_k_ptr
                + (route_base + tokens[:, None]) * D_HEAD
                + dims[None, :],
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
                input_precision=INPUT_PRECISION,
            )
            logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
            logits = tl.where(
                query_mask[:, None] & token_mask[None, :],
                logits,
                -float("inf"),
            )
            next_max = tl.maximum(row_max, tl.max(logits, axis=1))
            old_scale = tl.exp2(row_max - next_max)
            row_sum = row_sum * old_scale + tl.sum(
                tl.exp2(logits - next_max[:, None]), axis=1
            )
            row_max = next_max

        lse = row_max + tl.log2(row_sum)
        lse_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * ROWS
        )
        tl.store(
            lse_ptr + lse_base + query_rows,
            lse,
            mask=query_mask,
        )

    @triton.jit
    def _hard_c4_720p_responsibility_centroid_kernel(
        q_ptr,
        merged_k_ptr,
        bias_ptr,
        lse_ptr,
        centroid_ptr,
        selected_id_ptr,
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
        TOP_R: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Solve the responsibility-weighted centroid and select top-G clusters."""

        tl.static_assert(TOP_R == 2)
        token_tile = tl.program_id(0)
        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        query_outer = query_group // COLS
        query_col = query_group % COLS

        query_rows = tl.arange(0, BLOCK_I)
        tokens = token_tile * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        dims = tl.arange(0, 128)
        query_mask = query_rows < ROWS
        total_tokens = K_OUTER * ROWS * C
        token_mask = tokens < total_tokens
        q_token = (query_outer * ROWS + query_rows[:, None]) * COLS + query_col
        if HEAD_MAJOR:
            q_address = (
                ((batch * TOTAL_HEADS + global_head) * Q_OUTER * ROWS * COLS
                 + q_token)
                * D_HEAD
                + dims[None, :]
            )
        else:
            q_address = (
                (((batch * Q_OUTER * ROWS * COLS + q_token) * TOTAL_HEADS
                  + global_head)
                 * D_HEAD)
                + dims[None, :]
            )
        query = tl.load(
            q_ptr + q_address,
            mask=query_mask[:, None],
            other=0.0,
        )

        route_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * total_tokens
        )
        merged_key = tl.load(
            merged_k_ptr
            + (route_base + tokens[:, None]) * D_HEAD
            + dims[None, :],
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
        lse_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * ROWS
        )
        lse = tl.load(
            lse_ptr + lse_base + query_rows,
            mask=query_mask,
            other=float("inf"),
        )
        logits = tl.dot(
            query,
            tl.trans(merged_key),
            input_precision=INPUT_PRECISION,
        )
        logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
        probability = tl.where(
            query_mask[:, None] & token_mask[None, :],
            tl.exp2(logits - lse[:, None]),
            0.0,
        )
        mass = tl.sum(probability, axis=0)
        weighted_query = tl.dot(
            tl.trans(probability.to(query.dtype)),
            query,
            input_precision=INPUT_PRECISION,
        )
        safe_mass = tl.where(mass > 0.0, mass, 1.0)
        centroid = weighted_query / safe_mass[:, None]

        key_rows = (tokens // C) % ROWS
        seed_ids = tokens % C
        seed_start = tl.minimum(tl.maximum(key_rows - 1, 0), ROWS - C)
        seed_rows = seed_start + seed_ids
        wide_offset = tl.where(
            seed_ids == 0,
            -2,
            tl.where(seed_ids == 1, 0, tl.where(seed_ids == 2, 1, 3)),
        )
        use_wide_seeds = (key_rows >= 2) & (key_rows <= ROWS - 4)
        seed_rows = tl.where(use_wide_seeds, key_rows + wide_offset, seed_rows)
        seed_token = (query_outer * ROWS + seed_rows[:, None]) * COLS + query_col
        if HEAD_MAJOR:
            seed_address = (
                ((batch * TOTAL_HEADS + global_head) * Q_OUTER * ROWS * COLS
                 + seed_token)
                * D_HEAD
                + dims[None, :]
            )
        else:
            seed_address = (
                (((batch * Q_OUTER * ROWS * COLS + seed_token) * TOTAL_HEADS
                  + global_head)
                 * D_HEAD)
                + dims[None, :]
            )
        routing_prototype = tl.load(
            q_ptr + seed_address,
            mask=token_mask[:, None],
            other=0.0,
        )
        centroid = tl.where(
            (mass > 0.0)[:, None], centroid, routing_prototype
        )
        tl.store(
            centroid_ptr
            + (route_base + tokens[:, None]) * D_HEAD
            + dims[None, :],
            centroid,
            mask=token_mask[:, None],
            eviction_policy="evict_first",
        )

        gain = mass * tl.sum(
            (centroid - routing_prototype) * (centroid - routing_prototype),
            axis=1,
        )
        gain = tl.where(token_mask, gain, -float("inf"))
        gains_by_block = tl.reshape(
            gain, (BLOCK_TOKENS // C, C)
        )
        _, first_id = tl.max(
            gains_by_block,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        cluster_ids = tl.arange(0, C)[None, :]
        second_scores = tl.where(
            cluster_ids == first_id[:, None],
            -float("inf"),
            gains_by_block,
        )
        _, second_id = tl.max(
            second_scores,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        blocks = (
            token_tile * (BLOCK_TOKENS // C)
            + tl.arange(0, BLOCK_TOKENS // C)
        )
        block_mask = blocks < K_OUTER * ROWS
        selected_base = (
            ((((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
              * ACTIVE_HEADS
              + head)
             * K_OUTER
             * ROWS
             + blocks)
            * TOP_R
        )
        tl.store(
            selected_id_ptr + selected_base,
            first_id.to(tl.int32),
            mask=block_mask,
        )
        tl.store(
            selected_id_ptr + selected_base + 1,
            second_id.to(tl.int32),
            mask=block_mask,
        )

    @triton.jit
    def _hard_c4_720p_refine_selected_kernel(
        centroid_ptr,
        selected_id_ptr,
        assignment_ptr,
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
        HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        TOP_R: tl.constexpr,
        BLOCK_J: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Recompute K/V/entropy moments only for selected fixed-route clusters."""

        tl.static_assert(TOP_R == 2)
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

        selected_rows = tl.arange(0, BLOCK_J * TOP_R)
        selected_rank = selected_rows % TOP_R
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
            key_tail_offsets = row_base + tail_cols[:, None] * D_HEAD + dims[None, :]
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
            key_tail_offsets = (
                row_base
                + (tail_cols[:, None] * TOTAL_HEADS + global_head) * D_HEAD
                + dims[None, :]
            )
        key_main = tl.load(k_ptr + key_offsets, eviction_policy="evict_last")
        key_tail = tl.load(k_ptr + key_tail_offsets, eviction_policy="evict_last")
        value_main = tl.load(v_ptr + key_offsets, eviction_policy="evict_last")
        value_tail = tl.load(v_ptr + key_tail_offsets, eviction_policy="evict_last")

        total_tokens = K_OUTER * ROWS * C
        key_block = key_outer * ROWS + key_row
        for q_offset in tl.range(
            0, Q_GROUP, num_stages=LOOP_STAGES
        ):
            query_outer = query_group * Q_GROUP + q_offset
            for query_col_start in tl.range(0, COLS, BLOCK_J):
                query_cols = query_col_start + selected_rows // TOP_R
                selected_base = (
                    (((((batch_i64 * Q_OUTER + query_outer) * COLS
                        + query_cols)
                       * ACTIVE_HEADS
                       + head)
                      * K_OUTER
                      * ROWS
                      + key_block)
                     * TOP_R)
                    + selected_rank
                )
                selected_ids = tl.load(selected_id_ptr + selected_base)
                centroid_token = key_block * C + selected_ids
                route_base = (
                    (((batch_i64 * Q_OUTER + query_outer) * COLS
                      + query_cols)
                     * ACTIVE_HEADS
                     + head)
                    * total_tokens
                )
                centers = tl.load(
                    centroid_ptr
                    + (route_base[:, None] + centroid_token[:, None]) * D_HEAD
                    + dims[None, :],
                    eviction_policy="evict_last",
                )
                scores_main = tl.dot(
                    centers,
                    tl.trans(key_main),
                    input_precision=INPUT_PRECISION,
                ) * (SCALE * 1.4426950408889634)
                scores_tail = tl.dot(
                    centers,
                    tl.trans(key_tail),
                    input_precision=INPUT_PRECISION,
                ) * (SCALE * 1.4426950408889634)

                assignment_base = (
                    ((((((batch_i64 * Q_OUTER + query_outer) * COLS
                         + query_cols[:, None])
                        * ACTIVE_HEADS
                        + head)
                       * K_OUTER
                       + key_outer)
                      * ROWS
                      + key_row)
                     * COLS)
                )
                route_main = tl.load(
                    assignment_ptr + assignment_base + main_cols[None, :]
                ).to(tl.int32)
                route_tail = tl.load(
                    assignment_ptr + assignment_base + tail_cols[None, :]
                ).to(tl.int32)
                mask_main = route_main == selected_ids[:, None]
                mask_tail = route_tail == selected_ids[:, None]
                has_key = (
                    tl.sum(mask_main.to(tl.int32), axis=1)
                    + tl.sum(mask_tail.to(tl.int32), axis=1)
                ) > 0
                scores_main = tl.where(mask_main, scores_main, -float("inf"))
                scores_tail = tl.where(mask_tail, scores_tail, -float("inf"))
                row_max = tl.maximum(
                    tl.max(scores_main, axis=1), tl.max(scores_tail, axis=1)
                )
                safe_max = tl.where(has_key, row_max, 0.0)
                probs_main = tl.where(
                    mask_main, tl.exp2(scores_main - safe_max[:, None]), 0.0
                )
                probs_tail = tl.where(
                    mask_tail, tl.exp2(scores_tail - safe_max[:, None]), 0.0
                )
                prob_sum = tl.sum(probs_main, axis=1) + tl.sum(probs_tail, axis=1)
                safe_sum = tl.where(has_key, prob_sum, 1.0)
                inverse_sum = 1.0 / safe_sum
                weights_main = probs_main * inverse_sum[:, None]
                weights_tail = probs_tail * inverse_sum[:, None]
                log_norm = safe_max + tl.log2(safe_sum)
                expected_log_probability = tl.sum(
                    weights_main
                    * tl.where(mask_main, scores_main - log_norm[:, None], 0.0),
                    axis=1,
                )
                expected_log_probability += tl.sum(
                    weights_tail
                    * tl.where(mask_tail, scores_tail - log_norm[:, None], 0.0),
                    axis=1,
                )
                route_bias = expected_log_probability * 0.6931471805599453
                route_bias = tl.where(has_key, route_bias, float("inf"))

                key_value_main = tl.interleave(key_main, value_main)
                key_value_tail = tl.interleave(key_tail, value_tail)
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
                    tl.reshape(
                        merged_pair, (BLOCK_J * TOP_R, 128, 2)
                    )
                )
                output_token = key_block * C + selected_ids
                output_base = route_base + output_token
                tl.store(
                    merged_k_ptr
                    + output_base[:, None] * D_HEAD
                    + dims[None, :],
                    merged_key,
                    eviction_policy="evict_first",
                )
                tl.store(
                    merged_v_ptr
                    + output_base[:, None] * D_HEAD
                    + dims[None, :],
                    merged_value,
                    eviction_policy="evict_first",
                )
                tl.store(
                    bias_ptr + output_base,
                    route_bias,
                    eviction_policy="evict_first",
                )

    @triton.jit
    def _hard_c4_720p_prototype_distance_kernel(
        q_ptr,
        distance_ptr,
        Q_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        BLOCK_CLUSTERS: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        """Compute query-to-routing-prototype distances once per C4 cluster."""

        cluster_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        query_outer = query_group // COLS
        query_col = query_group % COLS

        query_rows = tl.arange(0, 64)
        cluster_ids = cluster_block * BLOCK_CLUSTERS + tl.arange(
            0, BLOCK_CLUSTERS
        )
        dims = tl.arange(0, 128)
        query_mask = query_rows < ROWS
        cluster_mask = cluster_ids < ROWS * C
        key_rows = cluster_ids // C
        seed_ids = cluster_ids % C
        seed_start = tl.minimum(tl.maximum(key_rows - 1, 0), ROWS - C)
        seed_rows = seed_start + seed_ids
        wide_offset = tl.where(
            seed_ids == 0,
            -2,
            tl.where(seed_ids == 1, 0, tl.where(seed_ids == 2, 1, 3)),
        )
        use_wide_seeds = (key_rows >= 2) & (key_rows <= ROWS - 4)
        seed_rows = tl.where(use_wide_seeds, key_rows + wide_offset, seed_rows)

        if HEAD_MAJOR:
            q_base = (
                (batch * TOTAL_HEADS + global_head)
                * Q_OUTER
                * ROWS
                * COLS
                * D_HEAD
                + query_outer * ROWS * COLS * D_HEAD
            )
            query_offsets = (
                q_base
                + (query_rows[:, None] * COLS + query_col) * D_HEAD
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
            query_offsets = (
                q_base
                + (
                    (query_rows[:, None] * COLS + query_col) * TOTAL_HEADS
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

        query = tl.load(
            q_ptr + query_offsets,
            mask=query_mask[:, None],
            other=0.0,
        )
        prototype = tl.load(
            q_ptr + seed_offsets,
            mask=cluster_mask[:, None],
            other=0.0,
        )
        query_norm = tl.sum(query * query, axis=1)
        prototype_norm = tl.sum(prototype * prototype, axis=1)
        distance = (
            query_norm[:, None]
            + prototype_norm[None, :]
            - 2.0
            * tl.dot(
                query,
                tl.trans(prototype),
                input_precision=INPUT_PRECISION,
            )
        )
        distance = tl.maximum(distance, 0.0)
        distance_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * ROWS
            * ROWS
            * C
        )
        distance_offsets = (
            distance_base
            + query_rows[:, None] * ROWS * C
            + cluster_ids[None, :]
        )
        tl.store(
            distance_ptr + distance_offsets,
            distance,
            mask=query_mask[:, None] & cluster_mask[None, :],
        )

    @triton.jit
    def _hard_c4_720p_route_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        merged_k_scale_ptr,
        merged_v_scale_ptr,
        bias_ptr,
        prototype_distance_ptr,
        distortion_score_ptr,
        out_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        HEAD_MAJOR: tl.constexpr,
        OUTPUT_HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        EXACT_INTRA_FRAME: tl.constexpr,
        EXACT_NEIGHBOR_WINDOW_COLS: tl.constexpr,
        RECORD_DISTORTION: tl.constexpr,
        QUERY_OUTER_OFFSET: tl.constexpr,
        SINGLETON_TOP_R: tl.constexpr,
        SINGLETON_QUERY_COLUMN: tl.constexpr,
        SINGLETON_SECONDARY_FRAME_STRIDE: tl.constexpr,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2: tl.constexpr,
        QUERY_COLUMN_MAJOR: tl.constexpr,
        PACKED_SINGLETON_KV: tl.constexpr,
        SCALED_FP8_KV: tl.constexpr,
        COMPRESSED_FP8: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Attend to compressed K/V plus optional indexed raw singleton K/V."""

        if SINGLETON_TOP_R > 0:
            tl.static_assert(not RECORD_DISTORTION)
        if SINGLETON_QUERY_COLUMN:
            tl.static_assert(SINGLETON_TOP_R > 0)
            tl.static_assert(SINGLETON_TOP_R <= 4)
        if SINGLETON_SECONDARY_FRAME_STRIDE > 1:
            tl.static_assert(SINGLETON_QUERY_COLUMN)
            tl.static_assert(SINGLETON_TOP_R == 2)
        if SINGLETON_PRIMARY_ROW_MOD4_OMIT2:
            tl.static_assert(SINGLETON_QUERY_COLUMN)
            tl.static_assert(SINGLETON_TOP_R == 1)
            tl.static_assert(SINGLETON_SECONDARY_FRAME_STRIDE == 1)
        if PACKED_SINGLETON_KV:
            tl.static_assert(SINGLETON_QUERY_COLUMN)
            tl.static_assert(QUERY_COLUMN_MAJOR)
            tl.static_assert(not EXACT_INTRA_FRAME)
            tl.static_assert(EXACT_NEIGHBOR_WINDOW_COLS == 0)
        if SCALED_FP8_KV:
            tl.static_assert(not COMPRESSED_FP8)

        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        if QUERY_COLUMN_MAJOR:
            # Keep the 21 routes that consume the same singleton column next
            # to each other in the launch order. Their singleton K/V working
            # set fits in L2 and is identical across query frames.
            query_col = query_group // Q_OUTER
            query_outer = query_group % Q_OUTER
        else:
            query_outer = query_group // COLS
            query_col = query_group % COLS

        query_rows = tl.arange(0, BLOCK_I)
        tokens_in_tile = tl.arange(0, BLOCK_TOKENS)
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
        if COMPRESSED_FP8:
            compressed_query = query.to(tl.float8e4nv)
        else:
            compressed_query = query

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
        distortion_sum = tl.zeros((BLOCK_I,), tl.float32)

        if EXACT_INTRA_FRAME:
            global_query_outer = QUERY_OUTER_OFFSET + query_outer
            local_tokens = ROWS * COLS
            for token_start in tl.range(
                0,
                local_tokens,
                BLOCK_TOKENS,
                num_stages=LOOP_STAGES,
            ):
                tokens = token_start + tokens_in_tile
                token_mask = tokens < local_tokens
                key_token = global_query_outer * local_tokens + tokens[:, None]
                if HEAD_MAJOR:
                    key_address = (
                        ((batch * TOTAL_HEADS + global_head)
                         * K_OUTER * local_tokens + key_token)
                        * D_HEAD
                        + dims[None, :]
                    )
                else:
                    key_address = (
                        (((batch * K_OUTER * local_tokens + key_token)
                          * TOTAL_HEADS + global_head)
                         * D_HEAD)
                        + dims[None, :]
                    )
                exact_key = tl.load(
                    k_ptr + key_address,
                    mask=token_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                exact_value = tl.load(
                    v_ptr + key_address,
                    mask=token_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                logits = tl.dot(
                    query,
                    tl.trans(exact_key),
                    input_precision=INPUT_PRECISION,
                )
                logits *= SCALE * 1.4426950408889634
                logits = tl.where(
                    token_mask[None, :], logits, -float("inf")
                )
                next_max = tl.maximum(row_max, tl.max(logits, axis=1))
                probabilities = tl.exp2(logits - next_max[:, None])
                old_scale = tl.exp2(row_max - next_max)
                accumulator = accumulator * old_scale[:, None] + tl.dot(
                    probabilities.to(query.dtype),
                    exact_value,
                    input_precision=INPUT_PRECISION,
                )
                row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
                row_max = next_max

        if EXACT_NEIGHBOR_WINDOW_COLS:
            global_query_outer = QUERY_OUTER_OFFSET + query_outer
            neighbor_0 = tl.where(
                global_query_outer == 0,
                1,
                global_query_outer - 1,
            )
            neighbor_1 = tl.where(
                global_query_outer == 0,
                2,
                tl.where(
                    global_query_outer == K_OUTER - 1,
                    global_query_outer - 2,
                    global_query_outer + 1,
                ),
            )
            window_start = tl.minimum(
                tl.maximum(
                    query_col - EXACT_NEIGHBOR_WINDOW_COLS // 2,
                    0,
                ),
                COLS - EXACT_NEIGHBOR_WINDOW_COLS,
            )
            neighbor_tokens = 2 * ROWS * EXACT_NEIGHBOR_WINDOW_COLS
            frame_tokens = ROWS * EXACT_NEIGHBOR_WINDOW_COLS
            for token_start in tl.range(
                0,
                neighbor_tokens,
                BLOCK_TOKENS,
                num_stages=LOOP_STAGES,
            ):
                tokens = token_start + tokens_in_tile
                token_mask = tokens < neighbor_tokens
                neighbor_slot = tokens // frame_tokens
                frame_token = tokens % frame_tokens
                key_row = frame_token // EXACT_NEIGHBOR_WINDOW_COLS
                key_col = (
                    window_start + frame_token % EXACT_NEIGHBOR_WINDOW_COLS
                )
                key_outer = tl.where(
                    neighbor_slot == 0, neighbor_0, neighbor_1
                )
                key_token = (
                    key_outer * ROWS * COLS + key_row * COLS + key_col
                )[:, None]
                if HEAD_MAJOR:
                    key_address = (
                        ((batch * TOTAL_HEADS + global_head)
                         * K_OUTER * ROWS * COLS + key_token)
                        * D_HEAD
                        + dims[None, :]
                    )
                else:
                    key_address = (
                        (((batch * K_OUTER * ROWS * COLS + key_token)
                          * TOTAL_HEADS + global_head)
                         * D_HEAD)
                        + dims[None, :]
                    )
                exact_key = tl.load(
                    k_ptr + key_address,
                    mask=token_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                exact_value = tl.load(
                    v_ptr + key_address,
                    mask=token_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                logits = tl.dot(
                    query,
                    tl.trans(exact_key),
                    input_precision=INPUT_PRECISION,
                )
                logits *= SCALE * 1.4426950408889634
                logits = tl.where(
                    token_mask[None, :], logits, -float("inf")
                )
                next_max = tl.maximum(row_max, tl.max(logits, axis=1))
                probabilities = tl.exp2(logits - next_max[:, None])
                old_scale = tl.exp2(row_max - next_max)
                accumulator = accumulator * old_scale[:, None] + tl.dot(
                    probabilities.to(query.dtype),
                    exact_value,
                    input_precision=INPUT_PRECISION,
                )
                row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
                row_max = next_max

        for token_start in tl.range(
            0,
            total_tokens,
            BLOCK_TOKENS,
            num_stages=LOOP_STAGES,
        ):
            tokens = token_start + tokens_in_tile
            token_mask = tokens < total_tokens
            if EXACT_INTRA_FRAME:
                global_query_outer = QUERY_OUTER_OFFSET + query_outer
                token_key_outer = tokens // (ROWS * C)
                token_mask = token_mask & (token_key_outer != global_query_outer)
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
            if SCALED_FP8_KV:
                compressed_scale_address = route_base + tokens
                merged_key_scale = tl.load(
                    merged_k_scale_ptr + compressed_scale_address,
                    mask=token_mask,
                    other=1.0,
                    cache_modifier=".cg",
                )
                merged_value_scale = tl.load(
                    merged_v_scale_ptr + compressed_scale_address,
                    mask=token_mask,
                    other=1.0,
                    cache_modifier=".cg",
                )
                merged_key = (
                    merged_key.to(tl.float32) * merged_key_scale[:, None]
                ).to(query.dtype)
                merged_value = (
                    merged_value.to(tl.float32) * merged_value_scale[:, None]
                ).to(query.dtype)
            bias = tl.load(
                bias_ptr + route_base + tokens,
                mask=token_mask,
                other=float("inf"),
                cache_modifier=".cg",
            )

            logits = tl.dot(
                compressed_query,
                tl.trans(merged_key),
                input_precision=INPUT_PRECISION,
            )
            logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
            logits = tl.where(token_mask[None, :], logits, -float("inf"))
            next_max = tl.maximum(row_max, tl.max(logits, axis=1))
            probabilities = tl.exp2(logits - next_max[:, None])
            old_scale = tl.exp2(row_max - next_max)
            if COMPRESSED_FP8:
                value_probabilities = probabilities.to(tl.float8e4nv)
            else:
                value_probabilities = probabilities.to(query.dtype)
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                value_probabilities,
                merged_value,
                input_precision=INPUT_PRECISION,
            )
            if RECORD_DISTORTION:
                cluster_ids = tokens % (ROWS * C)
                distance_base = (
                    (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
                     * ACTIVE_HEADS
                     + head)
                    * ROWS
                    * ROWS
                    * C
                )
                distance_offsets = (
                    distance_base
                    + query_rows[:, None] * ROWS * C
                    + cluster_ids[None, :]
                )
                prototype_distance = tl.load(
                    prototype_distance_ptr + distance_offsets,
                    mask=query_mask[:, None] & token_mask[None, :],
                    other=0.0,
                )
                distortion_sum = (
                    distortion_sum * old_scale
                    + tl.sum(probabilities * prototype_distance, axis=1)
                )
            row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
            row_max = next_max

        if SINGLETON_TOP_R > 0:
            if SINGLETON_PRIMARY_ROW_MOD4_OMIT2:
                active_primary_rows = ROWS - (ROWS + 1) // 4
                singleton_total = K_OUTER * active_primary_rows
            elif SINGLETON_SECONDARY_FRAME_STRIDE > 1:
                primary_singleton_total = K_OUTER * ROWS
                secondary_frame_count = (
                    K_OUTER + SINGLETON_SECONDARY_FRAME_STRIDE - 1
                ) // SINGLETON_SECONDARY_FRAME_STRIDE
                singleton_total = (
                    K_OUTER + secondary_frame_count
                ) * ROWS
            else:
                singleton_total = K_OUTER * ROWS * SINGLETON_TOP_R
            singleton_route_base = (
                (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
                 * ACTIVE_HEADS
                 + head)
                * singleton_total
            )
            for token_start in tl.range(
                0,
                singleton_total,
                BLOCK_TOKENS,
                num_stages=LOOP_STAGES,
            ):
                tokens = token_start + tokens_in_tile
                token_mask = tokens < singleton_total
                if SINGLETON_PRIMARY_ROW_MOD4_OMIT2:
                    singleton_key_outer = tokens // active_primary_rows
                    compact_row = tokens % active_primary_rows
                    row_group = compact_row // 3
                    row_in_group = compact_row % 3
                    singleton_key_row = (
                        row_group * 4
                        + row_in_group
                        + (row_in_group == 2).to(tl.int32)
                    )
                    singleton_rank = tl.zeros(
                        (BLOCK_TOKENS,), tl.int32
                    )
                elif SINGLETON_SECONDARY_FRAME_STRIDE > 1:
                    secondary_token = tokens - primary_singleton_total
                    is_secondary = tokens >= primary_singleton_total
                    singleton_key_outer = tl.where(
                        is_secondary,
                        (secondary_token // ROWS)
                        * SINGLETON_SECONDARY_FRAME_STRIDE,
                        tokens // ROWS,
                    )
                    singleton_key_row = tl.where(
                        is_secondary,
                        secondary_token % ROWS,
                        tokens % ROWS,
                    )
                    singleton_rank = is_secondary.to(tl.int32)
                else:
                    singleton_key_outer = tokens // (ROWS * SINGLETON_TOP_R)
                    singleton_rem = tokens % (ROWS * SINGLETON_TOP_R)
                    singleton_key_row = singleton_rem // SINGLETON_TOP_R
                    singleton_rank = singleton_rem % SINGLETON_TOP_R
                if SINGLETON_QUERY_COLUMN:
                    selected_offset = tl.zeros(
                        (BLOCK_TOKENS,), tl.int32
                    )
                    if SINGLETON_TOP_R >= 2:
                        selected_offset = tl.where(
                            singleton_rank == 1, -1, selected_offset
                        )
                    if SINGLETON_TOP_R >= 3:
                        selected_offset = tl.where(
                            singleton_rank == 2, 1, selected_offset
                        )
                    if SINGLETON_TOP_R >= 4:
                        selected_offset = tl.where(
                            singleton_rank == 3, -2, selected_offset
                        )
                    selected_col = (
                        query_col + selected_offset + COLS
                    ) % COLS
                else:
                    selected_col = tl.load(
                        prototype_distance_ptr
                        + singleton_route_base
                        + tokens,
                        mask=token_mask,
                        other=0,
                        cache_modifier=".cg",
                    ).to(tl.int32)
                if EXACT_INTRA_FRAME:
                    global_query_outer = QUERY_OUTER_OFFSET + query_outer
                    token_mask = token_mask & (
                        singleton_key_outer != global_query_outer
                    )
                if PACKED_SINGLETON_KV:
                    # Packed layout is [B, query_col, active_head, K, row, D].
                    # The 945 exact singleton vectors are contiguous for every
                    # route and reused by all 21 query frames.
                    packed_token = (
                        ((((batch * COLS + query_col) * ACTIVE_HEADS + head)
                           * K_OUTER + singleton_key_outer)
                          * ROWS + singleton_key_row)
                    )[:, None]
                    singleton_address = packed_token * D_HEAD + dims[None, :]
                else:
                    raw_token = (
                        singleton_key_outer * ROWS * COLS
                        + singleton_key_row * COLS
                        + selected_col
                    )[:, None]
                    if HEAD_MAJOR:
                        singleton_address = (
                            ((batch * TOTAL_HEADS + global_head)
                             * K_OUTER * ROWS * COLS + raw_token)
                            * D_HEAD
                            + dims[None, :]
                        )
                    else:
                        singleton_address = (
                            (((batch * K_OUTER * ROWS * COLS + raw_token)
                              * TOTAL_HEADS + global_head)
                             * D_HEAD)
                            + dims[None, :]
                        )
                singleton_key = tl.load(
                    k_ptr + singleton_address,
                    mask=token_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                singleton_value = tl.load(
                    v_ptr + singleton_address,
                    mask=token_mask[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                logits = tl.dot(
                    query,
                    tl.trans(singleton_key),
                    input_precision=INPUT_PRECISION,
                )
                logits *= SCALE * 1.4426950408889634
                logits = tl.where(
                    token_mask[None, :], logits, -float("inf")
                )
                next_max = tl.maximum(row_max, tl.max(logits, axis=1))
                probabilities = tl.exp2(logits - next_max[:, None])
                old_scale = tl.exp2(row_max - next_max)
                accumulator = accumulator * old_scale[:, None] + tl.dot(
                    probabilities.to(query.dtype),
                    singleton_value,
                    input_precision=INPUT_PRECISION,
                )
                row_sum = row_sum * old_scale + tl.sum(
                    probabilities, axis=1
                )
                row_max = next_max

        if OUTPUT_HEAD_MAJOR:
            output_address = q_address
        else:
            output_address = (
                (((batch * Q_OUTER * ROWS * COLS + q_token) * TOTAL_HEADS
                  + global_head)
                 * D_HEAD)
                + dims[None, :]
            )
        tl.store(
            out_ptr + output_address,
            accumulator / row_sum[:, None],
            mask=query_mask[:, None],
        )
        if RECORD_DISTORTION:
            score_address = (
                (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
                 * ACTIVE_HEADS
                 + head)
                * ROWS
                + query_rows
            )
            tl.store(
                distortion_score_ptr + score_address,
                distortion_sum / row_sum,
                mask=query_mask,
            )

    @triton.jit
    def _hard_c4_720p_query_column_deferred_route_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        metadata_ptr,
        out_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        HEAD_MAJOR: tl.constexpr,
        OUTPUT_HEAD_MAJOR: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        C: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        LOOP_STAGES: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        """Apply exact query-column peeling while routing base C4 centroids."""

        tl.static_assert(C == 4)
        tl.static_assert(BLOCK_TOKENS == 64)
        tl.static_assert(D_HEAD == 128)

        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)
        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        batch_i64 = batch.to(tl.int64)
        query_outer = query_group // COLS
        query_col = query_group % COLS

        query_rows = tl.arange(0, BLOCK_I)
        tokens_in_tile = tl.arange(0, BLOCK_TOKENS)
        dims = tl.arange(0, 128)
        query_mask = query_rows < ROWS
        q_token = (query_outer * ROWS + query_rows[:, None]) * COLS + query_col
        if HEAD_MAJOR:
            q_address = (
                ((batch * TOTAL_HEADS + global_head)
                 * Q_OUTER * ROWS * COLS + q_token)
                * D_HEAD
                + dims[None, :]
            )
        else:
            q_address = (
                (((batch * Q_OUTER * ROWS * COLS + q_token)
                  * TOTAL_HEADS + global_head)
                 * D_HEAD)
                + dims[None, :]
            )
        query = tl.load(
            q_ptr + q_address,
            mask=query_mask[:, None],
            other=0.0,
        )

        total_tokens = K_OUTER * ROWS * C
        merged_route_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * total_tokens
        )
        metadata_route_base = (
            (((batch_i64 * Q_OUTER + query_outer) * COLS + query_col)
             * ACTIVE_HEADS
             + head)
            * K_OUTER
            * ROWS
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
            tokens = token_start + tokens_in_tile
            token_mask = tokens < total_tokens
            route_rows = tokens // C
            cluster = tokens % C
            packed_metadata = tl.load(
                metadata_ptr + metadata_route_base + route_rows,
                mask=token_mask,
                other=0.0,
                cache_modifier=".cg",
            )
            owner = packed_metadata.to(tl.int32) // 2
            selected_weight = (
                packed_metadata - owner.to(tl.float32) * 2.0
            )
            residual_mass = 1.0 - selected_weight
            is_owner = cluster == owner
            keep_owner = token_mask & is_owner & (residual_mass > 1.0e-6)
            attention_mask = token_mask & (~is_owner | keep_owner)
            safe_residual_mass = tl.where(
                keep_owner,
                residual_mass,
                1.0,
            )

            merged_key = tl.load(
                merged_k_ptr
                + (merged_route_base + tokens[:, None]) * D_HEAD
                + dims[None, :],
                mask=token_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            merged_value = tl.load(
                merged_v_ptr
                + (merged_route_base + tokens[:, None]) * D_HEAD
                + dims[None, :],
                mask=token_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            merged_bias = tl.load(
                bias_ptr + merged_route_base + tokens,
                mask=token_mask,
                other=float("inf"),
                cache_modifier=".cg",
            )
            key_outer = route_rows // ROWS
            key_rows = route_rows % ROWS
            raw_token = (
                key_outer * ROWS * COLS
                + key_rows * COLS
                + query_col
            )[:, None]
            if HEAD_MAJOR:
                raw_address = (
                    ((batch * TOTAL_HEADS + global_head)
                     * K_OUTER * ROWS * COLS + raw_token)
                    * D_HEAD
                    + dims[None, :]
                )
            else:
                raw_address = (
                    (((batch * K_OUTER * ROWS * COLS + raw_token)
                      * TOTAL_HEADS + global_head)
                     * D_HEAD)
                    + dims[None, :]
                )
            raw_key = tl.load(
                k_ptr + raw_address,
                mask=keep_owner[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            raw_value = tl.load(
                v_ptr + raw_address,
                mask=keep_owner[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            corrected_key = (
                merged_key - selected_weight[:, None] * raw_key
            ) / safe_residual_mass[:, None]
            corrected_value = (
                merged_value - selected_weight[:, None] * raw_value
            ) / safe_residual_mass[:, None]
            selected_w_log_w = (
                selected_weight
                * tl.log2(tl.maximum(selected_weight, 1.0e-20))
                * 0.6931471805599453
            )
            corrected_bias = (
                (merged_bias - selected_w_log_w) / safe_residual_mass
                - tl.log2(safe_residual_mass) * 0.6931471805599453
            )
            effective_key = tl.where(
                keep_owner[:, None], corrected_key, merged_key
            ).to(query.dtype)
            effective_value = tl.where(
                keep_owner[:, None], corrected_value, merged_value
            ).to(query.dtype)
            effective_bias = tl.where(
                keep_owner, corrected_bias, merged_bias
            )
            logits = tl.dot(
                query,
                tl.trans(effective_key),
                input_precision=INPUT_PRECISION,
            )
            logits = (
                logits * SCALE - effective_bias[None, :]
            ) * 1.4426950408889634
            logits = tl.where(
                attention_mask[None, :], logits, -float("inf")
            )
            next_max = tl.maximum(row_max, tl.max(logits, axis=1))
            probabilities = tl.exp2(logits - next_max[:, None])
            old_scale = tl.exp2(row_max - next_max)
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(query.dtype),
                effective_value,
                input_precision=INPUT_PRECISION,
            )
            row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
            row_max = next_max

        singleton_total = K_OUTER * ROWS
        for token_start in tl.range(
            0,
            singleton_total,
            BLOCK_TOKENS,
            num_stages=LOOP_STAGES,
        ):
            tokens = token_start + tokens_in_tile
            token_mask = tokens < singleton_total
            key_outer = tokens // ROWS
            key_rows = tokens % ROWS
            raw_token = (
                key_outer * ROWS * COLS
                + key_rows * COLS
                + query_col
            )[:, None]
            if HEAD_MAJOR:
                raw_address = (
                    ((batch * TOTAL_HEADS + global_head)
                     * K_OUTER * ROWS * COLS + raw_token)
                    * D_HEAD
                    + dims[None, :]
                )
            else:
                raw_address = (
                    (((batch * K_OUTER * ROWS * COLS + raw_token)
                      * TOTAL_HEADS + global_head)
                     * D_HEAD)
                    + dims[None, :]
                )
            raw_key = tl.load(
                k_ptr + raw_address,
                mask=token_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            raw_value = tl.load(
                v_ptr + raw_address,
                mask=token_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            logits = tl.dot(
                query,
                tl.trans(raw_key),
                input_precision=INPUT_PRECISION,
            )
            logits *= SCALE * 1.4426950408889634
            logits = tl.where(
                token_mask[None, :], logits, -float("inf")
            )
            next_max = tl.maximum(row_max, tl.max(logits, axis=1))
            probabilities = tl.exp2(logits - next_max[:, None])
            old_scale = tl.exp2(row_max - next_max)
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(query.dtype),
                raw_value,
                input_precision=INPUT_PRECISION,
            )
            row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
            row_max = next_max

        if OUTPUT_HEAD_MAJOR:
            output_address = q_address
        else:
            output_address = (
                (((batch * Q_OUTER * ROWS * COLS + q_token)
                  * TOTAL_HEADS + global_head)
                 * D_HEAD)
                + dims[None, :]
            )
        tl.store(
            out_ptr + output_address,
            accumulator / row_sum[:, None],
            mask=query_mask[:, None],
        )


def _query_response_seed_rows_720p(
    q,
    k,
    scale=None,
    head_start=0,
    head_count=None,
):
    """Return response-span seed rows in compressed-token storage order."""

    if triton is None:
        raise RuntimeError("Triton is required for query-response seeds.")
    q, k = q.contiguous(), k.contiguous()
    batch, query_outer, rows, cols, total_heads, dim = q.shape
    key_outer = k.size(1)
    if (rows, cols, dim) != (HARD_720_ROWS, HARD_720_COLS, 128):
        raise ValueError("query-response seeds require 45x80 blocks with D=128.")
    active_heads = int(head_count or total_heads)
    scale = dim**-0.5 if scale is None else scale
    seed_tokens = (
        batch
        * query_outer
        * cols
        * active_heads
        * key_outer
        * rows
        * HARD_720_C
    )
    seed_rows = torch.empty(seed_tokens, device=q.device, dtype=torch.uint8)
    response_grid = (
        query_outer
        * key_outer
        * rows
        * triton.cdiv(cols, HARD_720_RESPONSE_BLOCK_J),
        batch * active_heads,
    )
    _hard_c4_720p_query_response_seed_kernel[response_grid](
        q,
        k,
        seed_rows,
        query_outer,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        rows,
        cols,
        HARD_720_C,
        HARD_720_RESPONSE_CANDIDATES,
        HARD_720_RESPONSE_PROXIES,
        HARD_720_RESPONSE_BLOCK_J,
        HARD_720_RESPONSE_BLOCK_ROWS,
        HARD_720_RESPONSE_BLOCK_PROXIES,
        HARD_720_INPUT_PRECISION,
        SCALE=float(scale),
        num_warps=HARD_720_RESPONSE_NUM_WARPS,
        num_stages=HARD_720_RESPONSE_NUM_STAGES,
    )
    return seed_rows


def _run_hard_c4_720p(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    head_major=False,
    output_head_major=None,
    use_entropy_bias=True,
    merge_disable_licm=None,
    exact_intra_frame=False,
    exact_neighbor_window_columns=0,
    query_outer_offset=0,
    query_centroid_window=0,
    responsibility_top_r=0,
    exact_query_rows=0,
    query_response_seeds=False,
    external_seed_rows=None,
    external_seed_rows_native=False,
    external_seed_rows_temporal_shared=False,
    temporal_query_row_map=None,
    temporal_compact_seed_rows=None,
    temporal_alpha=0.3,
    route_bias_scale=1.0,
    logit_variance_bias_scale=0.0,
    seed_q=None,
    timings=None,
):
    if triton is None:
        raise RuntimeError("Triton is required for hard C4.")
    scale = q.size(-1) ** -0.5 if scale is None else scale
    merge_disable_licm = (
        HARD_720_MERGE_DISABLE_LICM
        if merge_disable_licm is None
        else merge_disable_licm
    )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    seed_q = q if seed_q is None else seed_q.contiguous()
    if seed_q.shape != q.shape:
        raise ValueError(
            f"seed_q must match q shape {tuple(q.shape)}, got {tuple(seed_q.shape)}"
        )
    if head_major:
        batch, total_heads, query_length, dim = q.shape
        query_outer = query_length // (HARD_720_ROWS * HARD_720_COLS)
        key_outer = k.size(2) // (HARD_720_ROWS * HARD_720_COLS)
    else:
        batch, query_outer, _, _, total_heads, dim = q.shape
        key_outer = k.size(1)
    active_heads = int(head_count or total_heads)
    query_groups = query_outer // HARD_720_QUERY_OUTER_GROUP
    query_outer_offset = int(query_outer_offset)
    exact_neighbor_window_columns = int(exact_neighbor_window_columns or 0)
    query_centroid_window = int(query_centroid_window or 0)
    responsibility_top_r = int(responsibility_top_r or 0)
    exact_query_rows = int(exact_query_rows or 0)
    query_response_seeds = bool(query_response_seeds)
    use_response_seed_rows = query_response_seeds or external_seed_rows is not None
    external_seed_rows_native = bool(external_seed_rows_native)
    external_seed_rows_temporal_shared = bool(external_seed_rows_temporal_shared)
    use_temporal_query_row_map = temporal_query_row_map is not None
    use_temporal_compact_refine = temporal_compact_seed_rows is not None
    route_bias_scale = float(route_bias_scale)
    logit_variance_bias_scale = float(logit_variance_bias_scale)
    if not math.isfinite(logit_variance_bias_scale) or logit_variance_bias_scale < 0:
        raise ValueError("logit variance bias scale must be finite and non-negative")
    if query_centroid_window not in {0, HARD_720_CENTROID_CANDIDATES}:
        raise ValueError(
            "query-centroid Hard C4 supports only an 11-row local window."
        )
    if not 0 <= exact_neighbor_window_columns <= 30:
        raise ValueError(
            "exact neighbor windows support 0 to 30 columns per neighbor."
        )
    if exact_neighbor_window_columns and not exact_intra_frame:
        raise ValueError(
            "exact neighbor windows require exact intra-frame attention."
        )
    if query_centroid_window and (
        exact_intra_frame or exact_neighbor_window_columns
    ):
        raise ValueError(
            "query-centroid seeds and exact-window attention are mutually exclusive."
        )
    if responsibility_top_r not in {0, HARD_720_RESPONSIBILITY_TOP_R}:
        raise ValueError(
            "responsibility-centroid Hard C4 supports exactly two refinements."
        )
    if responsibility_top_r and (
        query_centroid_window
        or exact_intra_frame
        or exact_neighbor_window_columns
    ):
        raise ValueError(
            "responsibility-centroid refinement requires plain 45x80 Hard C4."
        )
    if exact_query_rows < 0 or exact_query_rows > HARD_720_ROWS:
        raise ValueError(
            f"exact query rows must be in [0, {HARD_720_ROWS}]."
        )
    if exact_query_rows and (
        head_major
        or query_centroid_window
        or responsibility_top_r
        or exact_intra_frame
        or exact_neighbor_window_columns
    ):
        raise ValueError(
            "budgeted exact-query refinement requires plain block-layout Hard C4."
        )
    if use_response_seed_rows and (
        head_major
        or query_centroid_window
        or responsibility_top_r
        or exact_query_rows
        or exact_intra_frame
        or exact_neighbor_window_columns
    ):
        raise ValueError(
            "query-response seeds require the plain block-layout 45x80 Hard C4 path."
        )
    if query_response_seeds and external_seed_rows is not None:
        raise ValueError(
            "internal query-response selection and external seed rows are mutually exclusive."
        )
    if external_seed_rows_native and external_seed_rows is None:
        raise ValueError("native external seed layout requires external seed rows.")
    if external_seed_rows_temporal_shared and external_seed_rows is None:
        raise ValueError("temporally shared seed layout requires external seed rows.")
    if use_temporal_query_row_map and not external_seed_rows_temporal_shared:
        raise ValueError(
            "temporal query row maps require temporally shared seed rows."
        )
    if external_seed_rows_native and external_seed_rows_temporal_shared:
        raise ValueError("external seed rows can use only one specialized layout.")
    if use_temporal_compact_refine and use_response_seed_rows:
        raise ValueError(
            "temporal compact refinement and response seed tables are mutually exclusive."
        )
    if use_temporal_compact_refine and (
        head_major
        or query_outer != HARD_720_QUERY_OUTER_GROUP
        or query_centroid_window
        or responsibility_top_r
        or exact_query_rows
        or exact_intra_frame
        or exact_neighbor_window_columns
    ):
        raise ValueError(
            "temporal compact refinement requires the plain 21-frame block-layout C4 path."
        )
    if exact_intra_frame:
        if query_outer_offset < 0 or query_outer_offset + query_outer > key_outer:
            raise ValueError(
                "exact intra-frame attention requires every query frame to "
                "have a matching key frame."
            )

    compressed_tokens = (
        batch
        * query_outer
        * HARD_720_COLS
        * active_heads
        * key_outer
        * HARD_720_ROWS
        * HARD_720_C
    )
    merged_k = torch.empty(
        (compressed_tokens, dim), device=q.device, dtype=q.dtype
    )
    merged_v = torch.empty_like(merged_k)
    unused_compressed_scale = torch.empty(
        1, device=q.device, dtype=torch.float32
    )
    bias = torch.empty(compressed_tokens, device=q.device, dtype=torch.float32)
    assignment_tokens = (
        batch
        * query_outer
        * HARD_720_COLS
        * active_heads
        * key_outer
        * HARD_720_ROWS
        * HARD_720_COLS
    )
    assignments = torch.empty(
        assignment_tokens if responsibility_top_r else 1,
        device=q.device,
        dtype=torch.uint8,
    )
    centroid_tokens = (
        batch
        * query_outer
        * HARD_720_ROWS
        * HARD_720_COLS
        * active_heads
        * HARD_720_C
    )
    centroids = (
        torch.empty((centroid_tokens, dim), device=q.device, dtype=q.dtype)
        if query_centroid_window
        else q
    )
    expected_seed_tokens = (
        batch
        * query_outer
        * HARD_720_COLS
        * active_heads
        * key_outer
        * HARD_720_ROWS
        * HARD_720_C
    )
    response_seed_frames = 1
    if external_seed_rows is not None and external_seed_rows_temporal_shared:
        if external_seed_rows.device != q.device:
            raise ValueError("temporally shared seed rows must be on the QKV device.")
        if external_seed_rows.ndim == 6:
            response_seed_frames = 1
            expected_shared_shape = (
                batch,
                HARD_720_COLS,
                active_heads,
                key_outer,
                HARD_720_ROWS,
                HARD_720_C,
            )
        elif external_seed_rows.ndim == 7:
            response_seed_frames = int(external_seed_rows.size(1))
            if not 1 <= response_seed_frames <= 4:
                raise ValueError("temporally shared seed rows support 1 to 4 frames.")
            expected_shared_shape = (
                batch,
                response_seed_frames,
                HARD_720_COLS,
                active_heads,
                key_outer,
                HARD_720_ROWS,
                HARD_720_C,
            )
        else:
            raise ValueError(
                "temporally shared seed rows must be [B, column, active_head, "
                "K, row, C] or [B, R, column, active_head, K, row, C]."
            )
        if tuple(external_seed_rows.shape) != expected_shared_shape:
            raise ValueError(
                "temporally shared seed rows must have shape "
                f"{expected_shared_shape}, got {tuple(external_seed_rows.shape)}."
            )
        response_seed_rows = external_seed_rows.to(torch.uint8).contiguous()
        if use_temporal_query_row_map:
            expected_row_map_shape = (
                batch,
                query_outer,
                HARD_720_COLS,
                total_heads,
                HARD_720_ROWS,
            )
            if temporal_query_row_map.device != q.device:
                raise ValueError("temporal query row maps must be on the QKV device.")
            if temporal_query_row_map.dtype != torch.uint8:
                raise TypeError("temporal query row maps must use uint8 row indices.")
            if tuple(temporal_query_row_map.shape) != expected_row_map_shape:
                raise ValueError(
                    "temporal query row maps must have shape "
                    f"{expected_row_map_shape}, got "
                    f"{tuple(temporal_query_row_map.shape)}."
                )
            temporal_query_row_map = temporal_query_row_map.contiguous()
    elif external_seed_rows is not None and external_seed_rows_native:
        if external_seed_rows.device != q.device:
            raise ValueError("native external seed rows must be on the QKV device.")
        if external_seed_rows.ndim != 7:
            raise ValueError(
                "native external seed rows must have shape "
                "[B, R, K, row, column, H, C]."
            )
        response_seed_frames = int(external_seed_rows.size(1))
        if response_seed_frames not in {1, query_outer}:
            raise ValueError(
                "native external seed rows must contain either one shared frame "
                f"or all {query_outer} query frames."
            )
        expected_native_shape = (
            batch,
            response_seed_frames,
            key_outer,
            HARD_720_ROWS,
            HARD_720_COLS,
            total_heads,
            HARD_720_C,
        )
        if tuple(external_seed_rows.shape) != expected_native_shape:
            raise ValueError(
                "native external seed rows must have shape "
                f"{expected_native_shape}, got {tuple(external_seed_rows.shape)}."
            )
        response_seed_rows = external_seed_rows.to(torch.uint8).contiguous()
    elif external_seed_rows is not None:
        if external_seed_rows.device != q.device:
            raise ValueError("external seed rows must be on the QKV device.")
        if external_seed_rows.numel() != expected_seed_tokens:
            raise ValueError(
                "external seed rows must contain "
                f"{expected_seed_tokens:,} entries, got "
                f"{external_seed_rows.numel():,}."
            )
        response_seed_rows = external_seed_rows.to(torch.uint8).contiguous()
    elif query_response_seeds:
        response_seed_rows = _query_response_seed_rows_720p(
            q,
            k,
            scale=scale,
            head_start=head_start,
            head_count=active_heads,
        )
    else:
        response_seed_rows = torch.empty(1, device=q.device, dtype=torch.uint8)
    expected_temporal_seed_tokens = (
        batch
        * key_outer
        * HARD_720_ROWS
        * HARD_720_COLS
        * total_heads
        * HARD_720_C
    )
    if use_temporal_compact_refine:
        if temporal_compact_seed_rows.device != q.device:
            raise ValueError("temporal compact seed rows must be on the QKV device.")
        if temporal_compact_seed_rows.dtype != torch.uint8:
            raise TypeError("temporal compact seed rows must use uint8 row indices.")
        if temporal_compact_seed_rows.numel() != expected_temporal_seed_tokens:
            raise ValueError(
                "temporal compact seed rows must contain "
                f"{expected_temporal_seed_tokens:,} entries, got "
                f"{temporal_compact_seed_rows.numel():,}."
            )
        temporal_seed_rows = temporal_compact_seed_rows.contiguous()
    else:
        temporal_seed_rows = torch.empty(1, device=q.device, dtype=torch.uint8)
    output_head_major = head_major if output_head_major is None else output_head_major
    if output is None:
        if head_major and not output_head_major:
            output = torch.empty(
                (batch, query_length, total_heads * dim),
                device=q.device,
                dtype=q.dtype,
            )
        else:
            output = torch.empty_like(q)

    distance_tokens = (
        batch
        * query_outer
        * HARD_720_COLS
        * active_heads
        * HARD_720_ROWS
        * HARD_720_ROWS
        * HARD_720_C
    )
    score_tokens = (
        batch
        * query_outer
        * HARD_720_COLS
        * active_heads
        * HARD_720_ROWS
    )
    prototype_distance = torch.empty(
        distance_tokens if exact_query_rows else 1,
        device=q.device,
        dtype=torch.float32,
    )
    distortion_score = torch.empty(
        score_tokens if exact_query_rows else 1,
        device=q.device,
        dtype=torch.float32,
    )

    if query_centroid_window:
        centroid_grid = (
            query_outer * HARD_720_ROWS * HARD_720_COLS,
            batch * active_heads,
        )
        _hard_c4_720p_query_centroid_kernel[centroid_grid](
            q,
            centroids,
            query_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            query_centroid_window,
            HARD_720_CENTROID_BLOCK_ROWS,
            HARD_720_INPUT_PRECISION,
            num_warps=HARD_720_CENTROID_NUM_WARPS,
            num_stages=HARD_720_CENTROID_NUM_STAGES,
        )

    merge_grid = (
        query_groups * key_outer * HARD_720_ROWS,
        batch * active_heads,
    )
    merge_loop_stages = (
        1 if use_temporal_compact_refine else HARD_720_MERGE_LOOP_STAGES
    )
    merge_num_stages = (
        1 if use_temporal_compact_refine else HARD_720_MERGE_NUM_STAGES
    )
    timing_events = None
    if timings is not None:
        timing_events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        timing_events[0].record()
    if external_seed_rows_temporal_shared:
        _hard_c4_720p_temporal_shared_merge_kernel[merge_grid](
            seed_q,
            response_seed_rows,
            temporal_query_row_map if use_temporal_query_row_map else seed_q,
            k,
            v,
            merged_k,
            merged_v,
            unused_compressed_scale,
            unused_compressed_scale,
            bias,
            assignments,
            distortion_score,
            query_outer,
            response_seed_frames,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_C,
            HARD_720_MERGE_BLOCK_J,
            merge_loop_stages,
            1,
            1,
            merge_disable_licm,
            HARD_720_INPUT_PRECISION,
            use_entropy_bias,
            ROUTE_BIAS_SCALE=route_bias_scale,
            LOGIT_VARIANCE_BIAS_SCALE=logit_variance_bias_scale,
            STORE_CLUSTER_MASS=False,
            FUSE_CLUSTER_MASS=False,
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
            USE_TEMPORAL_ROW_MAP=use_temporal_query_row_map,
            SCALE=float(scale),
            num_warps=HARD_720_MERGE_NUM_WARPS,
            num_stages=merge_num_stages,
        )
    else:
        _hard_c4_720p_merge_kernel[merge_grid](
            seed_q,
            centroids,
            response_seed_rows,
            temporal_seed_rows,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            assignments,
            query_outer,
            query_groups,
            HARD_720_QUERY_OUTER_GROUP,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_MERGE_BLOCK_J,
            merge_loop_stages,
            False,
            merge_disable_licm,
            HARD_720_INPUT_PRECISION,
            bool(query_centroid_window),
            use_response_seed_rows,
            external_seed_rows_native,
            external_seed_rows_temporal_shared,
            response_seed_frames,
            use_temporal_compact_refine,
            use_entropy_bias,
            bool(responsibility_top_r),
            exact_neighbor_window_columns,
            query_outer_offset,
            ROUTE_BIAS_SCALE=route_bias_scale,
            TEMPORAL_ALPHA=float(temporal_alpha),
            SCALE=float(scale),
            num_warps=HARD_720_MERGE_NUM_WARPS,
            num_stages=merge_num_stages,
        )
    if timing_events is not None:
        timing_events[1].record()

    if responsibility_top_r:
        lse_tokens = (
            batch
            * query_outer
            * HARD_720_COLS
            * active_heads
            * HARD_720_ROWS
        )
        lse = torch.empty(lse_tokens, device=q.device, dtype=torch.float32)
        expansion_centroids = torch.empty_like(merged_k)
        selected_ids = torch.empty(
            (
                batch
                * query_outer
                * HARD_720_COLS
                * active_heads
                * key_outer
                * HARD_720_ROWS
                * responsibility_top_r
            ),
            device=q.device,
            dtype=torch.int32,
        )
        lse_grid = (
            query_outer * HARD_720_COLS,
            batch * active_heads,
        )
        _hard_c4_720p_responsibility_lse_kernel[lse_grid](
            q,
            merged_k,
            bias,
            lse,
            query_outer,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_ROUTE_BLOCK_I,
            HARD_720_ROUTE_BLOCK_TOKENS,
            HARD_720_INPUT_PRECISION,
            SCALE=float(scale),
            num_warps=HARD_720_ROUTE_NUM_WARPS,
            num_stages=HARD_720_ROUTE_NUM_STAGES,
        )
        responsibility_grid = (
            triton.cdiv(
                key_outer * HARD_720_ROWS * HARD_720_C,
                HARD_720_RESPONSIBILITY_BLOCK_TOKENS,
            ),
            batch * active_heads,
            query_outer * HARD_720_COLS,
        )
        _hard_c4_720p_responsibility_centroid_kernel[responsibility_grid](
            q,
            merged_k,
            bias,
            lse,
            expansion_centroids,
            selected_ids,
            query_outer,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            responsibility_top_r,
            HARD_720_ROUTE_BLOCK_I,
            HARD_720_RESPONSIBILITY_BLOCK_TOKENS,
            HARD_720_INPUT_PRECISION,
            SCALE=float(scale),
            num_warps=HARD_720_RESPONSIBILITY_NUM_WARPS,
            num_stages=HARD_720_RESPONSIBILITY_NUM_STAGES,
        )
        _hard_c4_720p_refine_selected_kernel[merge_grid](
            expansion_centroids,
            selected_ids,
            assignments,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            query_outer,
            query_groups,
            HARD_720_QUERY_OUTER_GROUP,
            key_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            responsibility_top_r,
            HARD_720_MERGE_BLOCK_J,
            HARD_720_MERGE_LOOP_STAGES,
            HARD_720_INPUT_PRECISION,
            SCALE=float(scale),
            num_warps=HARD_720_MERGE_NUM_WARPS,
            num_stages=HARD_720_MERGE_NUM_STAGES,
        )

    if exact_query_rows:
        distance_grid = (
            triton.cdiv(
                HARD_720_ROWS * HARD_720_C,
                HARD_720_DISTORTION_BLOCK_CLUSTERS,
            ),
            batch * active_heads,
            query_outer * HARD_720_COLS,
        )
        _hard_c4_720p_prototype_distance_kernel[distance_grid](
            q,
            prototype_distance,
            query_outer,
            total_heads,
            active_heads,
            int(head_start),
            dim,
            head_major,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_DISTORTION_BLOCK_CLUSTERS,
            HARD_720_INPUT_PRECISION,
            num_warps=HARD_720_ROUTE_NUM_WARPS,
            num_stages=HARD_720_ROUTE_NUM_STAGES,
        )

    # MI300 exposes 64 KiB of shared memory to this Triton launch. Keeping one
    # loop stage preserves the same full route attention while fitting that limit.
    route_loop_stages = 1 if torch.version.hip is not None else 3
    route_grid = (1, batch * active_heads, query_outer * HARD_720_COLS)
    _hard_c4_720p_route_kernel[route_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        unused_compressed_scale,
        unused_compressed_scale,
        bias,
        prototype_distance,
        distortion_score,
        output,
        query_outer,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        head_major,
        output_head_major,
        HARD_720_ROWS,
        HARD_720_COLS,
        HARD_720_C,
        HARD_720_ROUTE_BLOCK_I,
        HARD_720_ROUTE_BLOCK_TOKENS,
        route_loop_stages,
        HARD_720_INPUT_PRECISION,
        bool(exact_intra_frame),
        exact_neighbor_window_columns,
        bool(exact_query_rows),
        query_outer_offset,
        SINGLETON_TOP_R=0,
        SINGLETON_QUERY_COLUMN=False,
        SINGLETON_SECONDARY_FRAME_STRIDE=1,
        SINGLETON_PRIMARY_ROW_MOD4_OMIT2=False,
        QUERY_COLUMN_MAJOR=False,
        PACKED_SINGLETON_KV=False,
        SCALED_FP8_KV=False,
        COMPRESSED_FP8=False,
        SCALE=float(scale),
        num_warps=HARD_720_ROUTE_NUM_WARPS,
        num_stages=HARD_720_ROUTE_NUM_STAGES,
    )
    if timing_events is not None:
        timing_events[2].record()
        timing_events[2].synchronize()
        timings["merge_ms"] = timing_events[0].elapsed_time(timing_events[1])
        timings["route_ms"] = timing_events[1].elapsed_time(timing_events[2])
    if exact_query_rows:
        selected_rows = distortion_score.view(
            batch,
            query_outer,
            HARD_720_COLS,
            active_heads,
            HARD_720_ROWS,
        ).topk(exact_query_rows, dim=-1).indices
        active_query = q[
            ..., int(head_start) : int(head_start) + active_heads, :
        ].permute(0, 1, 3, 4, 2, 5)
        gather_index = selected_rows.unsqueeze(-1).expand(
            *selected_rows.shape, dim
        )
        selected_query = active_query.gather(4, gather_index)
        selected_query = selected_query.permute(0, 3, 1, 2, 4, 5).reshape(
            batch, active_heads, -1, dim
        )
        dense_key = k[
            ..., int(head_start) : int(head_start) + active_heads, :
        ].permute(0, 4, 1, 2, 3, 5).reshape(
            batch, active_heads, -1, dim
        )
        dense_value = v[
            ..., int(head_start) : int(head_start) + active_heads, :
        ].permute(0, 4, 1, 2, 3, 5).reshape(
            batch, active_heads, -1, dim
        )
        exact_output = F.scaled_dot_product_attention(
            selected_query,
            dense_key,
            dense_value,
            dropout_p=0.0,
            is_causal=False,
            scale=float(scale),
        )
        exact_output = exact_output.reshape(
            batch,
            active_heads,
            query_outer,
            HARD_720_COLS,
            exact_query_rows,
            dim,
        ).permute(0, 2, 3, 1, 4, 5)
        active_output = output[
            ..., int(head_start) : int(head_start) + active_heads, :
        ].permute(0, 1, 3, 4, 2, 5)
        active_output.scatter_(4, gather_index, exact_output)
    return output


def hard_c4_720p_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    seed_q=None,
):
    """Run exact hard C4 on ``[B, 21, 45, 80, H, 128]`` QKV blocks."""

    if q.size(1) % HARD_720_QUERY_OUTER_GROUP:
        raise ValueError(
            "720p Hard C4 requires query-frame chunks divisible by "
            f"{HARD_720_QUERY_OUTER_GROUP}."
        )

    return _run_hard_c4_720p(
        q,
        k,
        v,
        seed_q=seed_q,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
    )


def hard_c4_720p_query_response_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run all-layer C4 with the regular 15x8 response-span selector."""

    if q.size(1) % HARD_720_QUERY_OUTER_GROUP:
        raise ValueError(
            "720p query-response C4 requires all 21 latent frames per chunk."
        )
    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        query_response_seeds=True,
        route_bias_scale=1.2,
    )


def hard_c4_720p_seed_rows_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run the optimized 21-frame Hard-C4 path with external row seeds.

    ``seed_rows`` uses compressed-token order
    ``[B, Q, column, active_head, K, key_row, C]``. Keeping only active heads
    allows a selector to amortize routing decisions without materializing a
    full 40-head table.
    """

    if q.size(1) % HARD_720_QUERY_OUTER_GROUP:
        raise ValueError(
            "external-seed 720p C4 requires all 21 latent frames per chunk."
        )
    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        external_seed_rows=seed_rows,
    )


def hard_c4_720p_native_seed_rows_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run Hard C4 directly from selector-native seed rows.

    ``seed_rows`` uses ``[B, R, K, row, column, H, C]``, where ``R`` is either
    one temporally shared routing frame or all query frames.  This avoids the
    21x temporal expansion and route-table transpose used by the legacy path.
    """

    if q.size(1) % HARD_720_QUERY_OUTER_GROUP:
        raise ValueError(
            "native external-seed 720p C4 requires all 21 latent frames per chunk."
        )
    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        external_seed_rows=seed_rows,
        external_seed_rows_native=True,
    )


def hard_c4_720p_temporal_shared_seed_rows_attention_triton_blocks(
    q,
    k,
    v,
    seed_rows,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    logit_variance_bias_scale=0.0,
    temporal_query_row_map=None,
):
    """Run all query frames from one route-contiguous shared seed table.

    ``seed_rows`` uses ``[B, column, active_head, K, row, C]``.  It retains the
    merge kernel's contiguous table order without expanding one selector frame
    into 21 identical copies.
    """

    if q.size(1) % HARD_720_QUERY_OUTER_GROUP:
        raise ValueError(
            "temporally shared 720p C4 requires all 21 latent frames per chunk."
        )
    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        external_seed_rows=seed_rows,
        external_seed_rows_temporal_shared=True,
        temporal_query_row_map=temporal_query_row_map,
        logit_variance_bias_scale=logit_variance_bias_scale,
    )


def hard_c4_720p_temporal_compact_attention_triton_blocks(
    q,
    k,
    v,
    coarse_seed_rows,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    alpha=0.3,
):
    """Fuse compact temporal seed refinement into the optimized C4 merge."""

    if q.size(1) != HARD_720_QUERY_OUTER_GROUP:
        raise ValueError("temporal compact C4 requires exactly 21 query frames.")
    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        temporal_compact_seed_rows=coarse_seed_rows,
        temporal_alpha=alpha,
    )


def hard_c4_720p_exact_intra_frame_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    query_outer_offset=0,
    exact_neighbor_window_columns=0,
):
    """Use exact local K/V and optional disjoint exact-neighbor windows."""

    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        exact_intra_frame=True,
        exact_neighbor_window_columns=exact_neighbor_window_columns,
        query_outer_offset=query_outer_offset,
    )


def hard_c4_720p_exact_query_top3_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Recompute the three highest-certainty-risk query rows exactly."""

    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        exact_query_rows=3,
    )


def hard_c4_720p_query_centroid_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Use P11 one-step local Lloyd centroids as the four routing seeds."""

    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        query_centroid_window=HARD_720_CENTROID_CANDIDATES,
    )


def hard_c4_720p_responsibility_centroid_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run RC-GMA with fixed routing and the top-two centroid refinements."""

    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        responsibility_top_r=HARD_720_RESPONSIBILITY_TOP_R,
    )


def hard_c4_720p_attention_triton_head_major(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    seed_q=None,
):
    """Run exact hard C4 directly on Diffusers-style ``[B, H, N, 128]`` QKV."""

    return _run_hard_c4_720p(
        q,
        k,
        v,
        seed_q=seed_q,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        head_major=True,
    )


def hard_c4_720p_attention_triton_head_major_to_token(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run exact hard C4 on head-major QKV and store contiguous token-major output."""

    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        head_major=True,
        output_head_major=False,
    )


def hard_c4_720p_attention_triton_head_major_no_entropy_bias(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run hard C4 with zero route bias while still masking empty seeds."""

    return _run_hard_c4_720p(
        q,
        k,
        v,
        scale=scale,
        output=output,
        head_start=head_start,
        head_count=head_count,
        head_major=True,
        use_entropy_bias=False,
    )


__all__ = [
    "HARD_720_C",
    "HARD_720_CENTROID_CANDIDATES",
    "HARD_720_COLS",
    "HARD_720_MERGE_BLOCK_J",
    "HARD_720_MERGE_BLOCK_L",
    "HARD_720_MERGE_DISABLE_LICM",
    "HARD_720_MERGE_LOOP_STAGES",
    "HARD_720_MERGE_NUM_STAGES",
    "HARD_720_MERGE_NUM_WARPS",
    "HARD_720_QUERY_OUTER_GROUP",
    "HARD_720_ROWS",
    "hard_c4_720p_attention_triton_blocks",
    "hard_c4_720p_query_response_attention_triton_blocks",
    "hard_c4_720p_seed_rows_attention_triton_blocks",
    "hard_c4_720p_temporal_compact_attention_triton_blocks",
    "hard_c4_720p_exact_query_top3_attention_triton_blocks",
    "hard_c4_720p_exact_intra_frame_attention_triton_blocks",
    "hard_c4_720p_query_centroid_attention_triton_blocks",
    "hard_c4_720p_attention_triton_head_major",
    "hard_c4_720p_attention_triton_head_major_to_token",
    "hard_c4_720p_attention_triton_head_major_no_entropy_bias",
]
