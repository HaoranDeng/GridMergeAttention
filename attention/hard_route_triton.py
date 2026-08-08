import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


if triton is not None:
    _WORKSPACE_CACHE = {}


    def _cached_empty(name, shape, device, dtype):
        """Reuse no-grad intermediate buffers for inference-only fast paths."""
        if torch.is_grad_enabled():
            return torch.empty(shape, device=device, dtype=dtype)
        cache_key = (name, torch.Size(shape), device, dtype)
        out = _WORKSPACE_CACHE.get(cache_key)
        if out is None:
            out = torch.empty(shape, device=device, dtype=dtype)
            _WORKSPACE_CACHE[cache_key] = out
        return out


    @triton.jit
    def _hard_route_merge_kernel(
        seed_ptr,
        key_ptr,
        value_ptr,
        merged_key_ptr,
        merged_value_ptr,
        routing_bias_ptr,
        A: tl.constexpr,
        F: tl.constexpr,
        K_ROWS: tl.constexpr,
        J: tl.constexpr,
        C_SEEDS: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_col_block = tl.program_id(1)

        head = group_id % H
        tmp = group_id // H
        key_row = tmp % K_ROWS
        tmp = tmp // K_ROWS
        key_outer = tmp % F
        tmp = tmp // F
        query_outer = tmp % A
        batch = tmp // A

        query_offsets = query_col_block * BLOCK_J + tl.arange(0, BLOCK_J)
        key_offsets = tl.arange(0, BLOCK_L)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_query = query_offsets < J
        valid_key = key_offsets < L
        valid_dim = dim_offsets < D_HEAD

        key_base = key_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        key_block = tl.load(
            key_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        value_base = value_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        value_block = tl.load(
            value_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )

        best_logits = tl.full((BLOCK_J, BLOCK_L), -float("inf"), tl.float32)
        assignments = tl.zeros((BLOCK_J, BLOCK_L), tl.int32)
        for seed_idx in tl.static_range(0, BLOCK_C):
            seed_offsets = (
                (((((batch * A + query_outer) * F + key_outer) * K_ROWS + key_row)
                  * J + query_offsets[:, None]) * C_SEEDS + seed_idx)
                * H + head
            ) * D_HEAD + dim_offsets[None, :]
            seed_vec = tl.load(
                seed_ptr + seed_offsets,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            logits = tl.dot(seed_vec, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
            logits = tl.where(
                (seed_idx < C_SEEDS) & valid_query[:, None] & valid_key[None, :],
                logits,
                -float("inf"),
            )
            update = logits > best_logits
            best_logits = tl.where(update, logits, best_logits)
            assignments = tl.where(update, seed_idx, assignments)

        for seed_idx in tl.static_range(0, BLOCK_C):
            seed_offsets = (
                (((((batch * A + query_outer) * F + key_outer) * K_ROWS + key_row)
                  * J + query_offsets[:, None]) * C_SEEDS + seed_idx)
                * H + head
            ) * D_HEAD + dim_offsets[None, :]
            seed_vec = tl.load(
                seed_ptr + seed_offsets,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            logits = tl.dot(seed_vec, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
            active_seed = (seed_idx < C_SEEDS) & valid_query[:, None] & valid_key[None, :]
            assigned_to_seed = assignments == seed_idx
            has_seed = tl.sum(tl.where(assigned_to_seed & active_seed, 1, 0), axis=1) > 0
            forced_first_key = (~has_seed[:, None]) & (key_offsets[None, :] == 0)
            active = active_seed & (assigned_to_seed | forced_first_key)
            has_keys = tl.sum(tl.where(active, 1, 0), axis=1) > 0

            row_max = tl.max(tl.where(active, logits, -float("inf")), axis=1)
            if USE_EXP2:
                probs = tl.where(
                    active,
                    tl.exp2((logits - row_max[:, None]) * 1.4426950408889634),
                    0.0,
                )
            else:
                probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weights = probs / safe_denom[:, None]
            if USE_EXP2:
                log_denom = row_max + tl.log2(safe_denom) * 0.6931471805599453
            else:
                log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            expected_logit = tl.sum(tl.where(active, weights * logits, 0.0), axis=1)
            routing_bias = expected_logit - log_denom
            routing_bias = tl.where(has_keys, routing_bias, 0.0)

            merged_offset = (
                ((((((batch * A + query_outer) * F + key_outer) * J + query_offsets[:, None])
                   * K_ROWS + key_row) * C_SEEDS + seed_idx) * H + head)
                * D_HEAD + dim_offsets[None, :]
            )
            tl.store(
                merged_key_ptr + merged_offset,
                merged_key,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
            )
            tl.store(
                merged_value_ptr + merged_offset,
                merged_value,
                mask=(seed_idx < C_SEEDS) & valid_query[:, None] & valid_dim[None, :],
            )

            bias_offset = (
                ((((((batch * H + head) * A + query_outer) * F + key_outer)
                   * K_ROWS + key_row) * J + query_offsets) * C_SEEDS + seed_idx)
            )
            tl.store(
                routing_bias_ptr + bias_offset,
                routing_bias,
                mask=(seed_idx < C_SEEDS) & valid_query,
            )

    @triton.jit
    def _hard_route_merge_c4_kernel(
        seed_ptr,
        key_ptr,
        value_ptr,
        merged_key_ptr,
        merged_value_ptr,
        routing_bias_ptr,
        A: tl.constexpr,
        F: tl.constexpr,
        K_ROWS: tl.constexpr,
        J: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D_HEAD: tl.constexpr,
        C_SEEDS: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_col_block = tl.program_id(1)

        head = group_id % H
        tmp = group_id // H
        key_row = tmp % K_ROWS
        tmp = tmp // K_ROWS
        key_outer = tmp % F
        tmp = tmp // F
        query_outer = tmp % A
        batch = tmp // A

        query_offsets = query_col_block * BLOCK_J + tl.arange(0, BLOCK_J)
        key_offsets = tl.arange(0, BLOCK_L)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_query = query_offsets < J
        valid_key = key_offsets < L
        valid_dim = dim_offsets < D_HEAD

        key_base = key_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        key_block = tl.load(
            key_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        value_base = value_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        value_block = tl.load(
            value_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )

        seed_base = (
            ((((((batch * A + query_outer) * F + key_outer) * K_ROWS + key_row) * J
               + query_offsets[:, None]) * 4) * H + head) * D_HEAD
        )
        seed0 = tl.load(
            seed_ptr + seed_base + dim_offsets[None, :],
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed1 = tl.load(
            seed_ptr + seed_base + H * D_HEAD + dim_offsets[None, :],
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed2 = tl.load(
            seed_ptr + seed_base + 2 * H * D_HEAD + dim_offsets[None, :],
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed3 = tl.load(
            seed_ptr + seed_base + 3 * H * D_HEAD + dim_offsets[None, :],
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )

        logits0 = tl.dot(seed0, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        logits1 = tl.dot(seed1, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        logits2 = tl.dot(seed2, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        logits3 = tl.dot(seed3, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        valid_logits = valid_query[:, None] & valid_key[None, :]
        logits0 = tl.where(valid_logits, logits0, -float("inf"))
        logits1 = tl.where(valid_logits, logits1, -float("inf"))
        logits2 = tl.where(valid_logits, logits2, -float("inf"))
        logits3 = tl.where(valid_logits, logits3, -float("inf"))

        best_logits = logits0
        assignments = tl.zeros((BLOCK_J, BLOCK_L), tl.int32)
        update = logits1 > best_logits
        best_logits = tl.where(update, logits1, best_logits)
        assignments = tl.where(update, 1, assignments)
        update = logits2 > best_logits
        best_logits = tl.where(update, logits2, best_logits)
        assignments = tl.where(update, 2, assignments)
        update = logits3 > best_logits
        assignments = tl.where(update, 3, assignments)

        for seed_idx in tl.static_range(0, 4):
            logits = logits0
            if seed_idx == 1:
                logits = logits1
            if seed_idx == 2:
                logits = logits2
            if seed_idx == 3:
                logits = logits3

            assigned_to_seed = assignments == seed_idx
            has_seed = tl.sum(tl.where(assigned_to_seed & valid_logits, 1, 0), axis=1) > 0
            forced_first_key = (~has_seed[:, None]) & (key_offsets[None, :] == 0)
            active = valid_logits & (assigned_to_seed | forced_first_key)
            has_keys = tl.sum(tl.where(active, 1, 0), axis=1) > 0

            row_max = tl.max(tl.where(active, logits, -float("inf")), axis=1)
            if USE_EXP2:
                probs = tl.where(
                    active,
                    tl.exp2((logits - row_max[:, None]) * 1.4426950408889634),
                    0.0,
                )
            else:
                probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weights = probs / safe_denom[:, None]
            if USE_EXP2:
                log_denom = row_max + tl.log2(safe_denom) * 0.6931471805599453
            else:
                log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            expected_logit = tl.sum(tl.where(active, weights * logits, 0.0), axis=1)
            routing_bias = expected_logit - log_denom
            routing_bias = tl.where(has_keys, routing_bias, 0.0)

            merged_offset = (
                ((((((batch * A + query_outer) * F + key_outer) * J + query_offsets[:, None])
                   * K_ROWS + key_row) * 4 + seed_idx) * H + head)
                * D_HEAD + dim_offsets[None, :]
            )
            tl.store(
                merged_key_ptr + merged_offset,
                merged_key,
                mask=valid_query[:, None] & valid_dim[None, :],
            )
            tl.store(
                merged_value_ptr + merged_offset,
                merged_value,
                mask=valid_query[:, None] & valid_dim[None, :],
            )
            bias_offset = (
                ((((((batch * H + head) * A + query_outer) * F + key_outer)
                   * K_ROWS + key_row) * J + query_offsets) * 4 + seed_idx)
            )
            tl.store(routing_bias_ptr + bias_offset, routing_bias, mask=valid_query)

    @triton.jit
    def _hard_route_merge_c4_direct_route_kernel(
        query_ptr,
        seed_rows_ptr,
        key_ptr,
        value_ptr,
        merged_key_ptr,
        merged_value_ptr,
        routing_bias_ptr,
        A: tl.constexpr,
        F: tl.constexpr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        J: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROUTE_GROUPS: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_EXP2: tl.constexpr,
        FORCE_SEED_KEYS: tl.constexpr,
        LOGIT_INPUT_PRECISION: tl.constexpr,
        INLINE_LOCAL_SEED: tl.constexpr,
        USE_ASSIGNED_LOGIT: tl.constexpr,
        USE_MASK_ASSIGN: tl.constexpr,
        PACK_KV: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_col_block = tl.program_id(1)

        head = group_id % H
        tmp = group_id // H
        key_row = tmp % K_ROWS
        tmp = tmp // K_ROWS
        key_outer = tmp % F
        tmp = tmp // F
        query_outer = tmp % A
        batch = tmp // A

        query_offsets = query_col_block * BLOCK_J + tl.arange(0, BLOCK_J)
        key_offsets = tl.arange(0, BLOCK_L)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_query = query_offsets < J
        valid_key = key_offsets < L
        valid_dim = dim_offsets < D_HEAD

        key_base = key_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        key_block = tl.load(
            key_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        if INLINE_LOCAL_SEED:
            seed1 = tl.where(key_row > 0, key_row - 1, 1)
            seed2 = tl.where(
                key_row == 0,
                2,
                tl.where(key_row == Q_ROWS - 1, Q_ROWS - 3, key_row + 1),
            )
            seed3 = tl.where(
                key_row <= 1,
                3,
                tl.where(key_row >= Q_ROWS - 2, Q_ROWS - 4, key_row - 2),
            )
            seed_row0 = tl.full((BLOCK_J,), key_row, tl.int64)
            seed_row1 = tl.full((BLOCK_J,), seed1, tl.int64)
            seed_row2 = tl.full((BLOCK_J,), seed2, tl.int64)
            seed_row3 = tl.full((BLOCK_J,), seed3, tl.int64)
        else:
            seed_rows_base = (
                seed_rows_ptr
                + ((((query_outer * F + key_outer) * K_ROWS + key_row) * J + query_offsets) * 4)
            )
            seed_row0 = tl.load(seed_rows_base, mask=valid_query, other=0)
            seed_row1 = tl.load(seed_rows_base + 1, mask=valid_query, other=0)
            seed_row2 = tl.load(seed_rows_base + 2, mask=valid_query, other=0)
            seed_row3 = tl.load(seed_rows_base + 3, mask=valid_query, other=0)

        query_base = (batch * A + query_outer) * Q_ROWS * J * H * D_HEAD
        seed0 = tl.load(
            query_ptr
            + query_base
            + (((seed_row0[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed1 = tl.load(
            query_ptr
            + query_base
            + (((seed_row1[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed2 = tl.load(
            query_ptr
            + query_base
            + (((seed_row2[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed3 = tl.load(
            query_ptr
            + query_base
            + (((seed_row3[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )

        logits0 = tl.dot(seed0, tl.trans(key_block), input_precision=LOGIT_INPUT_PRECISION) * SCORE_SCALE
        logits1 = tl.dot(seed1, tl.trans(key_block), input_precision=LOGIT_INPUT_PRECISION) * SCORE_SCALE
        logits2 = tl.dot(seed2, tl.trans(key_block), input_precision=LOGIT_INPUT_PRECISION) * SCORE_SCALE
        logits3 = tl.dot(seed3, tl.trans(key_block), input_precision=LOGIT_INPUT_PRECISION) * SCORE_SCALE
        valid_logits = valid_query[:, None] & valid_key[None, :]
        logits0 = tl.where(valid_logits, logits0, -float("inf"))
        logits1 = tl.where(valid_logits, logits1, -float("inf"))
        logits2 = tl.where(valid_logits, logits2, -float("inf"))
        logits3 = tl.where(valid_logits, logits3, -float("inf"))

        if USE_MASK_ASSIGN:
            max01 = tl.maximum(logits0, logits1)
            max012 = tl.maximum(max01, logits2)
            best_logits = tl.maximum(max012, logits3)
            assigned0 = (logits1 <= logits0) & (logits2 <= logits0) & (logits3 <= logits0)
            assigned1 = (logits1 > logits0) & (logits2 <= logits1) & (logits3 <= logits1)
            assigned2 = (logits2 > max01) & (logits3 <= logits2)
            assigned3 = logits3 > max012
        else:
            best_logits = logits0
            assignments = tl.zeros((BLOCK_J, BLOCK_L), tl.int32)
            update = logits1 > best_logits
            best_logits = tl.where(update, logits1, best_logits)
            assignments = tl.where(update, 1, assignments)
            update = logits2 > best_logits
            best_logits = tl.where(update, logits2, best_logits)
            assignments = tl.where(update, 2, assignments)
            update = logits3 > best_logits
            assignments = tl.where(update, 3, assignments)
            if FORCE_SEED_KEYS:
                assignments = tl.where(
                    valid_logits & (key_offsets[None, :] < 4),
                    key_offsets[None, :],
                    assignments,
                )
        if USE_ASSIGNED_LOGIT:
            if USE_MASK_ASSIGN:
                assigned_logits = best_logits
            else:
                assigned_logits = tl.where(update, logits3, best_logits)
            if FORCE_SEED_KEYS:
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 0), logits0, assigned_logits)
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 1), logits1, assigned_logits)
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 2), logits2, assigned_logits)
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 3), logits3, assigned_logits)

        route_group = (((batch * H + head) * A + query_outer) * J + query_offsets)
        route_token_base = (key_outer * K_ROWS + key_row) * 4
        value_base = value_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        value_block = tl.load(
            value_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        for seed_idx in tl.static_range(0, 4):
            if USE_ASSIGNED_LOGIT:
                logits = assigned_logits
            else:
                logits = logits0
                if seed_idx == 1:
                    logits = logits1
                if seed_idx == 2:
                    logits = logits2
                if seed_idx == 3:
                    logits = logits3

            if USE_MASK_ASSIGN:
                assigned_to_seed = assigned0
                if seed_idx == 1:
                    assigned_to_seed = assigned1
                if seed_idx == 2:
                    assigned_to_seed = assigned2
                if seed_idx == 3:
                    assigned_to_seed = assigned3
            else:
                assigned_to_seed = assignments == seed_idx
            if FORCE_SEED_KEYS:
                if USE_MASK_ASSIGN:
                    forced_key = key_offsets[None, :] == seed_idx
                    not_forced_key = key_offsets[None, :] >= 4
                    active = valid_logits & (forced_key | (not_forced_key & assigned_to_seed))
                else:
                    active = valid_logits & assigned_to_seed
            else:
                has_seed = tl.sum(tl.where(assigned_to_seed & valid_logits, 1, 0), axis=1) > 0
                forced_first_key = (~has_seed[:, None]) & (key_offsets[None, :] == 0)
                active = valid_logits & (assigned_to_seed | forced_first_key)

            row_max = tl.max(tl.where(active, logits, -float("inf")), axis=1)
            if USE_EXP2:
                probs = tl.where(
                    active,
                    tl.exp2((logits - row_max[:, None]) * 1.4426950408889634),
                    0.0,
                )
            else:
                probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            if FORCE_SEED_KEYS:
                safe_denom = tl.where(valid_query, denom, 1.0)
            else:
                safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weights = probs / safe_denom[:, None]
            if USE_EXP2:
                log_denom = row_max + tl.log2(safe_denom) * 0.6931471805599453
            else:
                log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            expected_logit = tl.sum(tl.where(active, weights * logits, 0.0), axis=1)
            routing_bias = expected_logit - log_denom
            if not FORCE_SEED_KEYS:
                has_keys = tl.sum(tl.where(active, 1, 0), axis=1) > 0
                routing_bias = tl.where(has_keys, routing_bias, 0.0)

            route_token = route_token_base + seed_idx
            if PACK_KV:
                route_offset = (
                    (route_token * ROUTE_GROUPS + route_group[:, None]) * (2 * D_HEAD)
                    + dim_offsets[None, :]
                )
                tl.store(
                    merged_key_ptr + route_offset,
                    merged_key,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
                tl.store(
                    merged_key_ptr + route_offset + D_HEAD,
                    merged_value,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
            else:
                route_offset = (
                    (route_token * ROUTE_GROUPS + route_group[:, None]) * D_HEAD
                    + dim_offsets[None, :]
                )
                tl.store(
                    merged_key_ptr + route_offset,
                    merged_key,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
                tl.store(
                    merged_value_ptr + route_offset,
                    merged_value,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
            bias_offset = route_token * ROUTE_GROUPS + route_group
            tl.store(routing_bias_ptr + bias_offset, routing_bias, mask=valid_query)

    @triton.jit
    def _hard_route_merge_c4_stripe_direct_route_kernel(
        query_ptr,
        seed_rows_ptr,
        key_ptr,
        value_ptr,
        merged_key_ptr,
        merged_value_ptr,
        routing_bias_ptr,
        A: tl.constexpr,
        F: tl.constexpr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        J: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROUTE_GROUPS: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_EXP2: tl.constexpr,
        INLINE_LOCAL_SEED: tl.constexpr,
        PACK_KV: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_col_block = tl.program_id(1)

        head = group_id % H
        tmp = group_id // H
        key_row = tmp % K_ROWS
        tmp = tmp // K_ROWS
        key_outer = tmp % F
        tmp = tmp // F
        query_outer = tmp % A
        batch = tmp // A

        query_offsets = query_col_block * BLOCK_J + tl.arange(0, BLOCK_J)
        stripe_offsets = tl.arange(0, BLOCK_L)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_query = query_offsets < J
        valid_dim = dim_offsets < D_HEAD

        if INLINE_LOCAL_SEED:
            seed1 = tl.where(key_row > 0, key_row - 1, 1)
            seed2 = tl.where(
                key_row == 0,
                2,
                tl.where(key_row == Q_ROWS - 1, Q_ROWS - 3, key_row + 1),
            )
            seed3 = tl.where(
                key_row <= 1,
                3,
                tl.where(key_row >= Q_ROWS - 2, Q_ROWS - 4, key_row - 2),
            )
            seed_row0 = tl.full((BLOCK_J,), key_row, tl.int64)
            seed_row1 = tl.full((BLOCK_J,), seed1, tl.int64)
            seed_row2 = tl.full((BLOCK_J,), seed2, tl.int64)
            seed_row3 = tl.full((BLOCK_J,), seed3, tl.int64)
        else:
            seed_rows_base = (
                seed_rows_ptr
                + ((((query_outer * F + key_outer) * K_ROWS + key_row) * J + query_offsets) * 4)
            )
            seed_row0 = tl.load(seed_rows_base, mask=valid_query, other=0)
            seed_row1 = tl.load(seed_rows_base + 1, mask=valid_query, other=0)
            seed_row2 = tl.load(seed_rows_base + 2, mask=valid_query, other=0)
            seed_row3 = tl.load(seed_rows_base + 3, mask=valid_query, other=0)

        query_base = (batch * A + query_outer) * Q_ROWS * J * H * D_HEAD
        route_group = (((batch * H + head) * A + query_outer) * J + query_offsets)
        route_token_base = (key_outer * K_ROWS + key_row) * 4
        key_base = key_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
        value_base = value_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD

        for seed_idx in tl.static_range(0, 4):
            seed_row = seed_row0
            if seed_idx == 1:
                seed_row = seed_row1
            if seed_idx == 2:
                seed_row = seed_row2
            if seed_idx == 3:
                seed_row = seed_row3

            stripe_start = (seed_idx * L) // 4
            stripe_end = ((seed_idx + 1) * L) // 4
            stripe_len = stripe_end - stripe_start
            key_offsets = stripe_start + stripe_offsets
            valid_key = stripe_offsets < stripe_len

            seed = tl.load(
                query_ptr
                + query_base
                + (((seed_row[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
                mask=valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            key_block = tl.load(
                key_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
                mask=valid_key[:, None] & valid_dim[None, :],
                other=0.0,
            )
            value_block = tl.load(
                value_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
                mask=valid_key[:, None] & valid_dim[None, :],
                other=0.0,
            )
            logits = tl.dot(seed, tl.trans(key_block), input_precision="tf32") * SCORE_SCALE
            active = valid_query[:, None] & valid_key[None, :]

            row_max = tl.max(tl.where(active, logits, -float("inf")), axis=1)
            if USE_EXP2:
                probs = tl.where(
                    active,
                    tl.exp2((logits - row_max[:, None]) * 1.4426950408889634),
                    0.0,
                )
            else:
                probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(valid_query, denom, 1.0)
            weights = probs / safe_denom[:, None]
            if USE_EXP2:
                log_denom = row_max + tl.log2(safe_denom) * 0.6931471805599453
            else:
                log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            expected_logit = tl.sum(tl.where(active, weights * logits, 0.0), axis=1)
            routing_bias = expected_logit - log_denom

            route_token = route_token_base + seed_idx
            if PACK_KV:
                route_offset = (
                    (route_token * ROUTE_GROUPS + route_group[:, None]) * (2 * D_HEAD)
                    + dim_offsets[None, :]
                )
                tl.store(
                    merged_key_ptr + route_offset,
                    merged_key,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
                tl.store(
                    merged_key_ptr + route_offset + D_HEAD,
                    merged_value,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
            else:
                route_offset = (
                    (route_token * ROUTE_GROUPS + route_group[:, None]) * D_HEAD
                    + dim_offsets[None, :]
                )
                tl.store(
                    merged_key_ptr + route_offset,
                    merged_key,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
                tl.store(
                    merged_value_ptr + route_offset,
                    merged_value,
                    mask=valid_query[:, None] & valid_dim[None, :],
                )
            bias_offset = route_token * ROUTE_GROUPS + route_group
            tl.store(routing_bias_ptr + bias_offset, routing_bias, mask=valid_query)

    @triton.jit
    def _hard_route_merge_c4_direct_route_keyouter_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        merged_key_ptr,
        merged_value_ptr,
        routing_bias_ptr,
        A: tl.constexpr,
        F: tl.constexpr,
        F_GROUPS: tl.constexpr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        J: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROUTE_GROUPS: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        KEY_OUTER_BLOCK: tl.constexpr,
        USE_ASSIGNED_LOGIT: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_col_block = tl.program_id(1)

        head = group_id % H
        tmp = group_id // H
        key_row = tmp % K_ROWS
        tmp = tmp // K_ROWS
        key_outer_group = tmp % F_GROUPS
        tmp = tmp // F_GROUPS
        query_outer = tmp % A
        batch = tmp // A

        query_offsets = query_col_block * BLOCK_J + tl.arange(0, BLOCK_J)
        key_offsets = tl.arange(0, BLOCK_L)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_query = query_offsets < J
        valid_key = key_offsets < L
        valid_dim = dim_offsets < D_HEAD

        seed1_row = tl.where(key_row > 0, key_row - 1, 1)
        seed2_row = tl.where(
            key_row == 0,
            2,
            tl.where(key_row == Q_ROWS - 1, Q_ROWS - 3, key_row + 1),
        )
        seed3_row = tl.where(
            key_row <= 1,
            3,
            tl.where(key_row >= Q_ROWS - 2, Q_ROWS - 4, key_row - 2),
        )
        seed_row0 = tl.full((BLOCK_J,), key_row, tl.int64)
        seed_row1 = tl.full((BLOCK_J,), seed1_row, tl.int64)
        seed_row2 = tl.full((BLOCK_J,), seed2_row, tl.int64)
        seed_row3 = tl.full((BLOCK_J,), seed3_row, tl.int64)

        query_base = (batch * A + query_outer) * Q_ROWS * J * H * D_HEAD
        seed0 = tl.load(
            query_ptr
            + query_base
            + (((seed_row0[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed1 = tl.load(
            query_ptr
            + query_base
            + (((seed_row1[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed2 = tl.load(
            query_ptr
            + query_base
            + (((seed_row2[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed3 = tl.load(
            query_ptr
            + query_base
            + (((seed_row3[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )

        route_group = (((batch * H + head) * A + query_outer) * J + query_offsets)
        for key_outer_delta in tl.static_range(0, KEY_OUTER_BLOCK):
            key_outer = key_outer_group * KEY_OUTER_BLOCK + key_outer_delta
            valid_outer = key_outer < F

            key_base = (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
            key_block = tl.load(
                key_ptr + key_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
                mask=valid_outer & valid_key[:, None] & valid_dim[None, :],
                other=0.0,
            )
            logits0 = tl.dot(seed0, tl.trans(key_block), input_precision="tf32") * SCORE_SCALE
            logits1 = tl.dot(seed1, tl.trans(key_block), input_precision="tf32") * SCORE_SCALE
            logits2 = tl.dot(seed2, tl.trans(key_block), input_precision="tf32") * SCORE_SCALE
            logits3 = tl.dot(seed3, tl.trans(key_block), input_precision="tf32") * SCORE_SCALE
            valid_logits = valid_outer & valid_query[:, None] & valid_key[None, :]
            logits0 = tl.where(valid_logits, logits0, -float("inf"))
            logits1 = tl.where(valid_logits, logits1, -float("inf"))
            logits2 = tl.where(valid_logits, logits2, -float("inf"))
            logits3 = tl.where(valid_logits, logits3, -float("inf"))

            best_logits = logits0
            assignments = tl.zeros((BLOCK_J, BLOCK_L), tl.int32)
            update = logits1 > best_logits
            best_logits = tl.where(update, logits1, best_logits)
            assignments = tl.where(update, 1, assignments)
            update = logits2 > best_logits
            best_logits = tl.where(update, logits2, best_logits)
            assignments = tl.where(update, 2, assignments)
            update = logits3 > best_logits
            assignments = tl.where(update, 3, assignments)
            assignments = tl.where(
                valid_logits & (key_offsets[None, :] < 4),
                key_offsets[None, :],
                assignments,
            )
            if USE_ASSIGNED_LOGIT:
                assigned_logits = tl.where(update, logits3, best_logits)
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 0), logits0, assigned_logits)
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 1), logits1, assigned_logits)
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 2), logits2, assigned_logits)
                assigned_logits = tl.where(valid_logits & (key_offsets[None, :] == 3), logits3, assigned_logits)

            route_token_base = (key_outer * K_ROWS + key_row) * 4
            value_block = tl.load(
                value_ptr + key_base + key_offsets[:, None] * H * D_HEAD + dim_offsets[None, :],
                mask=valid_outer & valid_key[:, None] & valid_dim[None, :],
                other=0.0,
            )
            for seed_idx in tl.static_range(0, 4):
                if USE_ASSIGNED_LOGIT:
                    logits = assigned_logits
                else:
                    logits = logits0
                    if seed_idx == 1:
                        logits = logits1
                    if seed_idx == 2:
                        logits = logits2
                    if seed_idx == 3:
                        logits = logits3

                active = valid_logits & (assignments == seed_idx)
                row_max = tl.max(tl.where(active, logits, -float("inf")), axis=1)
                probs = tl.where(
                    active,
                    tl.exp2((logits - row_max[:, None]) * 1.4426950408889634),
                    0.0,
                )
                denom = tl.sum(probs, axis=1)
                safe_denom = tl.where(valid_outer & valid_query, denom, 1.0)
                weights = probs / safe_denom[:, None]
                log_denom = row_max + tl.log2(safe_denom) * 0.6931471805599453

                merged_key = tl.dot(weights.to(key_block.dtype), key_block)
                merged_value = tl.dot(weights.to(value_block.dtype), value_block)
                expected_logit = tl.sum(tl.where(active, weights * logits, 0.0), axis=1)
                routing_bias = expected_logit - log_denom

                route_token = route_token_base + seed_idx
                route_offset = (
                    (route_token * ROUTE_GROUPS + route_group[:, None]) * D_HEAD
                    + dim_offsets[None, :]
                )
                tl.store(
                    merged_key_ptr + route_offset,
                    merged_key,
                    mask=valid_outer & valid_query[:, None] & valid_dim[None, :],
                )
                tl.store(
                    merged_value_ptr + route_offset,
                    merged_value,
                    mask=valid_outer & valid_query[:, None] & valid_dim[None, :],
                )
                bias_offset = route_token * ROUTE_GROUPS + route_group
                tl.store(routing_bias_ptr + bias_offset, routing_bias, mask=valid_outer & valid_query)

    @triton.jit
    def _biased_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        out_ptr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_block_id = tl.program_id(1)

        query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        query_base = query_ptr + group_id * Q_LEN * D_HEAD
        key_base = key_ptr + group_id * K_LEN * D_HEAD
        value_base = value_ptr + group_id * K_LEN * D_HEAD
        bias_base = bias_ptr + group_id * K_LEN
        out_base = out_ptr + group_id * Q_LEN * D_HEAD

        q = tl.load(
            query_base + query_offsets[:, None] * D_HEAD + dim_offsets[None, :],
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
            other=0.0,
        )

        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, K_LEN, BLOCK_N):
            keys = key_start + key_offsets
            k = tl.load(
                key_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            bias = tl.load(bias_base + keys, mask=keys < K_LEN, other=float("inf"))

            logits = tl.dot(q, tl.trans(k)) * SCORE_SCALE - bias[None, :]
            logits = tl.where(
                (query_offsets[:, None] < Q_LEN) & (keys[None, :] < K_LEN),
                logits,
                -float("inf"),
            )

            block_max = tl.max(logits, axis=1)
            next_max = tl.maximum(max_score, block_max)
            if USE_EXP2:
                p = tl.exp2((logits - next_max[:, None]) * 1.4426950408889634)
                alpha = tl.exp2((max_score - next_max) * 1.4426950408889634)
            else:
                p = tl.exp(logits - next_max[:, None])
                alpha = tl.exp(max_score - next_max)

            v = tl.load(
                value_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            max_score = next_max

        out = acc / normalizer[:, None]
        tl.store(
            out_base + query_offsets[:, None] * D_HEAD + dim_offsets[None, :],
            out,
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
        )

    @triton.jit
    def _direct_route_biased_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        out_ptr,
        A_OUTER: tl.constexpr,
        Q_ROWS: tl.constexpr,
        J_COLS: tl.constexpr,
        H_HEADS: tl.constexpr,
        K_LEN: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_EXP2: tl.constexpr,
        CAST_Q_TO_FP16: tl.constexpr,
        GROUPS: tl.constexpr,
        PACKED_KV: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_block_id = tl.program_id(1)

        query_col = group_id % J_COLS
        tmp = group_id // J_COLS
        query_outer = tmp % A_OUTER
        tmp = tmp // A_OUTER
        head = tmp % H_HEADS
        batch = tmp // H_HEADS

        query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        query_base = query_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD
        out_base = out_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD

        q = tl.load(
            query_base
            + (((query_outer * Q_ROWS + query_offsets[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
            mask=(query_offsets[:, None] < Q_ROWS) & (dim_offsets[None, :] < D_HEAD),
            other=0.0,
        )
        q_dot = q
        if CAST_Q_TO_FP16:
            q_dot = q.to(tl.float16)

        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, K_LEN, BLOCK_N):
            keys = key_start + key_offsets
            if PACKED_KV:
                packed_base = (keys[:, None] * GROUPS + group_id) * (2 * D_HEAD)
                k = tl.load(
                    key_ptr + packed_base + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
            else:
                k = tl.load(
                    key_ptr + (keys[:, None] * GROUPS + group_id) * D_HEAD + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
            bias = tl.load(bias_ptr + keys * GROUPS + group_id, mask=keys < K_LEN, other=float("inf"))

            logits = tl.dot(q_dot, tl.trans(k)) * SCORE_SCALE - bias[None, :]
            logits = tl.where(
                (query_offsets[:, None] < Q_ROWS) & (keys[None, :] < K_LEN),
                logits,
                -float("inf"),
            )

            block_max = tl.max(logits, axis=1)
            next_max = tl.maximum(max_score, block_max)
            if USE_EXP2:
                p = tl.exp2((logits - next_max[:, None]) * 1.4426950408889634)
                alpha = tl.exp2((max_score - next_max) * 1.4426950408889634)
            else:
                p = tl.exp(logits - next_max[:, None])
                alpha = tl.exp(max_score - next_max)

            if PACKED_KV:
                v = tl.load(
                    key_ptr + packed_base + D_HEAD + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
            else:
                v = tl.load(
                    value_ptr + (keys[:, None] * GROUPS + group_id) * D_HEAD + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            max_score = next_max

        out = acc / normalizer[:, None]
        tl.store(
            out_base
            + (((query_outer * Q_ROWS + query_offsets[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
            out,
            mask=(query_offsets[:, None] < Q_ROWS) & (dim_offsets[None, :] < D_HEAD),
        )


def triton_biased_attention(
    query,
    key,
    value,
    bias,
    block_m=16,
    block_n=128,
    scale=1.0,
    num_warps=4,
    num_stages=None,
    use_exp2=False,
):
    """Fused Triton attention for hard-route compressed KV routing.

    query: [groups, query_tokens, dim]
    key/value: [groups, compressed_tokens, dim]
    bias: [groups, compressed_tokens]

    The computed logits are scale * query @ key.T - bias, matching biased_sdpa.
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_backend='triton' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton' requires fp16 or bf16 query tensors.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    bias = bias.contiguous()

    groups, query_len, dim = query.shape
    _, key_len, key_dim = key.shape
    if key_dim != dim or value.shape != key.shape or bias.shape != (groups, key_len):
        raise ValueError("query/key/value/bias shapes are inconsistent for Triton hard-route attention.")

    block_d = triton.next_power_of_2(dim)
    if block_d > 128:
        raise ValueError(f"Triton hard-route attention supports head dim <= 128, got {dim}.")

    out = _cached_empty("grid_biased_attention_out", query.shape, query.device, query.dtype)
    grid = (groups, triton.cdiv(query_len, block_m))
    meta = {"num_warps": int(num_warps)}
    if num_stages is not None:
        meta["num_stages"] = int(num_stages)
    _biased_attention_kernel[grid](
        query,
        key,
        value,
        bias,
        out,
        Q_LEN=query_len,
        K_LEN=key_len,
        D_HEAD=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_EXP2=bool(use_exp2),
        **meta,
    )
    return out


def triton_direct_route_biased_attention(
    query,
    key,
    value,
    bias,
    block_m=32,
    block_n=64,
    scale=1.0,
    num_warps=1,
    num_stages=1,
    use_exp2=True,
    out=None,
    assume_contiguous=False,
):
    """Fused routing attention that reads/writes hard-route block layout.

    query/out: [B, A, query_rows, query_columns, heads, dim]
    key/value: [compressed_tokens, B*heads*A*query_columns, dim]
    bias:      [compressed_tokens, B*heads*A*query_columns]
    """
    if triton is None:
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires fp16 or bf16 tensors.")

    if not assume_contiguous:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        bias = bias.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    key_len, groups, key_dim = key.shape
    expected_groups = batch * heads * query_outer * query_columns
    if groups != expected_groups:
        raise ValueError(
            "direct-route attention expects route-layout KV with "
            f"{expected_groups} groups, got {groups}."
        )
    if key_dim != dim or value.shape != key.shape or bias.shape != (key_len, groups):
        raise ValueError("query/key/value/bias shapes are inconsistent for direct-route attention.")

    block_d = triton.next_power_of_2(dim)
    if block_d > 128:
        raise ValueError(f"Triton hard-route attention supports head dim <= 128, got {dim}.")

    if out is None:
        out = _cached_empty("direct_route_attention_out", query.shape, query.device, query.dtype)
    else:
        if out.shape != query.shape or out.device != query.device or out.dtype != query.dtype:
            raise ValueError("direct-route attention output tensor has inconsistent shape/device/dtype.")
        if not out.is_contiguous():
            raise ValueError("direct-route attention output tensor must be contiguous.")
    grid = (groups, triton.cdiv(query_rows, block_m))
    meta = {"num_warps": int(num_warps)}
    if num_stages is not None:
        meta["num_stages"] = int(num_stages)
    _direct_route_biased_attention_kernel[grid](
        query,
        key,
        value,
        bias,
        out,
        A_OUTER=query_outer,
        Q_ROWS=query_rows,
        J_COLS=query_columns,
        H_HEADS=heads,
        K_LEN=key_len,
        D_HEAD=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_EXP2=bool(use_exp2),
        CAST_Q_TO_FP16=bool(key.dtype == torch.float16 and query.dtype != torch.float16),
        GROUPS=groups,
        PACKED_KV=False,
        **meta,
    )
    return out


def triton_direct_route_biased_attention_packed_kv(
    query,
    packed_kv,
    bias,
    block_m=32,
    block_n=64,
    scale=1.0,
    num_warps=1,
    num_stages=1,
    use_exp2=True,
    out=None,
    assume_contiguous=False,
):
    """Direct-route attention for packed [K|V] token-major intermediates.

    query/out: [B, A, query_rows, query_columns, heads, dim]
    packed_kv: [compressed_tokens, B*heads*A*query_columns, 2*dim]
    bias:      [compressed_tokens, B*heads*A*query_columns]
    """
    if triton is None:
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires fp16 or bf16 tensors.")

    if not assume_contiguous:
        query = query.contiguous()
        packed_kv = packed_kv.contiguous()
        bias = bias.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    key_len, groups, packed_dim = packed_kv.shape
    expected_groups = batch * heads * query_outer * query_columns
    if groups != expected_groups:
        raise ValueError(
            "direct-route attention expects route-layout packed KV with "
            f"{expected_groups} groups, got {groups}."
        )
    if packed_dim != 2 * dim or bias.shape != (key_len, groups):
        raise ValueError("query/packed_kv/bias shapes are inconsistent for direct-route attention.")

    block_d = triton.next_power_of_2(dim)
    if block_d > 128:
        raise ValueError(f"Triton hard-route attention supports head dim <= 128, got {dim}.")

    if out is None:
        out = _cached_empty("direct_route_attention_out", query.shape, query.device, query.dtype)
    else:
        if out.shape != query.shape or out.device != query.device or out.dtype != query.dtype:
            raise ValueError("direct-route attention output tensor has inconsistent shape/device/dtype.")
        if not out.is_contiguous():
            raise ValueError("direct-route attention output tensor must be contiguous.")

    grid = (groups, triton.cdiv(query_rows, block_m))
    meta = {"num_warps": int(num_warps)}
    if num_stages is not None:
        meta["num_stages"] = int(num_stages)
    _direct_route_biased_attention_kernel[grid](
        query,
        packed_kv,
        packed_kv,
        bias,
        out,
        A_OUTER=query_outer,
        Q_ROWS=query_rows,
        J_COLS=query_columns,
        H_HEADS=heads,
        K_LEN=key_len,
        D_HEAD=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_EXP2=bool(use_exp2),
        CAST_Q_TO_FP16=bool(packed_kv.dtype == torch.float16 and query.dtype != torch.float16),
        GROUPS=groups,
        PACKED_KV=True,
        **meta,
    )
    return out


def triton_hard_route_merge(seed_queries, key, value, scale=1.0):
    """Fuse hard assignment, per-cluster softmax, KV merge, and routing bias.

    seed_queries: [B, A, F, K, J, C, H, D]
    key/value:    [B, F, K, L, H, D]

    Returns:
        merged_key/value: [B, A, F, J, K, C, H, D]
        routing_bias:     [B, H, A, F, K, J, C]
        cluster_has_keys: [B, H, A, F, K, J, C]
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton' requires Triton.")
    if not seed_queries.is_cuda:
        raise RuntimeError("hard_route_backend='triton' requires CUDA/ROCm tensors.")
    if seed_queries.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton' requires fp16 or bf16 tensors.")

    seed_queries = seed_queries.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, key_outer, key_rows, query_columns, num_seeds, heads, dim = seed_queries.shape
    if key.shape[:3] != (batch, key_outer, key_rows) or value.shape != key.shape:
        raise ValueError("seed_queries/key/value shapes are inconsistent for Triton hard-route merge.")
    key_columns = key.size(3)
    if key.size(4) != heads or key.size(5) != dim:
        raise ValueError("seed_queries/key head dimensions are inconsistent for Triton hard-route merge.")

    block_c = triton.next_power_of_2(num_seeds)
    target_block_j = 16 if key_columns <= 32 else 8
    block_j = min(target_block_j, triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    if block_c > 16:
        raise ValueError(f"Triton hard-route merge supports at most 16 seeds, got {num_seeds}.")
    if block_l > 128:
        raise ValueError(f"Triton hard-route merge supports at most 128 key tokens per block, got {key_columns}.")
    if block_d > 128:
        raise ValueError(f"Triton hard-route merge supports head dim <= 128, got {dim}.")

    output_elements = batch * query_outer * key_outer * query_columns * key_rows * num_seeds * heads * dim
    if output_elements > torch.iinfo(torch.int32).max:
        raise ValueError(
            "Triton hard-route merge output is too large for one launch. "
            "Reduce query_outer_chunk; for default 480p C=4 use query_outer_chunk<=10."
        )

    merged_key = torch.empty(
        (batch, query_outer, key_outer, query_columns, key_rows, num_seeds, heads, dim),
        device=key.device,
        dtype=key.dtype,
    )
    merged_value = torch.empty_like(merged_key)
    routing_bias = torch.empty(
        (batch, heads, query_outer, key_outer, key_rows, query_columns, num_seeds),
        device=key.device,
        dtype=torch.float32,
    )
    grid = (
        batch * query_outer * key_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    _hard_route_merge_kernel[grid](
        seed_queries,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias,
        A=query_outer,
        F=key_outer,
        K_ROWS=key_rows,
        J=query_columns,
        C_SEEDS=num_seeds,
        H=heads,
        L=key_columns,
        D_HEAD=dim,
        BLOCK_C=block_c,
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_EXP2=False,
        num_warps=4,
    )
    return merged_key, merged_value, routing_bias, None


def triton_hard_route_merge_c4(seed_queries, key, value, scale=1.0, block_j_target=8):
    """C=4 specialized hard-route merge that reuses seed logits."""
    if triton is None:
        raise RuntimeError("hard_route_backend='triton_c4' requires Triton.")
    if not seed_queries.is_cuda:
        raise RuntimeError("hard_route_backend='triton_c4' requires CUDA/ROCm tensors.")
    if seed_queries.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton_c4' requires fp16 or bf16 tensors.")

    seed_queries = seed_queries.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, key_outer, key_rows, query_columns, num_seeds, heads, dim = seed_queries.shape
    if num_seeds != 4:
        raise ValueError(f"triton_c4 requires exactly 4 seeds, got {num_seeds}.")
    if key.shape[:3] != (batch, key_outer, key_rows) or value.shape != key.shape:
        raise ValueError("seed_queries/key/value shapes are inconsistent for Triton hard-route merge.")
    key_columns = key.size(3)
    if key.size(4) != heads or key.size(5) != dim:
        raise ValueError("seed_queries/key head dimensions are inconsistent for Triton hard-route merge.")

    block_j = min(int(block_j_target), triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    if block_l > 128:
        raise ValueError(f"Triton hard-route merge supports at most 128 key tokens per block, got {key_columns}.")
    if block_d > 128:
        raise ValueError(f"Triton hard-route merge supports head dim <= 128, got {dim}.")

    output_elements = batch * query_outer * key_outer * query_columns * key_rows * num_seeds * heads * dim
    if output_elements > torch.iinfo(torch.int32).max:
        raise ValueError(
            "Triton hard-route merge output is too large for one launch. "
            "Reduce query_outer_chunk; for default 480p C=4 use query_outer_chunk<=10."
        )

    merged_key = torch.empty(
        (batch, query_outer, key_outer, query_columns, key_rows, num_seeds, heads, dim),
        device=key.device,
        dtype=key.dtype,
    )
    merged_value = torch.empty_like(merged_key)
    routing_bias = torch.empty(
        (batch, heads, query_outer, key_outer, key_rows, query_columns, num_seeds),
        device=key.device,
        dtype=torch.float32,
    )

    grid = (
        batch * query_outer * key_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    _hard_route_merge_c4_kernel[grid](
        seed_queries,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias,
        A=query_outer,
        F=key_outer,
        K_ROWS=key_rows,
        J=query_columns,
        H=heads,
        L=key_columns,
        D_HEAD=dim,
        C_SEEDS=4,
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_EXP2=False,
        num_warps=4,
    )
    return merged_key, merged_value, routing_bias, None


def triton_hard_route_merge_c4_direct_route(
    query,
    seed_rows,
    key,
    value,
    scale=1.0,
    block_j_target=16,
    num_warps=2,
    bias_dtype=torch.float32,
    merged_dtype=None,
    merged_key_dtype=None,
    merged_value_dtype=None,
    use_exp2=False,
    force_seed_keys=False,
    num_stages=None,
    logit_input_precision="ieee",
    inline_local_seed=False,
    key_outer_block=1,
    use_key_outer_kernel=False,
    use_assigned_logit=False,
    use_mask_assign=False,
    pack_kv=False,
    use_stripe_assign=False,
    assume_contiguous=False,
):
    """Exact non-shared C=4 merge that writes routing-ready K/V/bias.

    query:     [B, A, query_rows, query_columns, H, D]
    seed_rows: [A, F, key_rows, query_columns, 4], or None when
        inline_local_seed=True.
    key/value: [B, F, key_rows, key_columns, H, D]

    Returns token-major key/value [F*key_rows*4, B*H*A*query_columns, D]
    and bias [F*key_rows*4, B*H*A*query_columns], matching the current
    direct-route attention kernel.
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires fp16 or bf16 tensors.")

    if not assume_contiguous:
        query = query.contiguous()
    if seed_rows is None:
        if not inline_local_seed:
            raise ValueError("seed_rows is required unless inline_local_seed=True.")
        # The INLINE_LOCAL_SEED specialization never reads seed_rows_ptr.
        # Reuse query as a harmless placeholder and avoid an extra cache lookup.
        seed_rows = query
    else:
        if not assume_contiguous:
            seed_rows = seed_rows.contiguous()
    if not assume_contiguous:
        key = key.contiguous()
        value = value.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    key_outer = key.size(1)
    key_rows = key.size(2)
    key_columns = key.size(3)
    key_heads = key.size(4)
    key_dim = key.size(5)
    if not inline_local_seed and seed_rows.shape != (query_outer, key_outer, key_rows, query_columns, 4):
        raise ValueError("seed_rows shape is inconsistent for non-shared C4 direct-route merge.")
    if key.shape[:3] != (batch, key_outer, key_rows) or value.shape != key.shape:
        raise ValueError("query/key/value shapes are inconsistent for non-shared C4 direct-route merge.")
    if key_heads != heads or key_dim != dim:
        raise ValueError("query/key head dimensions are inconsistent for non-shared C4 direct-route merge.")

    block_j = min(int(block_j_target), triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    if block_l > 128:
        raise ValueError(f"Triton hard-route merge supports at most 128 key tokens per block, got {key_columns}.")
    if block_d > 128:
        raise ValueError(f"Triton hard-route merge supports head dim <= 128, got {dim}.")

    route_groups = batch * heads * query_outer * query_columns
    route_tokens = key_outer * key_rows * 4
    if merged_dtype is None:
        merged_dtype = key.dtype
    if merged_key_dtype is None:
        merged_key_dtype = merged_dtype
    if merged_value_dtype is None:
        merged_value_dtype = merged_dtype
    if pack_kv and merged_key_dtype != merged_value_dtype:
        raise ValueError("pack_kv requires merged_key_dtype and merged_value_dtype to match.")
    if pack_kv:
        merged_key = _cached_empty(
            "hard_route_c4_direct_route_packed_kv_token_major",
            (route_tokens, route_groups, 2 * dim),
            key.device,
            merged_key_dtype,
        )
        merged_value = merged_key
    else:
        merged_key = _cached_empty(
            "hard_route_c4_direct_route_key_token_major",
            (route_tokens, route_groups, dim),
            key.device,
            merged_key_dtype,
        )
        merged_value = _cached_empty(
            "hard_route_c4_direct_route_value_token_major",
            (route_tokens, route_groups, dim),
            key.device,
            merged_value_dtype,
        )
    routing_bias = _cached_empty(
        "hard_route_c4_direct_route_bias_token_major",
        (route_tokens, route_groups),
        key.device,
        bias_dtype,
    )

    key_outer_block = int(key_outer_block)
    if key_outer_block not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError(f"key_outer_block must be between 1 and 7, got {key_outer_block}.")
    use_key_outer_kernel = bool(use_key_outer_kernel or key_outer_block != 1)
    if use_assigned_logit and not force_seed_keys:
        raise ValueError("use_assigned_logit currently requires force_seed_keys=True.")
    if use_mask_assign and not force_seed_keys:
        raise ValueError("use_mask_assign currently requires force_seed_keys=True.")
    if use_mask_assign and use_key_outer_kernel:
        raise ValueError("use_mask_assign is only implemented for the non-keyouter direct C4 kernel.")
    if pack_kv and use_key_outer_kernel:
        raise ValueError("pack_kv is only implemented for the non-keyouter direct C4 kernel.")
    if use_stripe_assign and use_key_outer_kernel:
        raise ValueError("stripe C4 merge is only implemented for the non-keyouter direct C4 kernel.")
    if use_stripe_assign and (use_assigned_logit or use_mask_assign):
        raise ValueError("stripe C4 merge replaces hard assignment; do not combine it with bestlogit or maskassign.")
    if use_stripe_assign and key_columns < 4:
        raise ValueError("stripe C4 merge requires at least four key columns.")
    if use_key_outer_kernel and (
        (not inline_local_seed)
        or block_j != 32
        or int(num_warps) != 2
        or bias_dtype not in (torch.float16, torch.bfloat16)
        or merged_key_dtype not in (torch.float16, torch.bfloat16)
        or merged_value_dtype not in (torch.float16, torch.bfloat16)
        or (not use_exp2)
        or (not force_seed_keys)
        or num_stages != 1
        or logit_input_precision != "tf32"
    ):
        raise ValueError(
            "key-outer direct-route merge is specialized for the current token-major "
            "inline-local-seed hard C4 fast path."
        )

    if use_stripe_assign:
        stripe_block_l = triton.next_power_of_2((key_columns + 3) // 4)
        grid = (
            batch * query_outer * key_outer * key_rows * heads,
            triton.cdiv(query_columns, block_j),
        )
        _hard_route_merge_c4_stripe_direct_route_kernel[grid](
            query,
            seed_rows,
            key,
            value,
            merged_key,
            merged_value,
            routing_bias,
            A=query_outer,
            F=key_outer,
            Q_ROWS=query_rows,
            K_ROWS=key_rows,
            J=query_columns,
            H=heads,
            L=key_columns,
            D_HEAD=dim,
            ROUTE_GROUPS=route_groups,
            BLOCK_J=block_j,
            BLOCK_L=stripe_block_l,
            BLOCK_D=block_d,
            SCORE_SCALE=float(scale),
            USE_EXP2=bool(use_exp2),
            INLINE_LOCAL_SEED=bool(inline_local_seed),
            PACK_KV=bool(pack_kv),
            num_warps=int(num_warps),
            num_stages=int(num_stages) if num_stages is not None else 3,
        )
        return merged_key, merged_value, routing_bias, None

    if use_key_outer_kernel:
        key_outer_groups = triton.cdiv(key_outer, key_outer_block)
        grid = (
            batch * query_outer * key_outer_groups * key_rows * heads,
            triton.cdiv(query_columns, block_j),
        )
        _hard_route_merge_c4_direct_route_keyouter_kernel[grid](
            query,
            key,
            value,
            merged_key,
            merged_value,
            routing_bias,
            A=query_outer,
            F=key_outer,
            F_GROUPS=key_outer_groups,
            Q_ROWS=query_rows,
            K_ROWS=key_rows,
            J=query_columns,
            H=heads,
            L=key_columns,
            D_HEAD=dim,
            ROUTE_GROUPS=route_groups,
            BLOCK_J=block_j,
            BLOCK_L=block_l,
            BLOCK_D=block_d,
            SCORE_SCALE=float(scale),
            KEY_OUTER_BLOCK=key_outer_block,
            USE_ASSIGNED_LOGIT=bool(use_assigned_logit),
            num_warps=2,
            num_stages=1,
        )
        return merged_key, merged_value, routing_bias, None

    grid = (
        batch * query_outer * key_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    launch_kwargs = {"num_warps": int(num_warps)}
    if num_stages is not None:
        launch_kwargs["num_stages"] = int(num_stages)
    _hard_route_merge_c4_direct_route_kernel[grid](
        query,
        seed_rows,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias,
        A=query_outer,
        F=key_outer,
        Q_ROWS=query_rows,
        K_ROWS=key_rows,
        J=query_columns,
        H=heads,
        L=key_columns,
        D_HEAD=dim,
        ROUTE_GROUPS=route_groups,
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_EXP2=bool(use_exp2),
        FORCE_SEED_KEYS=bool(force_seed_keys),
        LOGIT_INPUT_PRECISION=logit_input_precision,
        INLINE_LOCAL_SEED=bool(inline_local_seed),
        USE_ASSIGNED_LOGIT=bool(use_assigned_logit),
        USE_MASK_ASSIGN=bool(use_mask_assign),
        PACK_KV=bool(pack_kv),
        **launch_kwargs,
    )
    return merged_key, merged_value, routing_bias, None



__all__ = [
    "triton_biased_attention",
    "triton_direct_route_biased_attention",
    "triton_direct_route_biased_attention_packed_kv",
    "triton_hard_route_merge",
    "triton_hard_route_merge_c4",
    "triton_hard_route_merge_c4_direct_route",
]
