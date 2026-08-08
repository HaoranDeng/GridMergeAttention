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
        KEY_HEAD_MAJOR: tl.constexpr,
        INLINE_LOCAL_SEED: tl.constexpr,
        TOKEN_MAJOR_OUTPUT: tl.constexpr,
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

        if KEY_HEAD_MAJOR:
            key_base = key_ptr + ((((batch * F + key_outer) * K_ROWS + key_row) * H + head) * L) * D_HEAD
            key_stride_l = D_HEAD
        else:
            key_base = key_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
            key_stride_l = H * D_HEAD
        key_block = tl.load(
            key_base + key_offsets[:, None] * key_stride_l + dim_offsets[None, :],
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

        route_group = (((batch * H + head) * A + query_outer) * J + query_offsets)
        route_token_base = (key_outer * K_ROWS + key_row) * 4
        route_tokens = F * K_ROWS * 4
        if KEY_HEAD_MAJOR:
            value_base = value_ptr + ((((batch * F + key_outer) * K_ROWS + key_row) * H + head) * L) * D_HEAD
            value_stride_l = D_HEAD
        else:
            value_base = value_ptr + (((batch * F + key_outer) * K_ROWS + key_row) * L * H + head) * D_HEAD
            value_stride_l = H * D_HEAD
        value_block = tl.load(
            value_base + key_offsets[:, None] * value_stride_l + dim_offsets[None, :],
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        for seed_idx in tl.static_range(0, 4):
            logits = logits0
            if seed_idx == 1:
                logits = logits1
            if seed_idx == 2:
                logits = logits2
            if seed_idx == 3:
                logits = logits3

            assigned_to_seed = assignments == seed_idx
            if FORCE_SEED_KEYS:
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
            if TOKEN_MAJOR_OUTPUT:
                route_offset = (
                    (route_token * ROUTE_GROUPS + route_group[:, None]) * D_HEAD
                    + dim_offsets[None, :]
                )
            else:
                route_offset = (
                    (route_group[:, None] * route_tokens + route_token) * D_HEAD
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
            if TOKEN_MAJOR_OUTPUT:
                bias_offset = route_token * ROUTE_GROUPS + route_group
            else:
                bias_offset = route_group * route_tokens + route_token
            tl.store(routing_bias_ptr + bias_offset, routing_bias, mask=valid_query)

    @triton.jit
    def _hard_route_shared_merge_c4_kernel(
        query_ptr,
        seed_rows_ptr,
        key_ptr,
        value_ptr,
        merged_key_ptr,
        merged_value_ptr,
        routing_bias_ptr,
        F: tl.constexpr,
        Q_ROWS: tl.constexpr,
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
        ROUTE_LAYOUT: tl.constexpr,
        WRITE_BIAS: tl.constexpr,
        BIAS_MODE: tl.constexpr,
        FORCE_SEED_KEYS: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_col_block = tl.program_id(1)

        head = group_id % H
        tmp = group_id // H
        key_row = tmp % K_ROWS
        tmp = tmp // K_ROWS
        key_outer = tmp % F
        batch = tmp // F

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

        seed_rows_base = seed_rows_ptr + (((key_outer * K_ROWS + key_row) * J + query_offsets) * C_SEEDS)
        seed_row0 = tl.load(seed_rows_base, mask=valid_query, other=0)
        seed_row1 = tl.load(seed_rows_base + 1, mask=valid_query, other=0)
        seed_row2 = tl.load(seed_rows_base + 2, mask=valid_query, other=0)
        if C_SEEDS == 4:
            seed_row3 = tl.load(seed_rows_base + 3, mask=valid_query, other=0)

        query_base = query_ptr + (batch * F + key_outer) * Q_ROWS * J * H * D_HEAD
        seed0 = tl.load(
            query_base + (((seed_row0[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed1 = tl.load(
            query_base + (((seed_row1[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        seed2 = tl.load(
            query_base + (((seed_row2[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        if C_SEEDS == 4:
            seed3 = tl.load(
                query_base + (((seed_row3[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
                mask=valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )

        logits0 = tl.dot(seed0, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        logits1 = tl.dot(seed1, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        logits2 = tl.dot(seed2, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        if C_SEEDS == 4:
            logits3 = tl.dot(seed3, tl.trans(key_block), input_precision="ieee") * SCORE_SCALE
        valid_logits = valid_query[:, None] & valid_key[None, :]
        logits0 = tl.where(valid_logits, logits0, -float("inf"))
        logits1 = tl.where(valid_logits, logits1, -float("inf"))
        logits2 = tl.where(valid_logits, logits2, -float("inf"))
        if C_SEEDS == 4:
            logits3 = tl.where(valid_logits, logits3, -float("inf"))

        best_logits = logits0
        assignments = tl.zeros((BLOCK_J, BLOCK_L), tl.int32)
        update = logits1 > best_logits
        best_logits = tl.where(update, logits1, best_logits)
        assignments = tl.where(update, 1, assignments)
        update = logits2 > best_logits
        best_logits = tl.where(update, logits2, best_logits)
        assignments = tl.where(update, 2, assignments)
        if C_SEEDS == 4:
            update = logits3 > best_logits
            assignments = tl.where(update, 3, assignments)
        if FORCE_SEED_KEYS:
            assignments = tl.where(
                valid_logits & (key_offsets[None, :] < C_SEEDS),
                key_offsets[None, :],
                assignments,
            )

        for seed_idx in tl.static_range(0, 4):
            if seed_idx < C_SEEDS:
                logits = logits0
                if seed_idx == 1:
                    logits = logits1
                if seed_idx == 2:
                    logits = logits2
                if seed_idx == 3:
                    logits = logits3

                assigned_to_seed = assignments == seed_idx
                if FORCE_SEED_KEYS:
                    active = valid_logits & assigned_to_seed
                else:
                    has_seed = tl.sum(tl.where(assigned_to_seed & valid_logits, 1, 0), axis=1) > 0
                    forced_first_key = (~has_seed[:, None]) & (key_offsets[None, :] == 0)
                    active = valid_logits & (assigned_to_seed | forced_first_key)

                row_max = tl.max(tl.where(active, logits, -float("inf")), axis=1)
                if USE_EXP2:
                    probs = tl.where(active, tl.exp2((logits - row_max[:, None]) * 1.4426950408889634), 0.0)
                else:
                    probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
                denom = tl.sum(probs, axis=1)
                safe_denom = tl.where(denom > 0.0, denom, 1.0)
                weights = probs / safe_denom[:, None]

                merged_key = tl.dot(weights.to(key_block.dtype), key_block)
                merged_value = tl.dot(weights.to(value_block.dtype), value_block)
                if WRITE_BIAS:
                    if BIAS_MODE == 1:
                        routing_bias = -tl.log(safe_denom)
                    elif BIAS_MODE == 2:
                        count = tl.sum(tl.where(active, 1.0, 0.0), axis=1)
                        routing_bias = -tl.log(tl.maximum(count, 1.0))
                    else:
                        log_denom = row_max + tl.log(safe_denom)
                        weighted_logits = tl.sum(tl.where(active, weights * logits, 0.0), axis=1)
                        routing_bias = weighted_logits - log_denom

                if ROUTE_LAYOUT:
                    group = (batch * H + head) * J + query_offsets
                    token = (key_outer * K_ROWS + key_row) * C_SEEDS + seed_idx
                    merged_offset = (
                        (group[:, None] * (F * K_ROWS * C_SEEDS) + token) * D_HEAD
                        + dim_offsets[None, :]
                    )
                    if WRITE_BIAS:
                        bias_offset = group * (F * K_ROWS * C_SEEDS) + token
                else:
                    merged_offset = (
                        (((((batch * F + key_outer) * J + query_offsets[:, None])
                           * K_ROWS + key_row) * C_SEEDS + seed_idx) * H + head)
                        * D_HEAD + dim_offsets[None, :]
                    )
                    if WRITE_BIAS:
                        bias_offset = (
                            (((((batch * H + head) * F + key_outer)
                               * K_ROWS + key_row) * J + query_offsets) * C_SEEDS + seed_idx)
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
                if WRITE_BIAS:
                    tl.store(routing_bias_ptr + bias_offset, routing_bias, mask=valid_query)

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
            v = tl.load(
                value_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
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
    def _grid_query_biased_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        out_ptr,
        A_OUTER: tl.constexpr,
        Q_ROWS: tl.constexpr,
        J_COLS: tl.constexpr,
        H_HEADS: tl.constexpr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_BIAS: tl.constexpr,
        USE_EXP2: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_block_id = tl.program_id(1)

        query_col = group_id % J_COLS
        tmp = group_id // J_COLS
        head = tmp % H_HEADS
        batch = tmp // H_HEADS

        query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        query_outer = query_offsets // Q_ROWS
        query_row = query_offsets - query_outer * Q_ROWS
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        query_base = query_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD
        key_base = key_ptr + group_id * K_LEN * D_HEAD
        value_base = value_ptr + group_id * K_LEN * D_HEAD
        bias_base = bias_ptr + group_id * K_LEN
        out_base = out_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD

        q = tl.load(
            query_base
            + (((query_outer[:, None] * Q_ROWS + query_row[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
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
            v = tl.load(
                value_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(k)) * SCORE_SCALE
            if USE_BIAS:
                bias = tl.load(bias_base + keys, mask=keys < K_LEN, other=float("inf"))
                logits = logits - bias[None, :]
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

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            max_score = next_max

        out = acc / normalizer[:, None]
        tl.store(
            out_base
            + (((query_outer[:, None] * Q_ROWS + query_row[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
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
        KEY_TOKEN_MAJOR: tl.constexpr,
        GROUPS: tl.constexpr,
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
        key_base = key_ptr + group_id * K_LEN * D_HEAD
        value_base = value_ptr + group_id * K_LEN * D_HEAD
        bias_base = bias_ptr + group_id * K_LEN
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
            if KEY_TOKEN_MAJOR:
                k = tl.load(
                    key_ptr + (keys[:, None] * GROUPS + group_id) * D_HEAD + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
                v = tl.load(
                    value_ptr + (keys[:, None] * GROUPS + group_id) * D_HEAD + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
                bias = tl.load(bias_ptr + keys * GROUPS + group_id, mask=keys < K_LEN, other=float("inf"))
            else:
                k = tl.load(
                    key_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
                v = tl.load(
                    value_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                    mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                    other=0.0,
                )
                bias = tl.load(bias_base + keys, mask=keys < K_LEN, other=float("inf"))

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

    @triton.jit
    def _grid_outer_local_biased_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        out_ptr,
        A_OUTER: tl.constexpr,
        Q_ROWS: tl.constexpr,
        J_COLS: tl.constexpr,
        H_HEADS: tl.constexpr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        TOKENS_PER_OUTER: tl.constexpr,
        LOCAL_RADIUS: tl.constexpr,
        LOCAL_KEY_LEN: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_BIAS: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_block_id = tl.program_id(1)

        query_col = group_id % J_COLS
        tmp = group_id // J_COLS
        head = tmp % H_HEADS
        batch = tmp // H_HEADS

        query_offsets = query_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        query_outer = query_offsets // Q_ROWS
        query_row = query_offsets - query_outer * Q_ROWS
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        first_outer = (query_block_id * BLOCK_M) // Q_ROWS
        local_outer_start = tl.maximum(first_outer - LOCAL_RADIUS, 0)
        local_token_start = local_outer_start * TOKENS_PER_OUTER

        query_base = query_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD
        key_base = key_ptr + group_id * K_LEN * D_HEAD
        value_base = value_ptr + group_id * K_LEN * D_HEAD
        bias_base = bias_ptr + group_id * K_LEN
        out_base = out_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD

        q = tl.load(
            query_base
            + (((query_outer[:, None] * Q_ROWS + query_row[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
            other=0.0,
        )

        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for local_start in range(0, LOCAL_KEY_LEN, BLOCK_N):
            keys = local_token_start + local_start + key_offsets
            valid = (query_offsets[:, None] < Q_LEN) & (keys[None, :] < K_LEN)
            k = tl.load(
                key_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            v = tl.load(
                value_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(k)) * SCORE_SCALE
            if USE_BIAS:
                bias = tl.load(bias_base + keys, mask=keys < K_LEN, other=float("inf"))
                logits = logits - bias[None, :]
            logits = tl.where(valid, logits, -float("inf"))

            has_block = tl.max(tl.where(valid, 1, 0), axis=1) > 0
            block_max = tl.max(tl.where(valid, logits, -float("inf")), axis=1)
            next_max = tl.where(has_block, tl.maximum(max_score, block_max), max_score)
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            p = tl.where(valid, tl.exp2((logits - safe_max[:, None]) * 1.4426950408889634), 0.0)
            alpha = tl.where(
                normalizer > 0.0,
                tl.exp2((max_score - safe_max) * 1.4426950408889634),
                0.0,
            )

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            max_score = next_max

        out = acc / normalizer[:, None]
        tl.store(
            out_base
            + (((query_outer[:, None] * Q_ROWS + query_row[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
            out,
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
        )

    @triton.jit
    def _grid_outer_exact_local_biased_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        out_ptr,
        A_OUTER: tl.constexpr,
        Q_ROWS: tl.constexpr,
        J_COLS: tl.constexpr,
        H_HEADS: tl.constexpr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        TOKENS_PER_OUTER: tl.constexpr,
        LOCAL_RADIUS: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        USE_BIAS: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        query_block_id = tl.program_id(1)

        query_col = group_id % J_COLS
        tmp = group_id // J_COLS
        head = tmp % H_HEADS
        batch = tmp // H_HEADS

        first_query = query_block_id * BLOCK_M
        last_query = tl.minimum(first_query + BLOCK_M - 1, Q_LEN - 1)
        first_outer = first_query // Q_ROWS
        last_outer = last_query // Q_ROWS
        local_outer_start = tl.maximum(first_outer - LOCAL_RADIUS, 0)
        local_outer_end = tl.minimum(last_outer + LOCAL_RADIUS, A_OUTER - 1)
        local_token_start = local_outer_start * TOKENS_PER_OUTER
        local_key_len = (local_outer_end - local_outer_start + 1) * TOKENS_PER_OUTER

        query_offsets = first_query + tl.arange(0, BLOCK_M)
        query_outer = query_offsets // Q_ROWS
        query_row = query_offsets - query_outer * Q_ROWS
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        query_base = query_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD
        key_base = key_ptr + group_id * K_LEN * D_HEAD
        value_base = value_ptr + group_id * K_LEN * D_HEAD
        bias_base = bias_ptr + group_id * K_LEN
        out_base = out_ptr + batch * A_OUTER * Q_ROWS * J_COLS * H_HEADS * D_HEAD

        q = tl.load(
            query_base
            + (((query_outer[:, None] * Q_ROWS + query_row[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
            other=0.0,
        )

        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        local_start = 0
        while local_start < local_key_len:
            local_keys = local_start + key_offsets
            keys = local_token_start + local_keys
            valid_key = local_keys < local_key_len
            valid = (query_offsets[:, None] < Q_LEN) & valid_key[None, :] & (keys[None, :] < K_LEN)
            k = tl.load(
                key_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=valid_key[:, None] & (keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            v = tl.load(
                value_base + keys[:, None] * D_HEAD + dim_offsets[None, :],
                mask=valid_key[:, None] & (keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(k)) * SCORE_SCALE
            if USE_BIAS:
                bias = tl.load(bias_base + keys, mask=valid_key & (keys < K_LEN), other=float("inf"))
                logits = logits - bias[None, :]
            logits = tl.where(valid, logits, -float("inf"))

            has_block = tl.max(tl.where(valid, 1, 0), axis=1) > 0
            block_max = tl.max(tl.where(valid, logits, -float("inf")), axis=1)
            next_max = tl.where(has_block, tl.maximum(max_score, block_max), max_score)
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            p = tl.where(valid, tl.exp2((logits - safe_max[:, None]) * 1.4426950408889634), 0.0)
            alpha = tl.where(
                normalizer > 0.0,
                tl.exp2((max_score - safe_max) * 1.4426950408889634),
                0.0,
            )

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            max_score = next_max
            local_start += BLOCK_N

        out = acc / normalizer[:, None]
        tl.store(
            out_base
            + (((query_outer[:, None] * Q_ROWS + query_row[:, None]) * J_COLS + query_col)
               * H_HEADS + head) * D_HEAD
            + dim_offsets[None, :],
            out,
            mask=(query_offsets[:, None] < Q_LEN) & (dim_offsets[None, :] < D_HEAD),
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
    key_token_major=False,
):
    """Fused routing attention that reads/writes hard-route block layout.

    query/out: [B, A, query_rows, query_columns, heads, dim]
    key/value: [B*heads*A*query_columns, compressed_tokens, dim]
    bias:      [B*heads*A*query_columns, compressed_tokens]
    """
    if triton is None:
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_routing_backend='triton_direct_*' requires fp16 or bf16 tensors.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    bias = bias.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    if key_token_major:
        key_len, groups, key_dim = key.shape
    else:
        groups, key_len, key_dim = key.shape
    expected_groups = batch * heads * query_outer * query_columns
    if groups != expected_groups:
        raise ValueError(
            "direct-route attention expects route-layout KV with "
            f"{expected_groups} groups, got {groups}."
        )
    expected_bias_shape = (key_len, groups) if key_token_major else (groups, key_len)
    if key_dim != dim or value.shape != key.shape or bias.shape != expected_bias_shape:
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
        KEY_TOKEN_MAJOR=bool(key_token_major),
        GROUPS=groups,
        **meta,
    )
    return out


def triton_grid_biased_attention(
    query,
    key,
    value,
    bias,
    block_m=128,
    block_n=32,
    scale=1.0,
    num_stages=1,
    use_bias=True,
    use_exp2=False,
):
    """Fused hard-route routing that reads/writes query block layout directly.

    query: [B, query_outer, query_rows, query_columns, heads, dim]
    key/value: [B * heads * query_columns, compressed_tokens, dim]
    bias: [B * heads * query_columns, compressed_tokens]
    """
    if triton is None:
        raise RuntimeError("hard_route_routing_backend='triton_grid_*' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_routing_backend='triton_grid_*' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_routing_backend='triton_grid_*' requires fp16 or bf16 query tensors.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if use_bias:
        bias = bias.contiguous()
    else:
        bias = torch.empty((1,), device=query.device, dtype=torch.float32)

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    groups, key_len, key_dim = key.shape
    expected_groups = batch * heads * query_columns
    if groups != expected_groups:
        raise ValueError(
            "grid hard-route attention expects route-layout KV with "
            f"{expected_groups} groups, got {groups}."
        )
    bias_shape_ok = (not use_bias) or bias.shape == (groups, key_len)
    if key_dim != dim or value.shape != key.shape or not bias_shape_ok:
        raise ValueError("query/key/value/bias shapes are inconsistent for grid Triton hard-route attention.")

    block_d = triton.next_power_of_2(dim)
    if block_d > 128:
        raise ValueError(f"Triton hard-route attention supports head dim <= 128, got {dim}.")

    query_len = query_outer * query_rows
    out = _cached_empty("grid_outer_local_attention_out", query.shape, query.device, query.dtype)
    grid = (groups, triton.cdiv(query_len, block_m))
    meta = {"num_warps": 4}
    if num_stages is not None:
        meta["num_stages"] = int(num_stages)
    _grid_query_biased_attention_kernel[grid](
        query,
        key,
        value,
        bias,
        out,
        A_OUTER=query_outer,
        Q_ROWS=query_rows,
        J_COLS=query_columns,
        H_HEADS=heads,
        Q_LEN=query_len,
        K_LEN=key_len,
        D_HEAD=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_BIAS=bool(use_bias),
        USE_EXP2=bool(use_exp2),
        **meta,
    )
    return out


def triton_grid_outer_local_biased_attention(
    query,
    key,
    value,
    bias,
    local_radius,
    block_m=128,
    block_n=32,
    scale=1.0,
    use_bias=True,
):
    """Grid routing over a local union of key outers for each query block.

    This keeps the compressed C4 KV representation unchanged, but each query
    block only reads compressed tokens from nearby key outers. The local window
    is shared by all queries in the Triton query block, making it a faster but
    approximate routing backend.
    """
    if triton is None:
        raise RuntimeError("hard_route_routing_backend='triton_grid_outer_local_*' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_routing_backend='triton_grid_outer_local_*' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_routing_backend='triton_grid_outer_local_*' requires fp16 or bf16 tensors.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if use_bias:
        bias = bias.contiguous()
    else:
        bias = torch.empty((1,), device=query.device, dtype=torch.float32)

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    groups, key_len, key_dim = key.shape
    expected_groups = batch * heads * query_columns
    if groups != expected_groups:
        raise ValueError(
            "outer-local grid routing expects route-layout KV with "
            f"{expected_groups} groups, got {groups}."
        )
    bias_shape_ok = (not use_bias) or bias.shape == (groups, key_len)
    if key_dim != dim or value.shape != key.shape or not bias_shape_ok:
        raise ValueError("query/key/value/bias shapes are inconsistent for outer-local grid routing.")
    if key_len % query_outer:
        raise ValueError("outer-local grid routing expects compressed tokens grouped evenly by key outer.")

    block_d = triton.next_power_of_2(dim)
    if block_d > 128:
        raise ValueError(f"Triton hard-route attention supports head dim <= 128, got {dim}.")

    local_radius = int(local_radius)
    if local_radius < 0:
        raise ValueError(f"local_radius must be non-negative, got {local_radius}.")

    query_len = query_outer * query_rows
    tokens_per_outer = key_len // query_outer
    max_query_outers_per_block = (int(block_m) + query_rows - 1) // query_rows + 1
    local_key_len = min(key_len, (max_query_outers_per_block + 2 * local_radius) * tokens_per_outer)

    out = _cached_empty("grid_outer_local_window_attention_out", query.shape, query.device, query.dtype)
    grid = (groups, triton.cdiv(query_len, block_m))
    _grid_outer_local_biased_attention_kernel[grid](
        query,
        key,
        value,
        bias,
        out,
        A_OUTER=query_outer,
        Q_ROWS=query_rows,
        J_COLS=query_columns,
        H_HEADS=heads,
        Q_LEN=query_len,
        K_LEN=key_len,
        TOKENS_PER_OUTER=tokens_per_outer,
        LOCAL_RADIUS=local_radius,
        LOCAL_KEY_LEN=int(local_key_len),
        D_HEAD=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_BIAS=bool(use_bias),
        num_warps=4,
        num_stages=1,
    )
    return out


def triton_grid_outer_exact_local_biased_attention(
    query,
    key,
    value,
    bias,
    local_radius,
    block_m=128,
    block_n=64,
    scale=1.0,
    use_bias=True,
    num_warps=4,
    num_stages=1,
):
    """Grid routing over the exact local key-outer window for each query tile."""
    if triton is None:
        raise RuntimeError("hard_route_routing_backend='triton_grid_outer_exact_*' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_routing_backend='triton_grid_outer_exact_*' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_routing_backend='triton_grid_outer_exact_*' requires fp16 or bf16 tensors.")

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if use_bias:
        bias = bias.contiguous()
    else:
        bias = torch.empty((1,), device=query.device, dtype=torch.float32)

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    groups, key_len, key_dim = key.shape
    expected_groups = batch * heads * query_columns
    if groups != expected_groups:
        raise ValueError(
            "exact outer-local grid routing expects route-layout KV with "
            f"{expected_groups} groups, got {groups}."
        )
    bias_shape_ok = (not use_bias) or bias.shape == (groups, key_len)
    if key_dim != dim or value.shape != key.shape or not bias_shape_ok:
        raise ValueError("query/key/value/bias shapes are inconsistent for exact outer-local grid routing.")
    if key_len % query_outer:
        raise ValueError("exact outer-local grid routing expects compressed tokens grouped evenly by key outer.")

    block_d = triton.next_power_of_2(dim)
    if block_d > 128:
        raise ValueError(f"Triton hard-route attention supports head dim <= 128, got {dim}.")

    local_radius = int(local_radius)
    if local_radius < 0:
        raise ValueError(f"local_radius must be non-negative, got {local_radius}.")

    query_len = query_outer * query_rows
    tokens_per_outer = key_len // query_outer

    out = _cached_empty("grid_outer_exact_local_attention_out", query.shape, query.device, query.dtype)
    grid = (groups, triton.cdiv(query_len, block_m))
    _grid_outer_exact_local_biased_attention_kernel[grid](
        query,
        key,
        value,
        bias,
        out,
        A_OUTER=query_outer,
        Q_ROWS=query_rows,
        J_COLS=query_columns,
        H_HEADS=heads,
        Q_LEN=query_len,
        K_LEN=key_len,
        TOKENS_PER_OUTER=tokens_per_outer,
        LOCAL_RADIUS=local_radius,
        D_HEAD=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_BIAS=bool(use_bias),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
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
    use_exp2=False,
    force_seed_keys=False,
    num_stages=None,
    logit_input_precision="ieee",
    key_head_major=False,
    inline_local_seed=False,
    token_major_output=False,
):
    """Exact non-shared C=4 merge that writes routing-ready K/V/bias.

    query:     [B, A, query_rows, query_columns, H, D]
    seed_rows: [A, F, key_rows, query_columns, 4]
    key/value: [B, F, key_rows, key_columns, H, D]

    Returns key/value [B*H*A*query_columns, F*key_rows*4, D] and bias
    [B*H*A*query_columns, F*key_rows*4], matching the layout consumed by
    triton_biased_attention.
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires fp16 or bf16 tensors.")

    query = query.contiguous()
    seed_rows = seed_rows.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    key_outer = key.size(1)
    key_rows = key.size(2)
    if key_head_major:
        if key.ndim != 6 or value.shape != key.shape:
            raise ValueError("head-major key/value must have shape [B, F, K, H, L, D].")
        key_heads = key.size(3)
        key_columns = key.size(4)
        key_dim = key.size(5)
    else:
        key_columns = key.size(3)
        key_heads = key.size(4)
        key_dim = key.size(5)
    if seed_rows.shape != (query_outer, key_outer, key_rows, query_columns, 4):
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
    if token_major_output:
        merged_shape = (route_tokens, route_groups, dim)
        bias_shape = (route_tokens, route_groups)
        cache_suffix = "token_major"
    else:
        merged_shape = (route_groups, route_tokens, dim)
        bias_shape = (route_groups, route_tokens)
        cache_suffix = "group_major"
    merged_key = _cached_empty(
        f"hard_route_c4_direct_route_key_{cache_suffix}",
        merged_shape,
        key.device,
        merged_dtype,
    )
    merged_value = _cached_empty(
        f"hard_route_c4_direct_route_value_{cache_suffix}",
        merged_shape,
        key.device,
        merged_dtype,
    )
    routing_bias = _cached_empty(
        f"hard_route_c4_direct_route_bias_{cache_suffix}",
        bias_shape,
        key.device,
        bias_dtype,
    )

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
        KEY_HEAD_MAJOR=bool(key_head_major),
        INLINE_LOCAL_SEED=bool(inline_local_seed),
        TOKEN_MAJOR_OUTPUT=bool(token_major_output),
        **launch_kwargs,
    )
    return merged_key, merged_value, routing_bias, None


def triton_hard_route_shared_merge_c4(query, seed_rows, key, value, scale=1.0, block_j_target=16):
    """C=4 shared hard-route merge that gathers seed queries inside Triton.

    query:     [B, F, query_rows, J, H, D]
    seed_rows: [F, key_rows, J, 4]
    key/value: [B, F, key_rows, L, H, D]

    Returns merged_key/value [B, 1, F, J, key_rows, 4, H, D] and routing_bias
    [B, H, 1, F, key_rows, J, 4], matching triton_hard_route_merge_c4.
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton_c4_direct' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_backend='triton_c4_direct' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton_c4_direct' requires fp16 or bf16 tensors.")

    query = query.contiguous()
    seed_rows = seed_rows.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    if key.shape[:2] != (batch, query_outer) or value.shape != key.shape:
        raise ValueError("query/key/value shapes are inconsistent for shared C4 Triton merge.")
    key_rows = key.size(2)
    key_columns = key.size(3)
    if seed_rows.shape != (query_outer, key_rows, query_columns, 4):
        raise ValueError("seed_rows must have shape [query_outer, key_rows, query_columns, 4].")
    if key.size(4) != heads or key.size(5) != dim:
        raise ValueError("query/key head dimensions are inconsistent for shared C4 Triton merge.")

    block_j = min(int(block_j_target), triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    if block_l > 128:
        raise ValueError(f"Triton hard-route merge supports at most 128 key tokens per block, got {key_columns}.")
    if block_d > 128:
        raise ValueError(f"Triton hard-route merge supports head dim <= 128, got {dim}.")

    merged_key = torch.empty(
        (batch, 1, query_outer, query_columns, key_rows, 4, heads, dim),
        device=key.device,
        dtype=key.dtype,
    )
    merged_value = torch.empty_like(merged_key)
    routing_bias = torch.empty(
        (batch, heads, 1, query_outer, key_rows, query_columns, 4),
        device=key.device,
        dtype=torch.float32,
    )

    grid = (
        batch * query_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    _hard_route_shared_merge_c4_kernel[grid](
        query,
        seed_rows,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias,
        F=query_outer,
        Q_ROWS=query_rows,
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
        ROUTE_LAYOUT=False,
        WRITE_BIAS=True,
        BIAS_MODE=0,
        FORCE_SEED_KEYS=False,
        USE_EXP2=False,
        num_warps=4,
    )
    return merged_key, merged_value, routing_bias, None


def triton_hard_route_shared_merge_c4_route_layout(
    query,
    seed_rows,
    key,
    value,
    scale=1.0,
    block_j_target=16,
    num_warps=4,
    write_bias=True,
    bias_mode="exact",
    bias_dtype=None,
    force_seed_keys=False,
    use_exp2=False,
    workspace_cache=False,
    num_stages=None,
):
    """C=4 shared hard-route merge with routing-ready K/V/bias layout.

    Returns:
        key/value: [B * H * J, F * key_rows * 4, D]
        bias:      [B * H * J, F * key_rows * 4]
    """
    if triton is None:
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires fp16 or bf16 tensors.")

    query = query.contiguous()
    seed_rows = seed_rows.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    if key.shape[:2] != (batch, query_outer) or value.shape != key.shape:
        raise ValueError("query/key/value shapes are inconsistent for shared C4 Triton merge.")
    key_rows = key.size(2)
    key_columns = key.size(3)
    if seed_rows.shape != (query_outer, key_rows, query_columns, 4):
        raise ValueError("seed_rows must have shape [query_outer, key_rows, query_columns, 4].")
    if key.size(4) != heads or key.size(5) != dim:
        raise ValueError("query/key head dimensions are inconsistent for shared C4 Triton merge.")

    block_j = min(int(block_j_target), triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    if block_l > 128:
        raise ValueError(f"Triton hard-route merge supports at most 128 key tokens per block, got {key_columns}.")
    if block_d > 128:
        raise ValueError(f"Triton hard-route merge supports head dim <= 128, got {dim}.")
    bias_mode = (bias_mode or "exact").lower()
    bias_mode_id = {"exact": 0, "maxprob": 1, "count": 2}.get(bias_mode)
    if bias_mode_id is None:
        raise ValueError(f"unsupported hard-route routing bias mode: {bias_mode!r}.")
    bias_dtype = bias_dtype or torch.float32

    route_groups = batch * heads * query_columns
    route_tokens = query_outer * key_rows * 4
    alloc = _cached_empty if workspace_cache else None
    if alloc is None:
        merged_key = torch.empty(
            (route_groups, route_tokens, dim),
            device=key.device,
            dtype=key.dtype,
        )
        merged_value = torch.empty_like(merged_key)
    else:
        merged_key = alloc("shared_c4_route_key", (route_groups, route_tokens, dim), key.device, key.dtype)
        merged_value = alloc("shared_c4_route_value", (route_groups, route_tokens, dim), key.device, key.dtype)
    routing_bias = None
    if write_bias:
        if alloc is None:
            routing_bias = torch.empty(
                (route_groups, route_tokens),
                device=key.device,
                dtype=bias_dtype,
            )
        else:
            routing_bias = alloc("shared_c4_route_bias", (route_groups, route_tokens), key.device, bias_dtype)
    routing_bias_arg = routing_bias
    if routing_bias_arg is None:
        routing_bias_arg = torch.empty((1,), device=key.device, dtype=torch.float32)

    grid = (
        batch * query_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    launch_kwargs = {"num_warps": int(num_warps)}
    if num_stages is not None:
        launch_kwargs["num_stages"] = int(num_stages)
    _hard_route_shared_merge_c4_kernel[grid](
        query,
        seed_rows,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias_arg,
        F=query_outer,
        Q_ROWS=query_rows,
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
        ROUTE_LAYOUT=True,
        WRITE_BIAS=bool(write_bias),
        BIAS_MODE=bias_mode_id,
        FORCE_SEED_KEYS=bool(force_seed_keys),
        USE_EXP2=bool(use_exp2),
        **launch_kwargs,
    )
    return merged_key, merged_value, routing_bias, None


def triton_hard_route_shared_merge_c4_default_route_layout(query, seed_rows, key, value, scale=1.0):
    """Fast wrapper for the default shared C4 hard-route merge launch."""
    if triton is None:
        raise RuntimeError("hard_route_backend='triton_c4_direct_route' requires Triton.")

    query = query.contiguous()
    seed_rows = seed_rows.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    key_rows = key.size(2)
    key_columns = key.size(3)
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)

    route_groups = batch * heads * query_columns
    route_tokens = query_outer * key_rows * 4
    merged_key = _cached_empty("shared_c4_route_key", (route_groups, route_tokens, dim), key.device, key.dtype)
    merged_value = _cached_empty("shared_c4_route_value", (route_groups, route_tokens, dim), key.device, key.dtype)
    routing_bias = _cached_empty("shared_c4_route_bias", (route_groups, route_tokens), key.device, key.dtype)

    grid = (
        batch * query_outer * key_rows * heads,
        triton.cdiv(query_columns, 32),
    )
    _hard_route_shared_merge_c4_kernel[grid](
        query,
        seed_rows,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias,
        F=query_outer,
        Q_ROWS=query_rows,
        K_ROWS=key_rows,
        J=query_columns,
        H=heads,
        L=key_columns,
        D_HEAD=dim,
        C_SEEDS=4,
        BLOCK_J=32,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        ROUTE_LAYOUT=True,
        WRITE_BIAS=True,
        BIAS_MODE=0,
        FORCE_SEED_KEYS=False,
        USE_EXP2=True,
        num_warps=2,
    )
    return merged_key, merged_value, routing_bias, None


def triton_hard_route_shared_merge_c3_route_layout(
    query,
    seed_rows,
    key,
    value,
    scale=1.0,
    block_j_target=32,
    num_warps=2,
    force_seed_keys=True,
    use_exp2=True,
):
    """C=3 shared hard-route merge with routing-ready K/V/bias layout."""
    if triton is None:
        raise RuntimeError("hard_route_backend='triton_c3_direct_route' requires Triton.")
    if not query.is_cuda:
        raise RuntimeError("hard_route_backend='triton_c3_direct_route' requires CUDA/ROCm tensors.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("hard_route_backend='triton_c3_direct_route' requires fp16 or bf16 tensors.")

    query = query.contiguous()
    seed_rows = seed_rows.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    if key.shape[:2] != (batch, query_outer) or value.shape != key.shape:
        raise ValueError("query/key/value shapes are inconsistent for shared C3 Triton merge.")
    key_rows = key.size(2)
    key_columns = key.size(3)
    if seed_rows.shape != (query_outer, key_rows, query_columns, 3):
        raise ValueError("seed_rows must have shape [query_outer, key_rows, query_columns, 3].")
    if key.size(4) != heads or key.size(5) != dim:
        raise ValueError("query/key head dimensions are inconsistent for shared C3 Triton merge.")

    block_j = min(int(block_j_target), triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    if block_l > 128:
        raise ValueError(f"Triton hard-route merge supports at most 128 key tokens per block, got {key_columns}.")
    if block_d > 128:
        raise ValueError(f"Triton hard-route merge supports head dim <= 128, got {dim}.")

    route_groups = batch * heads * query_columns
    route_tokens = query_outer * key_rows * 3
    merged_key = torch.empty(
        (route_groups, route_tokens, dim),
        device=key.device,
        dtype=key.dtype,
    )
    merged_value = torch.empty_like(merged_key)
    routing_bias = torch.empty(
        (route_groups, route_tokens),
        device=key.device,
        dtype=torch.float32,
    )

    grid = (
        batch * query_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    _hard_route_shared_merge_c4_kernel[grid](
        query,
        seed_rows,
        key,
        value,
        merged_key,
        merged_value,
        routing_bias,
        F=query_outer,
        Q_ROWS=query_rows,
        K_ROWS=key_rows,
        J=query_columns,
        H=heads,
        L=key_columns,
        D_HEAD=dim,
        C_SEEDS=3,
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        ROUTE_LAYOUT=True,
        WRITE_BIAS=True,
        BIAS_MODE=0,
        FORCE_SEED_KEYS=bool(force_seed_keys),
        USE_EXP2=bool(use_exp2),
        num_warps=int(num_warps),
    )
    return merged_key, merged_value, routing_bias, None


__all__ = [
    "triton_biased_attention",
    "triton_direct_route_biased_attention",
    "triton_grid_biased_attention",
    "triton_grid_outer_exact_local_biased_attention",
    "triton_grid_outer_local_biased_attention",
    "triton_hard_route_merge",
    "triton_hard_route_merge_c4",
    "triton_hard_route_merge_c4_direct_route",
    "triton_hard_route_shared_merge_c3_route_layout",
    "triton_hard_route_shared_merge_c4",
    "triton_hard_route_shared_merge_c4_default_route_layout",
    "triton_hard_route_shared_merge_c4_route_layout",
]
