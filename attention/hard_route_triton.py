import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


if triton is not None:

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
            probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weights = probs / safe_denom[:, None]
            log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            routing_bias = tl.sum(
                tl.where(active, weights * (logits - log_denom[:, None]), 0.0),
                axis=1,
            )
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
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
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
            probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weights = probs / safe_denom[:, None]
            log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            routing_bias = tl.sum(
                tl.where(active, weights * (logits - log_denom[:, None]), 0.0),
                axis=1,
            )
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
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        ROUTE_LAYOUT: tl.constexpr,
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

        seed_rows_base = seed_rows_ptr + (((key_outer * K_ROWS + key_row) * J + query_offsets) * 4)
        seed_row0 = tl.load(seed_rows_base, mask=valid_query, other=0)
        seed_row1 = tl.load(seed_rows_base + 1, mask=valid_query, other=0)
        seed_row2 = tl.load(seed_rows_base + 2, mask=valid_query, other=0)
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
        seed3 = tl.load(
            query_base + (((seed_row3[:, None] * J + query_offsets[:, None]) * H + head) * D_HEAD + dim_offsets[None, :]),
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
            probs = tl.where(active, tl.exp(logits - row_max[:, None]), 0.0)
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weights = probs / safe_denom[:, None]
            log_denom = row_max + tl.log(safe_denom)

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            routing_bias = tl.sum(
                tl.where(active, weights * (logits - log_denom[:, None]), 0.0),
                axis=1,
            )
            routing_bias = tl.where(has_keys, routing_bias, 0.0)

            if ROUTE_LAYOUT:
                group = (batch * H + head) * J + query_offsets
                token = (key_outer * K_ROWS + key_row) * 4 + seed_idx
                merged_offset = (
                    (group[:, None] * (F * K_ROWS * 4) + token) * D_HEAD
                    + dim_offsets[None, :]
                )
                bias_offset = group * (F * K_ROWS * 4) + token
            else:
                merged_offset = (
                    (((((batch * F + key_outer) * J + query_offsets[:, None])
                       * K_ROWS + key_row) * 4 + seed_idx) * H + head)
                    * D_HEAD + dim_offsets[None, :]
                )
                bias_offset = (
                    (((((batch * H + head) * F + key_outer)
                       * K_ROWS + key_row) * J + query_offsets) * 4 + seed_idx)
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


def triton_biased_attention(query, key, value, bias, block_m=16, block_n=128, scale=1.0):
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

    out = torch.empty_like(query)
    grid = (groups, triton.cdiv(query_len, block_m))
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
        num_warps=4,
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
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        num_warps=4,
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
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        ROUTE_LAYOUT=False,
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

    route_groups = batch * heads * query_columns
    route_tokens = query_outer * key_rows * 4
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
        BLOCK_J=block_j,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        ROUTE_LAYOUT=True,
        num_warps=int(num_warps),
    )
    return merged_key, merged_value, routing_bias, None


__all__ = [
    "triton_biased_attention",
    "triton_hard_route_merge",
    "triton_hard_route_merge_c4",
    "triton_hard_route_shared_merge_c4",
    "triton_hard_route_shared_merge_c4_route_layout",
]
