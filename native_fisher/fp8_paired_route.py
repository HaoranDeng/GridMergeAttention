"""Generated direct E4M3 storage route, BF16 arithmetic."""
import triton
import triton.language as tl


@triton.jit
def _via_float(x):
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;", constraints="=h,f",
        args=[x.to(tl.float32)], dtype=tl.bfloat16, is_pure=True, pack=1)

@triton.jit
def decode_pair(word):
    bits=word.to(tl.uint16,bitcast=True)
    k=(bits & 255).to(tl.uint8).to(tl.float8e4nv,bitcast=True)
    v=(bits >> 8).to(tl.uint8).to(tl.float8e4nv,bitcast=True)
    return _via_float(k),_via_float(v)


@triton.jit
def fp8_paired_route(
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

    tl.static_assert(not SCALED_FP8_KV and not COMPRESSED_FP8)
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
        pair = tl.load(
            merged_k_ptr + merged_base * D_HEAD + dims[None, :],
            mask=token_mask[:, None],other=0,cache_modifier=".cg",
        )
        merged_key,merged_value=decode_pair(pair)
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
