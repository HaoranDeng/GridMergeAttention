import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


HARD_C4_BACKEND = "hard_c4"
MERGE_BLOCK_J = 32
MERGE_NUM_WARPS = 2
MERGE_NUM_STAGES = 1
MERGE_LOGIT_INPUT_PRECISION = "ieee"
ROUTING_BLOCK_M = 32
ROUTING_BLOCK_N = 32
ROUTING_NUM_WARPS = 1
ROUTING_NUM_STAGES = 1


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
    def _hard_route_merge_c4_simple_bestlogit_kernel(
        query_ptr,
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
        LOGIT_INPUT_PRECISION: tl.constexpr,
    ):
        """Specialized exact C4 merge for the current bestlogit fast path."""
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
        valid_logits = valid_query[:, None] & valid_key[None, :]

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

        seed_row1 = tl.where(key_row > 0, key_row - 1, 1)
        seed_row2 = tl.where(
            key_row == 0,
            2,
            tl.where(key_row == Q_ROWS - 1, Q_ROWS - 3, key_row + 1),
        )
        seed_row3 = tl.where(
            key_row <= 1,
            3,
            tl.where(key_row >= Q_ROWS - 2, Q_ROWS - 4, key_row - 2),
        )
        seed_row0 = tl.full((BLOCK_J,), key_row, tl.int64)
        seed_row1 = tl.full((BLOCK_J,), seed_row1, tl.int64)
        seed_row2 = tl.full((BLOCK_J,), seed_row2, tl.int64)
        seed_row3 = tl.full((BLOCK_J,), seed_row3, tl.int64)

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
        best_logits = tl.where(update, logits3, best_logits)
        assignments = tl.where(update, 3, assignments)
        assigned_logits = best_logits

        route_group = (((batch * H + head) * A + query_outer) * J + query_offsets)
        route_token_base = (key_outer * K_ROWS + key_row) * 4

        for seed_idx in tl.static_range(0, 4):
            active = valid_logits & (assignments == seed_idx)
            row_max = tl.max(tl.where(active, assigned_logits, -float("inf")), axis=1)
            has_keys = row_max > -float("inf")
            safe_row_max = tl.where(has_keys, row_max, 0.0)
            probs = tl.where(
                active,
                tl.exp2((assigned_logits - safe_row_max[:, None]) * 1.4426950408889634),
                0.0,
            )
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(has_keys, denom, 1.0)
            weights = probs / safe_denom[:, None]
            log_denom = safe_row_max + tl.log2(safe_denom) * 0.6931471805599453

            merged_key = tl.dot(weights.to(key_block.dtype), key_block)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block)
            expected_logit = tl.sum(tl.where(active, weights * assigned_logits, 0.0), axis=1)
            routing_bias = tl.where(has_keys, expected_logit - log_denom, float("inf"))

            route_token = route_token_base + seed_idx
            bias_offset = route_token * ROUTE_GROUPS + route_group
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
            tl.store(routing_bias_ptr + bias_offset, routing_bias, mask=valid_query)

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
            k = tl.load(
                key_ptr + (keys[:, None] * GROUPS + group_id) * D_HEAD + dim_offsets[None, :],
                mask=(keys[:, None] < K_LEN) & (dim_offsets[None, :] < D_HEAD),
                other=0.0,
            )
            bias = tl.load(
                bias_ptr + keys * GROUPS + group_id,
                mask=keys < K_LEN,
                other=float("inf"),
            )

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


def route_compressed_attention(query, key, value, bias, scale, out=None):
    """Full compressed attention over token-major hard C4 K/V/bias.

    query/out: [B, A, query_rows, query_columns, heads, dim]
    key/value: [compressed_tokens, B*heads*A*query_columns, dim]
    bias:      [compressed_tokens, B*heads*A*query_columns]
    """
    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    key_len, groups, _ = key.shape
    block_d = triton.next_power_of_2(dim)

    if out is None:
        out = _cached_empty("direct_route_attention_out", query.shape, query.device, query.dtype)
    grid = (groups, triton.cdiv(query_rows, ROUTING_BLOCK_M))
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
        BLOCK_M=ROUTING_BLOCK_M,
        BLOCK_N=ROUTING_BLOCK_N,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        USE_EXP2=True,
        CAST_Q_TO_FP16=bool(key.dtype == torch.float16 and query.dtype != torch.float16),
        GROUPS=groups,
        num_warps=ROUTING_NUM_WARPS,
        num_stages=ROUTING_NUM_STAGES,
    )
    return out


def merge_kv_direct_c4(query, key, value, scale):
    """Merge each 30q x 52k hard C4 block into four compressed K/V tokens.

    This is intentionally narrow: it implements only the accepted
    no-force bestlogit_simplemerge path.

    query:     [B, A, query_rows, query_columns, H, D]
    key/value: [B, F, key_rows, key_columns, H, D]

    Returns token-major K/V/bias for direct compressed attention:
      key/value: [F * key_rows * 4, B * H * A * query_columns, D]
      bias:      [F * key_rows * 4, B * H * A * query_columns]
    """
    batch, query_outer, query_rows, query_columns, heads, dim = query.shape
    _, key_outer, key_rows, key_columns, _, _ = key.shape

    block_j = min(MERGE_BLOCK_J, triton.next_power_of_2(query_columns))
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)

    route_groups = batch * heads * query_outer * query_columns
    route_tokens = key_outer * key_rows * 4

    merged_key = _cached_empty(
        "hard_route_c4_direct_route_key_token_major",
        (route_tokens, route_groups, dim),
        key.device,
        torch.float16,
    )
    merged_value = _cached_empty(
        "hard_route_c4_direct_route_value_token_major",
        (route_tokens, route_groups, dim),
        key.device,
        torch.float16,
    )
    routing_bias = _cached_empty(
        "hard_route_c4_direct_route_bias_token_major",
        (route_tokens, route_groups),
        key.device,
        torch.float16,
    )

    grid = (
        batch * query_outer * key_outer * key_rows * heads,
        triton.cdiv(query_columns, block_j),
    )
    _hard_route_merge_c4_simple_bestlogit_kernel[grid](
        query,
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
        LOGIT_INPUT_PRECISION=MERGE_LOGIT_INPUT_PRECISION,
        num_warps=MERGE_NUM_WARPS,
        num_stages=MERGE_NUM_STAGES,
    )
    return merged_key, merged_value, routing_bias


def hard_route_attention_triton(
    q,
    k,
    v,
    scale,
    output=None,
):
    """Optimized hard C4 attention: Triton merge followed by full compressed attention."""
    q = q.contiguous()
    key, value, bias = merge_kv_direct_c4(q, k, v, scale)
    output = output if output is not None and output.is_contiguous() else None
    return route_compressed_attention(
        q,
        key,
        value,
        bias,
        scale,
        out=output,
    )


__all__ = [
    "HARD_C4_BACKEND",
    "MERGE_BLOCK_J",
    "MERGE_LOGIT_INPUT_PRECISION",
    "MERGE_NUM_STAGES",
    "MERGE_NUM_WARPS",
    "ROUTING_BLOCK_M",
    "ROUTING_BLOCK_N",
    "ROUTING_NUM_STAGES",
    "ROUTING_NUM_WARPS",
    "hard_route_attention_triton",
    "merge_kv_direct_c4",
    "route_compressed_attention",
]
