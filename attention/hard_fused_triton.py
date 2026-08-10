import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


HARD_C4_FUSED_BACKEND = "hard_c4_fused"
FUSED_NUM_WARPS = 4
FUSED_NUM_STAGES = 1
FUSED_BLOCK_M = 32
FUSED_LOGIT_INPUT_PRECISION = "ieee"


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
    def _hard_route_fused_c4_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        out_ptr,
        A: tl.constexpr,
        F: tl.constexpr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        J: tl.constexpr,
        H: tl.constexpr,
        L: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCORE_SCALE: tl.constexpr,
        LOGIT_INPUT_PRECISION: tl.constexpr,
    ):
        """Fuse hard C4 merge and full compressed attention.

        One program owns one (batch, head, query outer, query column). It loads
        the 30 query rows once, streams every 52-key block once, builds the four
        compressed K/V tokens in registers, and immediately applies online
        softmax. No merged K/V/bias tensors are written to global memory.
        """
        group_id = tl.program_id(0)

        query_col = group_id % J
        tmp = group_id // J
        query_outer = tmp % A
        tmp = tmp // A
        head = tmp % H
        batch = tmp // H

        query_rows = tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_L)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_query = query_rows < Q_ROWS
        valid_key = key_offsets < L
        valid_dim = dim_offsets < D_HEAD

        query_base = (batch * A + query_outer) * Q_ROWS * J * H * D_HEAD
        q = tl.load(
            query_ptr
            + query_base
            + (((query_rows[:, None] * J + query_col) * H + head) * D_HEAD + dim_offsets[None, :]),
            mask=valid_query[:, None] & valid_dim[None, :],
            other=0.0,
        )
        q_for_attention = q.to(tl.float16)

        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for block_index in range(0, F * K_ROWS):
            key_outer = block_index // K_ROWS
            key_row = block_index - key_outer * K_ROWS

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

            seed_ids = tl.arange(0, 4)
            seed_rows = tl.where(
                seed_ids == 0,
                key_row,
                tl.where(seed_ids == 1, seed_row1, tl.where(seed_ids == 2, seed_row2, seed_row3)),
            )
            seeds = tl.load(
                query_ptr
                + query_base
                + (((seed_rows[:, None] * J + query_col) * H + head) * D_HEAD + dim_offsets[None, :]),
                mask=valid_dim[None, :],
                other=0.0,
            )
            seed_logits = tl.dot(seeds, tl.trans(key_block), input_precision=LOGIT_INPUT_PRECISION) * SCORE_SCALE
            seed_logits = tl.where(valid_key[None, :], seed_logits, -float("inf"))

            logits0 = tl.sum(tl.where(seed_ids[:, None] == 0, seed_logits, 0.0), axis=0)
            logits1 = tl.sum(tl.where(seed_ids[:, None] == 1, seed_logits, 0.0), axis=0)
            logits2 = tl.sum(tl.where(seed_ids[:, None] == 2, seed_logits, 0.0), axis=0)
            logits3 = tl.sum(tl.where(seed_ids[:, None] == 3, seed_logits, 0.0), axis=0)

            best_logits = logits0
            assignments = tl.zeros((BLOCK_L,), tl.int32)
            update = logits1 > best_logits
            best_logits = tl.where(update, logits1, best_logits)
            assignments = tl.where(update, 1, assignments)
            update = logits2 > best_logits
            best_logits = tl.where(update, logits2, best_logits)
            assignments = tl.where(update, 2, assignments)
            update = logits3 > best_logits
            best_logits = tl.where(update, logits3, best_logits)
            assignments = tl.where(update, 3, assignments)

            active = valid_key[None, :] & (assignments[None, :] == seed_ids[:, None])
            row_max = tl.max(tl.where(active, best_logits[None, :], -float("inf")), axis=1)
            has_keys = row_max > -float("inf")
            safe_row_max = tl.where(has_keys, row_max, 0.0)
            probs = tl.where(
                active,
                tl.exp2((best_logits[None, :] - safe_row_max[:, None]) * 1.4426950408889634),
                0.0,
            )
            denom = tl.sum(probs, axis=1)
            safe_denom = tl.where(has_keys, denom, 1.0)
            weights = probs / safe_denom[:, None]
            log_denom = safe_row_max + tl.log2(safe_denom) * 0.6931471805599453

            merged_key = tl.dot(weights.to(key_block.dtype), key_block).to(tl.float16)
            merged_value = tl.dot(weights.to(value_block.dtype), value_block).to(tl.float16)
            expected_logit = tl.sum(tl.where(active, weights * best_logits[None, :], 0.0), axis=1)
            routing_bias = tl.where(has_keys, expected_logit - log_denom, float("inf"))

            scores = tl.dot(q_for_attention, tl.trans(merged_key)) * SCORE_SCALE - routing_bias[None, :]
            scores = tl.where(valid_query[:, None] & has_keys[None, :], scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(max_score, block_max)
            p = tl.exp2((scores - next_max[:, None]) * 1.4426950408889634)
            alpha = tl.exp2((max_score - next_max) * 1.4426950408889634)
            acc = acc * alpha[:, None] + tl.dot(p.to(merged_value.dtype), merged_value)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            max_score = next_max

        out = acc / normalizer[:, None]
        out_base = out_ptr + batch * A * Q_ROWS * J * H * D_HEAD
        tl.store(
            out_base
            + (((query_outer * Q_ROWS + query_rows[:, None]) * J + query_col)
               * H + head) * D_HEAD
            + dim_offsets[None, :],
            out,
            mask=valid_query[:, None] & valid_dim[None, :],
        )


def hard_route_attention_triton_fused(q, k, v, scale, output=None):
    """Experimental single-kernel hard C4 path kept for manual optimization."""
    q = q.contiguous()
    batch, query_outer, query_rows, query_columns, heads, dim = q.shape
    _, key_outer, key_rows, key_columns, _, _ = k.shape
    block_l = triton.next_power_of_2(key_columns)
    block_d = triton.next_power_of_2(dim)
    output = output if output is not None and output.is_contiguous() else None
    if output is None:
        output = _cached_empty("hard_route_c4_fused_out", q.shape, q.device, q.dtype)

    grid = (batch * heads * query_outer * query_columns,)
    _hard_route_fused_c4_kernel[grid](
        q,
        k,
        v,
        output,
        A=query_outer,
        F=key_outer,
        Q_ROWS=query_rows,
        K_ROWS=key_rows,
        J=query_columns,
        H=heads,
        L=key_columns,
        D_HEAD=dim,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_L=block_l,
        BLOCK_D=block_d,
        SCORE_SCALE=float(scale),
        LOGIT_INPUT_PRECISION=FUSED_LOGIT_INPUT_PRECISION,
        num_warps=FUSED_NUM_WARPS,
        num_stages=FUSED_NUM_STAGES,
    )
    return output


__all__ = [
    "HARD_C4_FUSED_BACKEND",
    "FUSED_BLOCK_M",
    "FUSED_LOGIT_INPUT_PRECISION",
    "FUSED_NUM_STAGES",
    "FUSED_NUM_WARPS",
    "hard_route_attention_triton_fused",
]
