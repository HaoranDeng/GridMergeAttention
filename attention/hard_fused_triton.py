import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


HARD_C4_FUSED_BACKEND = "hard_c4_fused"
FUSED_BLOCK_Q_ROWS = 32
FUSED_NUM_WARPS = 4
FUSED_NUM_STAGES = 1
FUSED_LOGIT_INPUT_PRECISION = "ieee"


if triton is not None:

    @triton.jit
    def _hard_c4_fused_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        Q_ROWS: tl.constexpr,
        K_ROWS: tl.constexpr,
        Q_COLS: tl.constexpr,
        HEADS: tl.constexpr,
        K_COLS: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
        LOGIT_INPUT_PRECISION: tl.constexpr,
    ):
        """FlashAttention-style streaming over video key grids.

        This is the hard-C4 analogue of a dense flash attention forward kernel:
        each program owns one (batch, head, query outer, query column) and
        streams through a logical [K_ROWS, K_COLS] key grid, normally 30 x 52
        for Wan 480p tokens. It never materializes 30 x 52 x D in one program:
        one key row [K_COLS, D] is loaded, compressed, and immediately folded
        into the online softmax accumulator.
        """
        group = tl.program_id(0)
        query_col = tl.program_id(1)

        query_outer = group % Q_OUTER
        tmp = group // Q_OUTER
        head = tmp % HEADS
        batch = tmp // HEADS

        query_rows = tl.arange(0, BLOCK_Q)
        key_cols = tl.arange(0, BLOCK_K)
        dims = tl.arange(0, BLOCK_D)
        query_valid = query_rows < Q_ROWS
        key_col_valid = key_cols < K_COLS
        dim_valid = dims < D_HEAD

        q_base = (batch * Q_OUTER + query_outer) * Q_ROWS * Q_COLS * HEADS * D_HEAD
        query = tl.load(
            q_ptr
            + q_base
            + (((query_rows[:, None] * Q_COLS + query_col) * HEADS + head) * D_HEAD + dims[None, :]),
            mask=query_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        query_for_dot = query.to(tl.float16)

        softmax_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        softmax_sum = tl.zeros((BLOCK_Q,), tl.float32)
        output_acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

        for key_outer in range(0, K_OUTER):
            # The [K_ROWS, K_COLS] key grid is logical; each iteration only
            # loads one key row [K_COLS, D].
            for key_row in range(0, K_ROWS):
                seed_ids = tl.arange(0, 4)
                seed_row_1 = tl.where(key_row > 0, key_row - 1, 1)
                seed_row_2 = tl.where(
                    key_row == 0,
                    2,
                    tl.where(key_row == Q_ROWS - 1, Q_ROWS - 3, key_row + 1),
                )
                seed_row_3 = tl.where(
                    key_row <= 1,
                    3,
                    tl.where(key_row >= Q_ROWS - 2, Q_ROWS - 4, key_row - 2),
                )
                seed_rows = tl.where(
                    seed_ids == 0,
                    key_row,
                    tl.where(seed_ids == 1, seed_row_1, tl.where(seed_ids == 2, seed_row_2, seed_row_3)),
                )

                key_base = (
                    k_ptr
                    + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
                )
                key = tl.load(
                    key_base + key_cols[:, None] * HEADS * D_HEAD + dims[None, :],
                    mask=key_col_valid[:, None] & dim_valid[None, :],
                    other=0.0,
                )
                value_base = (
                    v_ptr
                    + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
                )
                value = tl.load(
                    value_base + key_cols[:, None] * HEADS * D_HEAD + dims[None, :],
                    mask=key_col_valid[:, None] & dim_valid[None, :],
                    other=0.0,
                )

                seed_queries = tl.load(
                    q_ptr
                    + q_base
                    + (((seed_rows[:, None] * Q_COLS + query_col) * HEADS + head) * D_HEAD + dims[None, :]),
                    mask=dim_valid[None, :],
                    other=0.0,
                )
                seed_logits = tl.dot(
                    seed_queries,
                    tl.trans(key),
                    input_precision=LOGIT_INPUT_PRECISION,
                ) * SCALE
                seed_logits = tl.where(key_col_valid[None, :], seed_logits, -float("inf"))

                logits_0 = tl.sum(tl.where(seed_ids[:, None] == 0, seed_logits, 0.0), axis=0)
                logits_1 = tl.sum(tl.where(seed_ids[:, None] == 1, seed_logits, 0.0), axis=0)
                logits_2 = tl.sum(tl.where(seed_ids[:, None] == 2, seed_logits, 0.0), axis=0)
                logits_3 = tl.sum(tl.where(seed_ids[:, None] == 3, seed_logits, 0.0), axis=0)

                best_logits = logits_0
                assigned_seed = tl.zeros((BLOCK_K,), tl.int32)
                take = logits_1 > best_logits
                best_logits = tl.where(take, logits_1, best_logits)
                assigned_seed = tl.where(take, 1, assigned_seed)
                take = logits_2 > best_logits
                best_logits = tl.where(take, logits_2, best_logits)
                assigned_seed = tl.where(take, 2, assigned_seed)
                take = logits_3 > best_logits
                best_logits = tl.where(take, logits_3, best_logits)
                assigned_seed = tl.where(take, 3, assigned_seed)

                routed = key_col_valid[None, :] & (assigned_seed[None, :] == seed_ids[:, None])
                route_max = tl.max(tl.where(routed, best_logits[None, :], -float("inf")), axis=1)
                route_has_key = route_max > -float("inf")
                safe_route_max = tl.where(route_has_key, route_max, 0.0)
                route_prob = tl.where(
                    routed,
                    tl.exp2((best_logits[None, :] - safe_route_max[:, None]) * 1.4426950408889634),
                    0.0,
                )
                route_sum = tl.sum(route_prob, axis=1)
                safe_route_sum = tl.where(route_has_key, route_sum, 1.0)
                merge_weight = route_prob / safe_route_sum[:, None]
                log_route_sum = safe_route_max + tl.log2(safe_route_sum) * 0.6931471805599453

                merged_key = tl.dot(merge_weight.to(key.dtype), key).to(tl.float16)
                merged_value = tl.dot(merge_weight.to(value.dtype), value).to(tl.float16)
                expected_logit = tl.sum(tl.where(routed, merge_weight * best_logits[None, :], 0.0), axis=1)
                routing_bias = tl.where(route_has_key, expected_logit - log_route_sum, float("inf"))

                scores = tl.dot(query_for_dot, tl.trans(merged_key)) * SCALE - routing_bias[None, :]
                scores = tl.where(query_valid[:, None] & route_has_key[None, :], scores, -float("inf"))

                next_max = tl.maximum(softmax_max, tl.max(scores, axis=1))
                prob = tl.exp2((scores - next_max[:, None]) * 1.4426950408889634)
                old_scale = tl.exp2((softmax_max - next_max) * 1.4426950408889634)
                output_acc = output_acc * old_scale[:, None] + tl.dot(prob.to(merged_value.dtype), merged_value)
                softmax_sum = softmax_sum * old_scale + tl.sum(prob, axis=1)
                softmax_max = next_max

        output = output_acc / softmax_sum[:, None]
        out_base = out_ptr + batch * Q_OUTER * Q_ROWS * Q_COLS * HEADS * D_HEAD
        tl.store(
            out_base
            + (((query_outer * Q_ROWS + query_rows[:, None]) * Q_COLS + query_col) * HEADS + head) * D_HEAD
            + dims[None, :],
            output,
            mask=query_valid[:, None] & dim_valid[None, :],
        )


def hard_route_attention_triton_fused(q, k, v, scale, output=None):
    """Experimental fused hard C4 attention, kept separate for manual tuning."""
    q = q.contiguous()
    batch, query_outer, query_rows, query_columns, heads, dim = q.shape
    _, key_outer, key_rows, key_columns, _, _ = k.shape

    output = output if output is not None and output.is_contiguous() else torch.empty_like(q)
    grid = (batch * heads * query_outer, query_columns)

    _hard_c4_fused_kernel[grid](
        q,
        k,
        v,
        output,
        Q_OUTER=query_outer,
        K_OUTER=key_outer,
        Q_ROWS=query_rows,
        K_ROWS=key_rows,
        Q_COLS=query_columns,
        HEADS=heads,
        K_COLS=key_columns,
        D_HEAD=dim,
        BLOCK_Q=FUSED_BLOCK_Q_ROWS,
        BLOCK_K=triton.next_power_of_2(key_columns),
        BLOCK_D=triton.next_power_of_2(dim),
        SCALE=float(scale),
        LOGIT_INPUT_PRECISION=FUSED_LOGIT_INPUT_PRECISION,
        num_warps=FUSED_NUM_WARPS,
        num_stages=FUSED_NUM_STAGES,
    )
    return output


__all__ = [
    "HARD_C4_FUSED_BACKEND",
    "FUSED_BLOCK_Q_ROWS",
    "FUSED_LOGIT_INPUT_PRECISION",
    "FUSED_NUM_STAGES",
    "FUSED_NUM_WARPS",
    "hard_route_attention_triton_fused",
]
