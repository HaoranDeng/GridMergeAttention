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
        """Single-kernel experimental hard C4 attention.

        Each program owns one (batch, head, query outer, query column). It
        streams all key-row blocks, compresses each block into four route tokens,
        and updates the final query output with online softmax.
        """
        program = tl.program_id(0)

        query_col = program % Q_COLS
        tmp = program // Q_COLS
        query_outer = tmp % Q_OUTER
        tmp = tmp // Q_OUTER
        head = tmp % HEADS
        batch = tmp // HEADS

        q_rows = tl.arange(0, BLOCK_Q)
        k_cols = tl.arange(0, BLOCK_K)
        dims = tl.arange(0, BLOCK_D)
        q_valid = q_rows < Q_ROWS
        k_valid = k_cols < K_COLS
        d_valid = dims < D_HEAD

        q_base = (batch * Q_OUTER + query_outer) * Q_ROWS * Q_COLS * HEADS * D_HEAD
        q = tl.load(
            q_ptr
            + q_base
            + (((q_rows[:, None] * Q_COLS + query_col) * HEADS + head) * D_HEAD + dims[None, :]),
            mask=q_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        q_for_dot = q.to(tl.float16)

        row_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_Q,), tl.float32)
        row_acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

        for block in range(0, K_OUTER * K_ROWS):
            key_outer = block // K_ROWS
            key_row = block - key_outer * K_ROWS

            # The four local seeds are the matched row plus nearby rows.
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

            key_base = k_ptr + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
            key_block = tl.load(
                key_base + k_cols[:, None] * HEADS * D_HEAD + dims[None, :],
                mask=k_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            value_base = v_ptr + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
            value_block = tl.load(
                value_base + k_cols[:, None] * HEADS * D_HEAD + dims[None, :],
                mask=k_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            seed_queries = tl.load(
                q_ptr
                + q_base
                + (((seed_rows[:, None] * Q_COLS + query_col) * HEADS + head) * D_HEAD + dims[None, :]),
                mask=d_valid[None, :],
                other=0.0,
            )

            seed_logits = tl.dot(
                seed_queries,
                tl.trans(key_block),
                input_precision=LOGIT_INPUT_PRECISION,
            ) * SCALE
            seed_logits = tl.where(k_valid[None, :], seed_logits, -float("inf"))

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

            # Softmax only over keys routed to each seed, then merge K and V.
            routed = k_valid[None, :] & (assigned_seed[None, :] == seed_ids[:, None])
            cluster_max = tl.max(tl.where(routed, best_logits[None, :], -float("inf")), axis=1)
            cluster_has_key = cluster_max > -float("inf")
            safe_cluster_max = tl.where(cluster_has_key, cluster_max, 0.0)
            cluster_prob = tl.where(
                routed,
                tl.exp2((best_logits[None, :] - safe_cluster_max[:, None]) * 1.4426950408889634),
                0.0,
            )
            cluster_sum = tl.sum(cluster_prob, axis=1)
            safe_cluster_sum = tl.where(cluster_has_key, cluster_sum, 1.0)
            merge_weight = cluster_prob / safe_cluster_sum[:, None]
            log_cluster_sum = safe_cluster_max + tl.log2(safe_cluster_sum) * 0.6931471805599453

            merged_key = tl.dot(merge_weight.to(key_block.dtype), key_block).to(tl.float16)
            merged_value = tl.dot(merge_weight.to(value_block.dtype), value_block).to(tl.float16)
            expected_logit = tl.sum(tl.where(routed, merge_weight * best_logits[None, :], 0.0), axis=1)
            routing_bias = tl.where(cluster_has_key, expected_logit - log_cluster_sum, float("inf"))

            scores = tl.dot(q_for_dot, tl.trans(merged_key)) * SCALE - routing_bias[None, :]
            scores = tl.where(q_valid[:, None] & cluster_has_key[None, :], scores, -float("inf"))

            # FlashAttention-style online softmax update over compressed route tokens.
            block_max = tl.max(scores, axis=1)
            next_row_max = tl.maximum(row_max, block_max)
            prob = tl.exp2((scores - next_row_max[:, None]) * 1.4426950408889634)
            old_scale = tl.exp2((row_max - next_row_max) * 1.4426950408889634)
            row_acc = row_acc * old_scale[:, None] + tl.dot(prob.to(merged_value.dtype), merged_value)
            row_sum = row_sum * old_scale + tl.sum(prob, axis=1)
            row_max = next_row_max

        out = row_acc / row_sum[:, None]
        out_base = out_ptr + batch * Q_OUTER * Q_ROWS * Q_COLS * HEADS * D_HEAD
        tl.store(
            out_base
            + (((query_outer * Q_ROWS + q_rows[:, None]) * Q_COLS + query_col) * HEADS + head) * D_HEAD
            + dims[None, :],
            out,
            mask=q_valid[:, None] & d_valid[None, :],
        )


def hard_route_attention_triton_fused(q, k, v, scale, output=None):
    """Experimental fused hard C4 attention, kept separate for manual tuning."""
    q = q.contiguous()
    batch, query_outer, query_rows, query_columns, heads, dim = q.shape
    _, key_outer, key_rows, key_columns, _, _ = k.shape

    output = output if output is not None and output.is_contiguous() else torch.empty_like(q)
    grid = (batch * heads * query_outer * query_columns,)

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
