import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


HARD_C4_BACKEND = "hard_c4"
HARD_C4_FUSED_BACKEND = "hard_c4_fused"
HARD_BLOCK_M = 32
HARD_BLOCK_N = 64
HARD_NUM_WARPS = 4
HARD_NUM_STAGES = 1
HARD_INPUT_PRECISION = "ieee"


if triton is not None:

    @triton.jit
    def _hard_c4_fused_forward_kernel(
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        group = tl.program_id(0)
        q_col = tl.program_id(1)

        q_outer = group % Q_OUTER
        tmp = group // Q_OUTER
        head = tmp % HEADS
        batch = tmp // HEADS

        q_offsets = tl.arange(0, BLOCK_M)
        k_offsets = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        q_mask = q_offsets < Q_ROWS
        k_mask = k_offsets < K_COLS
        d_mask = dims < D_HEAD

        q_base = (batch * Q_OUTER + q_outer) * Q_ROWS * Q_COLS * HEADS * D_HEAD
        q = tl.load(
            q_ptr
            + q_base
            + (((q_offsets[:, None] * Q_COLS + q_col) * HEADS + head) * D_HEAD + dims[None, :]),
            mask=q_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float16)

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        seed_ids = tl.arange(0, 4)

        for key_block in range(0, K_OUTER * K_ROWS):
            key_outer = key_block // K_ROWS
            key_row = key_block - key_outer * K_ROWS
            keys = k_offsets
            k_base = (
                k_ptr
                + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
            )
            k = tl.load(
                k_base + keys[:, None] * HEADS * D_HEAD + dims[None, :],
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            v_base = (
                v_ptr
                + (((batch * K_OUTER + key_outer) * K_ROWS + key_row) * K_COLS * HEADS + head) * D_HEAD
            )
            v = tl.load(
                v_base + keys[:, None] * HEADS * D_HEAD + dims[None, :],
                mask=k_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

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

            seed_q = tl.load(
                q_ptr
                + q_base
                + (((seed_rows[:, None] * Q_COLS + q_col) * HEADS + head) * D_HEAD + dims[None, :]),
                mask=d_mask[None, :],
                other=0.0,
            )
            seed_scores = tl.dot(seed_q, tl.trans(k), input_precision=INPUT_PRECISION) * SCALE
            seed_scores = tl.where(k_mask[None, :], seed_scores, -float("inf"))

            seed_score_0 = tl.sum(tl.where(seed_ids[:, None] == 0, seed_scores, 0.0), axis=0)
            seed_score_1 = tl.sum(tl.where(seed_ids[:, None] == 1, seed_scores, 0.0), axis=0)
            seed_score_2 = tl.sum(tl.where(seed_ids[:, None] == 2, seed_scores, 0.0), axis=0)
            seed_score_3 = tl.sum(tl.where(seed_ids[:, None] == 3, seed_scores, 0.0), axis=0)

            best_seed_score = seed_score_0
            route_id = tl.zeros((BLOCK_N,), tl.int32)
            take = seed_score_1 > best_seed_score
            best_seed_score = tl.where(take, seed_score_1, best_seed_score)
            route_id = tl.where(take, 1, route_id)
            take = seed_score_2 > best_seed_score
            best_seed_score = tl.where(take, seed_score_2, best_seed_score)
            route_id = tl.where(take, 2, route_id)
            take = seed_score_3 > best_seed_score
            best_seed_score = tl.where(take, seed_score_3, best_seed_score)
            route_id = tl.where(take, 3, route_id)

            route_mask = k_mask[None, :] & (route_id[None, :] == seed_ids[:, None])
            route_max = tl.max(tl.where(route_mask, best_seed_score[None, :], -float("inf")), axis=1)
            route_has_key = route_max > -float("inf")
            safe_route_max = tl.where(route_has_key, route_max, 0.0)
            route_probs = tl.where(
                route_mask,
                tl.exp2((best_seed_score[None, :] - safe_route_max[:, None]) * 1.4426950408889634),
                0.0,
            )
            route_sum = tl.sum(route_probs, axis=1)
            safe_route_sum = tl.where(route_has_key, route_sum, 1.0)
            merge_weights = route_probs / safe_route_sum[:, None]
            log_route_sum = safe_route_max + tl.log2(safe_route_sum) * 0.6931471805599453

            merged_k = tl.dot(merge_weights.to(k.dtype), k).to(tl.float16)
            merged_v = tl.dot(merge_weights.to(v.dtype), v).to(tl.float16)
            expected_score = tl.sum(tl.where(route_mask, merge_weights * best_seed_score[None, :], 0.0), axis=1)
            route_bias = tl.where(route_has_key, expected_score - log_route_sum, float("inf"))

            scores = tl.dot(q, tl.trans(merged_k), input_precision=INPUT_PRECISION) * SCALE - route_bias[None, :]
            scores = tl.where(q_mask[:, None] & route_has_key[None, :], scores, -float("inf"))

            next_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
            probs = tl.exp2((scores - next_row_max[:, None]) * 1.4426950408889634)
            old_scale = tl.exp2((row_max - next_row_max) * 1.4426950408889634)

            acc = acc * old_scale[:, None] + tl.dot(probs.to(merged_v.dtype), merged_v)
            row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
            row_max = next_row_max

        out = acc / row_sum[:, None]
        out_base = out_ptr + batch * Q_OUTER * Q_ROWS * Q_COLS * HEADS * D_HEAD
        tl.store(
            out_base
            + (((q_outer * Q_ROWS + q_offsets[:, None]) * Q_COLS + q_col) * HEADS + head) * D_HEAD
            + dims[None, :],
            out,
            mask=q_mask[:, None] & d_mask[None, :],
        )


def hard_route_attention_triton_fused(q, k, v, scale=None, output=None):
    """Forward-only hard C4 FlashAttention-style Triton kernel.

    q: [B, query_outer, query_rows, query_cols, H, D]
    k: [B, key_outer, key_rows, key_cols, H, D]
    v: [B, key_outer, key_rows, key_cols, H, D]
    """
    if triton is None:
        raise RuntimeError("Triton is required for hard_route_attention_triton_fused.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, q_outer, q_rows, q_cols, heads, dim = q.shape
    _, k_outer, k_rows, k_cols, _, _ = k.shape
    if scale is None:
        scale = dim**-0.5

    output = output if output is not None and output.is_contiguous() else torch.empty_like(q)
    block_n = max(HARD_BLOCK_N, triton.next_power_of_2(k_cols))
    grid = (batch * heads * q_outer, q_cols)
    _hard_c4_fused_forward_kernel[grid](
        q,
        k,
        v,
        output,
        Q_OUTER=q_outer,
        K_OUTER=k_outer,
        Q_ROWS=q_rows,
        K_ROWS=k_rows,
        Q_COLS=q_cols,
        HEADS=heads,
        K_COLS=k_cols,
        D_HEAD=dim,
        BLOCK_M=HARD_BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(dim),
        SCALE=float(scale),
        INPUT_PRECISION=HARD_INPUT_PRECISION,
        num_warps=HARD_NUM_WARPS,
        num_stages=HARD_NUM_STAGES,
    )
    return output


__all__ = [
    "HARD_BLOCK_M",
    "HARD_BLOCK_N",
    "HARD_C4_BACKEND",
    "HARD_C4_FUSED_BACKEND",
    "HARD_INPUT_PRECISION",
    "HARD_NUM_STAGES",
    "HARD_NUM_WARPS",
    "hard_route_attention_triton_fused",
]
