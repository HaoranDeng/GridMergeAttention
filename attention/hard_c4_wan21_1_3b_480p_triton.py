"""Hard-C4 GridMergeAttention for Wan2.1-T2V-1.3B at 480p.

Validated config: ``configs/token_merging_wan_hard_c4_triton_fewstep.yaml``.
The public function also delegates a 45 x 80 grid to the dedicated Wan2.1-14B
720p module so the existing ``hard_c4`` backend remains shape-compatible.

This mirrors ``monarch_rt_wan21_1_3b_480p_triton.py``:

1. For each ``30-query-row x 52-key-column`` grid block, gather four local seed
   queries, hard-route each key column to one seed, and write four compressed
   K/V tokens plus the entropy routing bias.
2. Run full softmax attention over all compressed tokens.

Inputs and output use block layout ``[B, outer, row, column, heads, dim]``.
The maintained fast merge is specialized for the finite, RMS-normalized BF16
Q/K distribution produced by Wan. Its winner-probability reuse assumes each
nonempty routed cluster retains nonzero FP32 probability mass.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


HARD_C4_BACKEND = "hard_c4"
HARD_RT_BLOCK_J = 64
HARD_RT_BLOCK_L = 64
HARD_RT_BLOCK_I = 32
HARD_RT_BLOCK_K = 32
HARD_RT_C = 4
HARD_RT_MERGE_NUM_WARPS = 4
HARD_RT_ROUTE_NUM_WARPS = 4
HARD_RT_MERGE_NUM_STAGES = 2
HARD_RT_ROUTE_NUM_STAGES = 2
HARD_RT_INPUT_PRECISION = "ieee"


if triton is not None:

    @triton.jit
    def _hard_c4_merge_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        BATCH: tl.constexpr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        C: tl.constexpr,
        SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        tl.static_assert(C == 4, "the specialized hard-C4 merge requires C == 4")
        tl.static_assert(D_HEAD == BLOCK_D, "the fast hard-C4 merge expects a power-of-two head dimension")

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)

        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        query_outer = block_id % Q_OUTER
        tmp = block_id // Q_OUTER
        key_outer = tmp % K_OUTER
        key_row = tmp // K_OUTER

        js = tl.arange(0, BLOCK_J)
        ls = tl.arange(0, BLOCK_L)
        dims = tl.arange(0, BLOCK_D)
        j_mask = js < COLS
        key_mask = ls < COLS

        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * TOTAL_HEADS * D_HEAD
        k_base = (batch * K_OUTER + key_outer) * ROWS * COLS * TOTAL_HEADS * D_HEAD
        key_base = k_base + (key_row * COLS + ls[:, None]) * TOTAL_HEADS * D_HEAD
        key = tl.load(
            k_ptr + key_base + global_head * D_HEAD + dims[None, :],
            mask=key_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )

        # C is fixed to four by the public hard-C4 backend.  Keeping each seed
        # as a 2-D tile gives reductions and MMAs simpler layouts than carrying
        # one rank-3 [C, J, L] value throughout the kernel.
        seed_start = tl.minimum(tl.maximum(key_row - 1, 0), ROWS - C)
        q0 = tl.load(
            q_ptr
            + q_base
            + ((((seed_start + 0) * COLS + js[:, None]) * TOTAL_HEADS + global_head) * D_HEAD + dims[None, :]),
            mask=j_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        q1 = tl.load(
            q_ptr
            + q_base
            + ((((seed_start + 1) * COLS + js[:, None]) * TOTAL_HEADS + global_head) * D_HEAD + dims[None, :]),
            mask=j_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        # Work in log2 units throughout the key-softmax/routing part.  This is
        # the same softmax objective as natural-log logits, but avoids repeated
        # large-tile multiplications by log2(e) before exp2.
        scale_log2 = SCALE * 1.4426950408889634
        s0 = tl.dot(q0, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2
        s1 = tl.dot(q1, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2
        # Consume two Q tiles before loading the other two.  The compiler can
        # reuse their shared-memory buffers, reducing this kernel from 80 KiB
        # to 48 KiB per CTA without changing any hard-C4 arithmetic.
        q2 = tl.load(
            q_ptr
            + q_base
            + ((((seed_start + 2) * COLS + js[:, None]) * TOTAL_HEADS + global_head) * D_HEAD + dims[None, :]),
            mask=j_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        q3 = tl.load(
            q_ptr
            + q_base
            + ((((seed_start + 3) * COLS + js[:, None]) * TOTAL_HEADS + global_head) * D_HEAD + dims[None, :]),
            mask=j_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        s2 = tl.dot(q2, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2
        s3 = tl.dot(q3, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2
        real_mask = j_mask[:, None] & key_mask[None, :]
        key_col_mask = key_mask[None, :]
        s0 = tl.where(key_col_mask, s0, -float("inf"))
        s1 = tl.where(key_col_mask, s1, -float("inf"))
        s2 = tl.where(key_col_mask, s2, -float("inf"))
        s3 = tl.where(key_col_mask, s3, -float("inf"))

        m0 = tl.max(s0, axis=1)
        m1 = tl.max(s1, axis=1)
        m2 = tl.max(s2, axis=1)
        m3 = tl.max(s3, axis=1)
        z0 = tl.sum(tl.exp2(s0 - m0[:, None]), axis=1)
        z1 = tl.sum(tl.exp2(s1 - m1[:, None]), axis=1)
        z2 = tl.sum(tl.exp2(s2 - m2[:, None]), axis=1)
        z3 = tl.sum(tl.exp2(s3 - m3[:, None]), axis=1)
        n0 = m0 + tl.log2(z0)
        n1 = m1 + tl.log2(z1)
        n2 = m2 + tl.log2(z2)
        n3 = m3 + tl.log2(z3)

        # Keep the normalized log-probability of only the winning seed.  For a
        # routed cluster, exp(score - seed_logsumexp) differs from exp(score)
        # by one query-row scalar, which cancels when that cluster is
        # normalized again below.
        winner_logprob = s0 - n0[:, None]
        route = tl.full((BLOCK_J, BLOCK_L), 0, tl.int32)
        candidate = s1 - n1[:, None]
        take = candidate > winner_logprob
        winner_logprob = tl.where(take, candidate, winner_logprob)
        route = tl.where(take, 1, route)
        candidate = s2 - n2[:, None]
        take = candidate > winner_logprob
        winner_logprob = tl.where(take, candidate, winner_logprob)
        route = tl.where(take, 2, route)
        candidate = s3 - n3[:, None]
        take = candidate > winner_logprob
        winner_logprob = tl.where(take, candidate, winner_logprob)
        route = tl.where(take, 3, route)
        route = tl.where(real_mask, route, -1)

        # Exponentiate once for the winning seed at every key column.  The old
        # form repeated a full 64x64 exp and masked max for each of four seeds.
        # Wan's RMS-normalized BF16 logits keep these 52-way probabilities in
        # the representable FP32 range; the real-QKV profiler covers all 360
        # maintained self-attention calls.
        winner_prob = tl.where(
            real_mask,
            tl.exp2(winner_logprob),
            0.0,
        )

        # Delay V until assignment and the shared probability tile are ready.
        value = tl.load(
            v_ptr + key_base + global_head * D_HEAD + dims[None, :],
            mask=key_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        for seed_id in tl.static_range(0, 4):
            route_mask = route == seed_id
            probs = tl.where(route_mask, winner_prob, 0.0)
            row_sum = tl.sum(probs, axis=1)
            has_key = row_sum > 0.0
            safe_sum = tl.where(has_key, row_sum, 1.0)
            weights = probs / safe_sum[:, None]
            logsum = tl.log2(safe_sum)
            expected = tl.sum(
                tl.where(route_mask, weights * winner_logprob, 0.0), axis=1
            )
            route_bias = tl.where(
                has_key,
                (expected - logsum) * 0.6931471805599453,
                float("inf"),
            )

            row_major_bias = (
                (((((batch * ACTIVE_HEADS + head) * Q_OUTER + query_outer) * K_OUTER + key_outer) * ROWS + key_row)
                  * COLS
                  + js)
                 * 4
                 + seed_id
            )
            tl.store(
                bias_ptr + row_major_bias,
                route_bias,
                mask=j_mask,
                eviction_policy="evict_first",
            )

            # Consume and store one dot accumulator before starting the next
            # one so merged K and V do not need overlapping live ranges.
            store_base = (
                (((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer) * ROWS + key_row) * COLS
                   + js[:, None])
                  * 4)
                 + seed_id
            )
            merged_key = tl.dot(weights.to(key.dtype), key, input_precision=INPUT_PRECISION)
            tl.store(
                merged_k_ptr + (((store_base * ACTIVE_HEADS + head) * D_HEAD) + dims[None, :]),
                merged_key,
                mask=j_mask[:, None],
                eviction_policy="evict_first",
            )
            merged_value = tl.dot(weights.to(value.dtype), value, input_precision=INPUT_PRECISION)
            tl.store(
                merged_v_ptr + (((store_base * ACTIVE_HEADS + head) * D_HEAD) + dims[None, :]),
                merged_value,
                mask=j_mask[:, None],
                eviction_policy="evict_first",
            )

    @triton.jit
    def _hard_c4_route_kernel(
        q_ptr,
        merged_k_ptr,
        merged_v_ptr,
        bias_ptr,
        out_ptr,
        BATCH: tl.constexpr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        ACTIVE_HEADS: tl.constexpr,
        HEAD_START: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        C: tl.constexpr,
        SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        tl.static_assert(D_HEAD == BLOCK_D, "the fast hard-C4 route expects a power-of-two head dimension")

        i_start = tl.program_id(0) * BLOCK_I
        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)

        head = batch_head % ACTIVE_HEADS
        global_head = HEAD_START + head
        batch = batch_head // ACTIVE_HEADS
        query_outer = query_group // COLS
        query_col = query_group % COLS

        rows = i_start + tl.arange(0, BLOCK_I)
        token_offsets = tl.arange(0, BLOCK_K * C)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < ROWS

        q_base = (((batch * Q_OUTER + query_outer) * ROWS + rows[:, None]) * COLS + query_col)
        query = tl.load(
            q_ptr + ((q_base * TOTAL_HEADS + global_head) * D_HEAD + dims[None, :]),
            mask=row_mask[:, None],
            other=0.0,
        )

        row_max = tl.full((BLOCK_I,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_I,), tl.float32)
        acc = tl.zeros((BLOCK_I, BLOCK_D), tl.float32)

        for key_outer in tl.range(0, K_OUTER):
            for key_start in tl.static_range(0, ROWS, BLOCK_K):
                key_row = key_start + token_offsets // C
                seed_id = token_offsets - (token_offsets // C) * C
                key_mask = key_row < ROWS
                merged_base = (
                    (((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer) * ROWS + key_row[:, None]) * COLS
                      + query_col)
                     * C)
                    + seed_id[:, None]
                )
                merged_key = tl.load(
                    merged_k_ptr + (((merged_base * ACTIVE_HEADS + head) * D_HEAD) + dims[None, :]),
                    mask=key_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                merged_value = tl.load(
                    merged_v_ptr + (((merged_base * ACTIVE_HEADS + head) * D_HEAD) + dims[None, :]),
                    mask=key_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                bias_base = (
                    (((((batch * ACTIVE_HEADS + head) * Q_OUTER + query_outer) * K_OUTER + key_outer) * ROWS + key_row)
                     * COLS
                     + query_col)
                    * C
                    + seed_id
                )
                bias = tl.load(
                    bias_ptr + bias_base,
                    mask=key_mask,
                    other=float("inf"),
                    eviction_policy="evict_first",
                )

                logits = tl.dot(query, tl.trans(merged_key), input_precision=INPUT_PRECISION)
                logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
                logits = tl.where(key_mask[None, :], logits, -float("inf"))

                new_max = tl.maximum(row_max, tl.max(logits, axis=1))
                probs = tl.exp2(logits - new_max[:, None])
                block_sum = tl.sum(probs, axis=1)
                old_scale = tl.exp2(row_max - new_max)
                acc = acc * old_scale[:, None] + tl.dot(
                    probs.to(query.dtype),
                    merged_value,
                    input_precision=INPUT_PRECISION,
                )
                row_sum = row_sum * old_scale + block_sum
                row_max = new_max

        out = acc / row_sum[:, None]
        tl.store(
            out_ptr + ((q_base * TOTAL_HEADS + global_head) * D_HEAD + dims[None, :]),
            out,
            mask=row_mask[:, None],
        )


def launch_hard_c4_merge(
    q,
    k,
    v,
    merged_k,
    merged_v,
    bias,
    scale,
    block_d,
    head_start=0,
    head_count=None,
):
    """Launch the Wan-specialized hard C4 merge stage."""

    batch, query_outer, rows, cols, total_heads, dim = q.shape
    active_heads = int(head_count or total_heads)
    key_outer = k.size(1)
    _hard_c4_merge_kernel[(query_outer * key_outer * rows, batch * active_heads)](
        q,
        k,
        v,
        merged_k,
        merged_v,
        bias,
        batch,
        query_outer,
        key_outer,
        rows,
        cols,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        BLOCK_J=HARD_RT_BLOCK_J,
        BLOCK_L=HARD_RT_BLOCK_L,
        BLOCK_D=block_d,
        C=HARD_RT_C,
        SCALE=float(scale),
        INPUT_PRECISION=HARD_RT_INPUT_PRECISION,
        num_warps=HARD_RT_MERGE_NUM_WARPS,
        num_stages=HARD_RT_MERGE_NUM_STAGES,
    )


def hard_c4_rt_attention_triton_blocks(
    q,
    k,
    v,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
    seed_q=None,
):
    """Run hard C4 on block-layout Q/K/V with MonarchRT-style staged kernels.

    Args:
        q, k, v: ``[B, outer, row, column, heads, dim]`` contiguous tensors.
            The fast merge expects finite RMS-normalized Wan BF16 Q/K values.
        scale: usual attention scale. Defaults to ``dim ** -0.5``.
        output: optional output tensor with the same shape as ``q``.
    """

    if q.shape[2:4] == (45, 80):
        from .hard_c4_wan21_14b_720p_triton import hard_c4_720p_attention_triton_blocks

        return hard_c4_720p_attention_triton_blocks(
            q,
            k,
            v,
            scale=scale,
            seed_q=seed_q,
            output=output,
            head_start=head_start,
            head_count=head_count,
        )

    if seed_q is not None:
        raise ValueError("separate seed_q is currently specialized for 720p Hard C4")
    if triton is None:
        raise RuntimeError("Triton is required for hard_c4_rt_attention_triton_blocks.")
    if scale is None:
        scale = q.size(-1) ** -0.5
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, query_outer, rows, cols, total_heads, dim = q.shape
    active_heads = int(head_count or total_heads)
    key_outer = k.size(1)
    block_d = triton.next_power_of_2(dim)

    merged_shape = (batch, query_outer, key_outer, rows, cols, HARD_RT_C, active_heads, dim)
    merged_k = torch.empty(merged_shape, device=q.device, dtype=q.dtype)
    merged_v = torch.empty(merged_shape, device=q.device, dtype=q.dtype)
    bias = torch.empty(
        (batch, active_heads, query_outer, key_outer, rows, cols, HARD_RT_C),
        device=q.device,
        dtype=torch.float32,
    )
    if output is None:
        output = torch.empty_like(q)

    launch_hard_c4_merge(q, k, v, merged_k, merged_v, bias, scale, block_d, head_start, active_heads)

    route_grid = (
        triton.cdiv(rows, HARD_RT_BLOCK_I),
        batch * active_heads,
        query_outer * cols,
    )
    _hard_c4_route_kernel[route_grid](
        q,
        merged_k,
        merged_v,
        bias,
        output,
        batch,
        query_outer,
        key_outer,
        rows,
        cols,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        BLOCK_I=HARD_RT_BLOCK_I,
        BLOCK_K=HARD_RT_BLOCK_K,
        BLOCK_D=block_d,
        C=HARD_RT_C,
        SCALE=float(scale),
        INPUT_PRECISION=HARD_RT_INPUT_PRECISION,
        num_warps=HARD_RT_ROUTE_NUM_WARPS,
        num_stages=HARD_RT_ROUTE_NUM_STAGES,
    )
    return output


__all__ = [
    "HARD_C4_BACKEND",
    "HARD_RT_BLOCK_I",
    "HARD_RT_BLOCK_J",
    "HARD_RT_BLOCK_K",
    "HARD_RT_BLOCK_L",
    "HARD_RT_MERGE_NUM_WARPS",
    "HARD_RT_ROUTE_NUM_WARPS",
    "hard_c4_rt_attention_triton_blocks",
    "launch_hard_c4_merge",
]
