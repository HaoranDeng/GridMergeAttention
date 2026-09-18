"""Adaptive-seed hard-C4 GridMerge attention -- Triton implementation.

Pairs with ``adaptive_seed_sdpa.py`` (same directory), which supplies the
selector. Covers both maintained shapes:

    Wan2.1-1.3B 480p   frames 21, rows 30, cols 52, heads 12, dim 128
    Wan2.1-14B  720p   frames 21, rows 45, cols 80, heads 40, dim 128

WHY THE SELECTOR IS NOT A KERNEL
--------------------------------
Two previous attempts to write the seed selector as a Triton kernel both came in
*slower* than the PyTorch they replaced (4.06 ms vs 0.92 ms; a tiled version was
worse still, and got worse as the tile grew). The selection work is top-k,
sequential greedy and cross-lane reductions -- control flow, not matmul -- so the
tensor cores idle, and tiling multiplies the serial logic per program.

So this file splits the pipeline the way that evidence points:

    PyTorch  ->  seed table   [B, A, F, K, J, H, 4] of int32 row indices
    Triton   ->  merge kernel reads the table, emits merged K/V + entropy bias
    Triton   ->  route kernel runs full attention over the merged tokens

The merge and route kernels are the maintained hard-C4 kernels with exactly one
change: instead of computing ``seed_start = clamp(key_row - 1, 0, ROWS - 4)``
inline, the merge kernel loads four row indices per query column and head. That
keeps every tile shape and the whole second stage untouched, and it is portable
-- the table comes from ordinary PyTorch, so nothing here is device-specific
beyond what the existing kernels already require.

COST NOTE, UNMEASURED
---------------------
The seed table is per query column, so the four Q tile loads stop being
contiguous across the column tile: each lane reads its own row. Every lane still
reads ``D`` contiguous elements, but this is a real change to the access pattern
and may cost some of the fixed-seed kernel's throughput. It has not been
benchmarked. Treat this file as correct and portable, not as tuned.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:                                   # CPU-only environments
    triton = None
    tl = None

C4 = 4
# log2(e) and ln(2) are inlined at their use sites: Triton kernels cannot read
# module-level globals unless they are tl.constexpr.

# (BLOCK_J, BLOCK_L, BLOCK_I, BLOCK_K, merge warps, route warps)
SHAPE_CONFIG = {
    (30, 52): (64, 64, 32, 32, 4, 4),               # Wan2.1-1.3B 480p
    (45, 80): (32, 128, 64, 16, 8, 8),              # Wan2.1-14B 720p
}


if triton is not None:

    @triton.jit
    def _temporal_simpson_query_kernel(
        q_ptr,
        output_ptr,
        OUTPUT_ELEMENTS: tl.constexpr,
        FRAME_STRIDE: tl.constexpr,
        FRAMES: tl.constexpr,
        CENTER_FRAME: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < OUTPUT_ELEMENTS
        batch = offsets // FRAME_STRIDE
        within_frame = offsets - batch * FRAME_STRIDE
        base = batch * FRAMES * FRAME_STRIDE + within_frame

        first = tl.load(q_ptr + base, mask=mask, other=0.0).to(tl.float32)
        center = tl.load(
            q_ptr + base + CENTER_FRAME * FRAME_STRIDE,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        last = tl.load(
            q_ptr + base + (FRAMES - 1) * FRAME_STRIDE,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(output_ptr + offsets, (first + 4.0 * center + last) / 6.0, mask=mask)

    @triton.jit
    def _query_l2_kcenter_seed_kernel(
        q_ptr,
        seed_ptr,
        Q_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        SKETCH_D: tl.constexpr,
        CANDIDATES: tl.constexpr,
        BLOCK_CANDIDATES: tl.constexpr,
        C: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Anchored local farthest-point selection using only query L2."""

        tl.static_assert(CANDIDATES == 6)
        tl.static_assert(C == 4)
        tl.static_assert(D_HEAD == 128)
        tl.static_assert(SKETCH_D <= D_HEAD)

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS

        column_block = block_id % tl.cdiv(COLS, BLOCK_J)
        block_id //= tl.cdiv(COLS, BLOCK_J)
        key_row = block_id % ROWS
        query_outer = block_id // ROWS

        columns = column_block * BLOCK_J + tl.arange(0, BLOCK_J)
        candidate_slots = tl.arange(0, BLOCK_CANDIDATES)
        dims = tl.arange(0, BLOCK_D)
        column_mask = columns < COLS
        candidate_mask = candidate_slots < CANDIDATES
        candidate_start = tl.minimum(
            tl.maximum(key_row - CANDIDATES // 2, 0),
            ROWS - CANDIDATES,
        )
        candidate_rows = tl.minimum(
            candidate_start + candidate_slots, ROWS - 1
        )
        query_offset = (
            (
                (
                    ((batch * Q_OUTER + query_outer) * ROWS
                     + candidate_rows[None, :, None])
                    * COLS
                    + columns[:, None, None]
                )
                * HEADS
                + head
            )
            * D_HEAD
            + dims[None, None, :]
        )
        candidates = tl.load(
            q_ptr + query_offset,
            mask=(
                column_mask[:, None, None]
                & candidate_mask[None, :, None]
                & (dims[None, None, :] < SKETCH_D)
            ),
            other=0.0,
        ).to(tl.float32)

        anchor = key_row - candidate_start
        selected = tl.full((BLOCK_J,), anchor, tl.int32)
        used = (~candidate_mask[None, :]) | (
            candidate_slots[None, :] == selected[:, None]
        )
        selected_vector = tl.sum(
            tl.where(
                candidate_slots[None, :, None] == selected[:, None, None],
                candidates,
                0.0,
            ),
            axis=1,
        )
        minimum = tl.sum(
            (candidates - selected_vector[:, None, :])
            * (candidates - selected_vector[:, None, :]),
            axis=2,
        )

        seed_slots = tl.arange(0, C)
        seeds = selected[:, None] * (seed_slots[None, :] == 0).to(tl.int32)
        for step in tl.static_range(1, C):
            score = tl.where(used, -float("inf"), minimum)
            _, selected = tl.max(
                score,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            seeds += selected[:, None] * (
                seed_slots[None, :] == step
            ).to(tl.int32)
            used |= candidate_slots[None, :] == selected[:, None]
            selected_vector = tl.sum(
                tl.where(
                    candidate_slots[None, :, None]
                    == selected[:, None, None],
                    candidates,
                    0.0,
                ),
                axis=1,
            )
            distance = tl.sum(
                (candidates - selected_vector[:, None, :])
                * (candidates - selected_vector[:, None, :]),
                axis=2,
            )
            minimum = tl.minimum(minimum, distance)

        output_offset = (
            (
                (
                    (
                        ((batch * Q_OUTER + query_outer) * ROWS + key_row)
                        * COLS
                        + columns[:, None]
                    )
                    * HEADS
                    + head
                )
                * C
            )
            + seed_slots[None, :]
        )
        tl.store(
            seed_ptr + output_offset,
            (candidate_start + seeds).to(tl.uint8),
            mask=column_mask[:, None],
        )

    @triton.jit
    def _merge_kernel(
        q_ptr, k_ptr, v_ptr, seed_ptr,
        merged_k_ptr, merged_v_ptr, bias_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
        J_TILES: tl.constexpr, C: tl.constexpr,
        SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """One program per (query frame, key frame, key row, query-column tile)."""
        tl.static_assert(C == 4, "hard C4 keeps exactly four merged tokens")

        block_id = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS

        j_tile = block_id % J_TILES
        tmp = block_id // J_TILES
        query_outer = tmp % Q_OUTER
        tmp = tmp // Q_OUTER
        key_outer = tmp % K_OUTER
        key_row = tmp // K_OUTER

        js = j_tile * BLOCK_J + tl.arange(0, BLOCK_J)
        ls = tl.arange(0, BLOCK_L)
        dims = tl.arange(0, BLOCK_D)
        j_mask = js < COLS
        key_mask = ls < COLS

        k_base = (batch * K_OUTER + key_outer) * ROWS * COLS * HEADS * D_HEAD
        key_off = k_base + (key_row * COLS + ls[:, None]) * HEADS * D_HEAD
        key = tl.load(k_ptr + key_off + head * D_HEAD + dims[None, :],
                      mask=key_mask[:, None], other=0.0, eviction_policy="evict_last")

        # Seed rows for this (block, head): four int32 per query column. This is
        # the only difference from the fixed-seed kernel, which computed
        # seed_start = clamp(key_row - 1, 0, ROWS - C) inline.
        seed_base = (((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                       * ROWS + key_row) * COLS + js) * HEADS + head) * C
        row0 = tl.load(seed_ptr + seed_base + 0, mask=j_mask, other=0)
        row1 = tl.load(seed_ptr + seed_base + 1, mask=j_mask, other=0)
        row2 = tl.load(seed_ptr + seed_base + 2, mask=j_mask, other=0)
        row3 = tl.load(seed_ptr + seed_base + 3, mask=j_mask, other=0)

        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
        q_col = q_base + (js[:, None] * HEADS + head) * D_HEAD + dims[None, :]
        q_stride = COLS * HEADS * D_HEAD

        # Work in log2 units: same softmax, but exp2/log2 avoid multiplying full
        # tiles by log2(e). Two Q tiles are consumed before the other two load so
        # the compiler can reuse their shared-memory buffers.
        scale_log2 = SCALE * 1.4426950408889634
        q0 = tl.load(q_ptr + q_col + row0[:, None] * q_stride, mask=j_mask[:, None], other=0.0)
        q1 = tl.load(q_ptr + q_col + row1[:, None] * q_stride, mask=j_mask[:, None], other=0.0)
        s0 = tl.dot(q0, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2
        s1 = tl.dot(q1, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2
        q2 = tl.load(q_ptr + q_col + row2[:, None] * q_stride, mask=j_mask[:, None], other=0.0)
        q3 = tl.load(q_ptr + q_col + row3[:, None] * q_stride, mask=j_mask[:, None], other=0.0)
        s2 = tl.dot(q2, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2
        s3 = tl.dot(q3, tl.trans(key), input_precision=INPUT_PRECISION) * scale_log2

        real_mask = j_mask[:, None] & key_mask[None, :]
        col_mask = key_mask[None, :]
        s0 = tl.where(col_mask, s0, -float("inf"))
        s1 = tl.where(col_mask, s1, -float("inf"))
        s2 = tl.where(col_mask, s2, -float("inf"))
        s3 = tl.where(col_mask, s3, -float("inf"))

        # Key-softmax assignment: normalise each seed over its keys, then every
        # key joins the seed giving it the largest normalised score.
        m0 = tl.max(s0, axis=1)
        m1 = tl.max(s1, axis=1)
        m2 = tl.max(s2, axis=1)
        m3 = tl.max(s3, axis=1)
        n0 = m0 + tl.log2(tl.sum(tl.exp2(s0 - m0[:, None]), axis=1))
        n1 = m1 + tl.log2(tl.sum(tl.exp2(s1 - m1[:, None]), axis=1))
        n2 = m2 + tl.log2(tl.sum(tl.exp2(s2 - m2[:, None]), axis=1))
        n3 = m3 + tl.log2(tl.sum(tl.exp2(s3 - m3[:, None]), axis=1))

        winner = s0 - n0[:, None]
        route = tl.full((BLOCK_J, BLOCK_L), 0, tl.int32)
        cand = s1 - n1[:, None]
        take = cand > winner
        winner = tl.where(take, cand, winner)
        route = tl.where(take, 1, route)
        cand = s2 - n2[:, None]
        take = cand > winner
        winner = tl.where(take, cand, winner)
        route = tl.where(take, 2, route)
        cand = s3 - n3[:, None]
        take = cand > winner
        winner = tl.where(take, cand, winner)
        route = tl.where(take, 3, route)
        route = tl.where(real_mask, route, -1)
        winner_prob = tl.where(real_mask, tl.exp2(winner), 0.0)

        value = tl.load(v_ptr + key_off + head * D_HEAD + dims[None, :],
                        mask=key_mask[:, None], other=0.0, eviction_policy="evict_last")

        for seed_id in tl.static_range(0, 4):
            route_mask = route == seed_id
            probs = tl.where(route_mask, winner_prob, 0.0)
            row_sum = tl.sum(probs, axis=1)
            has_key = row_sum > 0.0
            safe = tl.where(has_key, row_sum, 1.0)
            weights = probs / safe[:, None]
            expected = tl.sum(tl.where(route_mask, weights * winner, 0.0), axis=1)
            # Entropy bias: exp of it is the cluster's effective size, which is
            # what collapsing the cluster to one token would otherwise lose.
            route_bias = tl.where(
                has_key, (expected - tl.log2(safe)) * 0.6931471805599453, float("inf")
            )

            bias_off = (((((batch * HEADS + head) * Q_OUTER + query_outer) * K_OUTER
                          + key_outer) * ROWS + key_row) * COLS + js) * C + seed_id
            tl.store(bias_ptr + bias_off, route_bias, mask=j_mask,
                     eviction_policy="evict_first")

            store_base = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                            * ROWS + key_row) * COLS + js[:, None]) * C) + seed_id)
            merged_key = tl.dot(weights.to(key.dtype), key, input_precision=INPUT_PRECISION)
            tl.store(merged_k_ptr + ((store_base * HEADS + head) * D_HEAD) + dims[None, :],
                     merged_key, mask=j_mask[:, None], eviction_policy="evict_first")
            merged_value = tl.dot(weights.to(value.dtype), value, input_precision=INPUT_PRECISION)
            tl.store(merged_v_ptr + ((store_base * HEADS + head) * D_HEAD) + dims[None, :],
                     merged_value, mask=j_mask[:, None], eviction_policy="evict_first")

    @triton.jit
    def _route_kernel(
        q_ptr, merged_k_ptr, merged_v_ptr, bias_ptr, out_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        BLOCK_I: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
        C: tl.constexpr, SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """Full attention over the merged tokens. Unchanged by seed selection."""
        i_start = tl.program_id(0) * BLOCK_I
        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        query_outer = query_group // COLS
        query_col = query_group % COLS

        rows = i_start + tl.arange(0, BLOCK_I)
        tokens = tl.arange(0, BLOCK_K * C)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < ROWS

        q_base = (((batch * Q_OUTER + query_outer) * ROWS + rows[:, None]) * COLS + query_col)
        query = tl.load(q_ptr + ((q_base * HEADS + head) * D_HEAD + dims[None, :]),
                        mask=row_mask[:, None], other=0.0)

        row_max = tl.full((BLOCK_I,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_I,), tl.float32)
        acc = tl.zeros((BLOCK_I, BLOCK_D), tl.float32)

        for key_outer in tl.range(0, K_OUTER):
            for key_start in tl.static_range(0, ROWS, BLOCK_K):
                key_row = key_start + tokens // C
                seed_id = tokens - (tokens // C) * C
                key_mask = key_row < ROWS
                merged_base = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                                 * ROWS + key_row[:, None]) * COLS + query_col) * C)
                               + seed_id[:, None])
                merged_key = tl.load(
                    merged_k_ptr + (((merged_base * HEADS + head) * D_HEAD) + dims[None, :]),
                    mask=key_mask[:, None], other=0.0, eviction_policy="evict_first")
                merged_value = tl.load(
                    merged_v_ptr + (((merged_base * HEADS + head) * D_HEAD) + dims[None, :]),
                    mask=key_mask[:, None], other=0.0, eviction_policy="evict_first")
                bias_off = (((((batch * HEADS + head) * Q_OUTER + query_outer) * K_OUTER
                              + key_outer) * ROWS + key_row) * COLS + query_col) * C + seed_id
                bias = tl.load(bias_ptr + bias_off, mask=key_mask, other=float("inf"),
                               eviction_policy="evict_first")

                logits = tl.dot(query, tl.trans(merged_key), input_precision=INPUT_PRECISION)
                logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
                logits = tl.where(key_mask[None, :], logits, -float("inf"))

                new_max = tl.maximum(row_max, tl.max(logits, axis=1))
                probs = tl.exp2(logits - new_max[:, None])
                rescale = tl.exp2(row_max - new_max)
                acc = acc * rescale[:, None] + tl.dot(
                    probs.to(query.dtype), merged_value, input_precision=INPUT_PRECISION)
                row_sum = row_sum * rescale + tl.sum(probs, axis=1)
                row_max = new_max

        out = acc / row_sum[:, None]
        tl.store(out_ptr + ((q_base * HEADS + head) * D_HEAD + dims[None, :]), out,
                 mask=row_mask[:, None])


def temporal_simpson_query_triton(q, *, center_frame=10):
    """Approximate the temporal entropy barycenter with Simpson's 1:4:1 rule."""

    if triton is None:
        raise RuntimeError("Triton is required for temporal-Simpson queries.")
    if q.ndim != 6:
        raise ValueError("query must have shape [B, T, row, column, H, D]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("temporal Simpson requires bfloat16 or float16 queries")
    q = q.contiguous()
    batch, frames, rows, columns, heads, head_dim = q.shape
    center_frame = int(center_frame)
    if frames % 2 != 1 or center_frame != frames // 2:
        raise ValueError("temporal Simpson requires the middle frame as its center")

    output = torch.empty(
        (batch, 1, rows, columns, heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    frame_stride = rows * columns * heads * head_dim
    block = 256
    _temporal_simpson_query_kernel[(triton.cdiv(output.numel(), block),)](
        q,
        output,
        output.numel(),
        frame_stride,
        frames,
        center_frame,
        BLOCK=block,
        num_warps=8,
        num_stages=1,
    )
    return output


def query_l2_kcenter_seed_rows_triton(
    q, *, candidate_count=6, sketch_dimensions=None, block_j=4
):
    """Return native query-only C4 rows as ``[B, A, row, column, H, 4]``."""

    if triton is None:
        raise RuntimeError("Triton is required for query L2 k-center seeds.")
    if q.ndim != 6:
        raise ValueError("query must have shape [B, A, row, column, H, D]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("query L2 k-center requires bfloat16 or float16 queries")
    q = q.contiguous()
    batch, query_outer, rows, columns, heads, head_dim = q.shape
    if (rows, columns, head_dim) != (45, 80, 128):
        raise ValueError("query L2 k-center is specialized for 45x80, D=128")
    if int(candidate_count) != 6:
        raise ValueError("the production query L2 k-center uses six candidates")
    sketch_dimensions = int(sketch_dimensions or head_dim)
    if sketch_dimensions not in {16, 32, 64, 128}:
        raise ValueError("sketch_dimensions must be 16, 32, 64, or 128")
    if int(block_j) not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("block_j must be one of 1, 2, 4, 8, 16, or 32")

    seed_rows = torch.empty(
        (batch, query_outer, rows, columns, heads, C4),
        device=q.device,
        dtype=torch.uint8,
    )
    column_blocks = triton.cdiv(columns, int(block_j))
    _query_l2_kcenter_seed_kernel[
        (query_outer * rows * column_blocks, batch * heads)
    ](
        q,
        seed_rows,
        query_outer,
        rows,
        columns,
        heads,
        head_dim,
        SKETCH_D=sketch_dimensions,
        CANDIDATES=int(candidate_count),
        BLOCK_CANDIDATES=triton.next_power_of_2(int(candidate_count)),
        C=C4,
        BLOCK_J=int(block_j),
        BLOCK_D=triton.next_power_of_2(sketch_dimensions),
        num_warps=4,
        num_stages=1,
    )
    return seed_rows


def adaptive_seed_attention_triton(q, k, v, seed_table, scale=None, output=None,
                                   input_precision="ieee"):
    """Hard C4 with a caller-supplied seed table.

    q:          [B, A, I, J, H, D] contiguous
    k, v:       [B, F, K, L, H, D] contiguous
    seed_table: [B, A, F, K, J, H, 4] int32 query-row indices, from
                ``adaptive_seed_sdpa.adaptive_seed_row_indices``
    """
    if triton is None:
        raise RuntimeError("Triton is required for adaptive_seed_attention_triton.")
    # The tile sizes in SHAPE_CONFIG are sized for 16-bit operands. FP32 inputs
    # do not merely run slower: Triton spends minutes in codegen and never
    # reaches the GPU, so fail fast rather than appear to hang.
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"adaptive_seed_attention_triton expects bfloat16 or float16 Q/K/V, "
            f"got {q.dtype}. FP32 needs retuned tiles.")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K and V must share a dtype.")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    seed_table = seed_table.to(torch.int32).contiguous()

    batch, query_outer, rows, cols, heads, dim = q.shape
    key_outer = k.size(1)
    if (rows, cols) not in SHAPE_CONFIG:
        raise ValueError(
            f"no tuned tile config for a {rows}x{cols} grid; "
            f"known shapes are {sorted(SHAPE_CONFIG)}")
    block_j, block_l, block_i, block_k, merge_warps, route_warps = SHAPE_CONFIG[(rows, cols)]
    if block_l < cols:
        raise ValueError(f"BLOCK_L={block_l} cannot cover {cols} key columns")
    expected = (batch, query_outer, key_outer, rows, cols, heads, C4)
    if tuple(seed_table.shape) != expected:
        raise ValueError(f"seed_table must be {expected}, got {tuple(seed_table.shape)}")
    # Triton indexes with int32. The merged K/V intermediate is the largest
    # tensor here and its offsets overflow silently into an illegal memory
    # access rather than an error, so check it up front. The shipped 480p
    # config avoids this with head_chunk=4; 12 heads at 21 frames does not fit.
    merged_elements = batch * query_outer * key_outer * rows * cols * C4 * heads * dim
    if merged_elements > 2 ** 31 - 1:
        raise ValueError(
            f"merged K/V would need {merged_elements:,} elements, past the int32 "
            f"offset limit ({2**31 - 1:,}). Chunk heads or query frames: at "
            f"{rows}x{cols} with {query_outer} query frames, "
            f"{(2**31 - 1) // max(1, merged_elements // heads)} heads fit per call.")

    if scale is None:
        scale = dim ** -0.5
    block_d = triton.next_power_of_2(dim)
    j_tiles = triton.cdiv(cols, block_j)

    merged_shape = (batch, query_outer, key_outer, rows, cols, C4, heads, dim)
    merged_k = torch.empty(merged_shape, device=q.device, dtype=q.dtype)
    merged_v = torch.empty(merged_shape, device=q.device, dtype=q.dtype)
    bias = torch.empty((batch, heads, query_outer, key_outer, rows, cols, C4),
                       device=q.device, dtype=torch.float32)
    if output is None:
        output = torch.empty_like(q)

    _merge_kernel[(query_outer * key_outer * rows * j_tiles, batch * heads)](
        q, k, v, seed_table, merged_k, merged_v, bias,
        query_outer, key_outer, rows, cols, heads, dim,
        BLOCK_J=block_j, BLOCK_L=block_l, BLOCK_D=block_d,
        J_TILES=j_tiles, C=C4, SCALE=float(scale),
        INPUT_PRECISION=input_precision, num_warps=merge_warps, num_stages=2)

    _route_kernel[(triton.cdiv(rows, block_i), batch * heads, query_outer * cols)](
        q, merged_k, merged_v, bias, output,
        query_outer, key_outer, rows, cols, heads, dim,
        BLOCK_I=block_i, BLOCK_K=block_k, BLOCK_D=block_d,
        C=C4, SCALE=float(scale), INPUT_PRECISION=input_precision,
        num_warps=route_warps, num_stages=2)
    return output


def adaptive_seed_attention(q, k, v, scale=None, *, alpha=0.3, candidate_count=None,
                            probe_count=16, value_dimensions=32, seed_table=None,
                            output=None):
    """End to end: select seeds in PyTorch, then run both Triton stages.

    Pass ``seed_table`` to reuse a cached table and skip selection entirely.
    Measured seed stability across denoising steps is low outside the first
    layer (7-11% exact set match between consecutive steps), so reuse is a
    quality/cost tradeoff to measure, not a free win.
    """
    from .adaptive_seed_pytorch import adaptive_seed_row_indices

    if scale is None:
        scale = q.size(-1) ** -0.5
    if seed_table is None:
        seed_table = adaptive_seed_row_indices(
            q, k, v, scale, alpha=alpha, candidate_count=candidate_count,
            probe_count=probe_count, value_dimensions=value_dimensions)
    return adaptive_seed_attention_triton(q, k, v, seed_table, scale=scale, output=output)


# ---------------------------------------------------------------------------
# Triton seed selection.
#
# Modelled on upstream's _hard_c4_720p_query_response_seed_kernel: same program
# decomposition and the same trick of running Gram-Schmidt as masked reductions
# over a [BLOCK_J, BLOCK_ROWS, FEATURES] tile. Three things differ, each
# measured on real Wan QKV (16 step/layer points, 480p):
#
#   probe columns    upstream uses eight hardcoded positions. Choosing them by
#                    top-k over anchor responses is worth 13.0% rel-L2, so this
#                    kernel selects them. They are shared across the query
#                    column tile, which costs 6.7% but keeps the candidate x key
#                    product a single tl.dot; per-column probes would turn it
#                    into a per-column loop.
#   residual space   upstream carries 8-dim probability residuals; this carries
#                    value-output profiles, what a candidate would retrieve.
#   importance       objective is scaled by exp(2*alpha*z), z being the z-scored
#                    log attention mass. Worth 9.9%. Scaling commutes with the
#                    whole Gram-Schmidt because every basis is normalised, so it
#                    applies to the objective and never touches the residual.
#
# Probes are applied as a mask over all keys rather than gathered into a [P, D]
# tile. That costs extra matmul on masked-out columns but avoids a data
# dependent gather, which is what made two earlier selector kernels slower than
# PyTorch.
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _span_seed_kernel(
        q_ptr, k_ptr, v_ptr, seed_row_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr, TOP_PROBES: tl.constexpr,
        UNIFORM_PROBES: tl.constexpr, VDIMS: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr, C: tl.constexpr,
        ALPHA: tl.constexpr, SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """One program per (query frame, key row, query-column tile); loops key frames."""
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        j_blocks: tl.constexpr = (COLS + BLOCK_J - 1) // BLOCK_J
        j_block = pid % j_blocks
        key_row = (pid // j_blocks) % ROWS
        query_outer = (pid // j_blocks) // ROWS

        dims = tl.arange(0, BLOCK_D)
        ls = tl.arange(0, BLOCK_L)
        key_valid = ls < COLS
        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
        q_stride = COLS * HEADS * D_HEAD
        col0 = j_block * BLOCK_J
        log2e: tl.constexpr = 1.4426950408889634

        # Candidate queries depend on (query frame, row, column), never on the
        # key frame, so this 64 KiB tile loads once and is reused across all
        # K_OUTER frames. Putting K_OUTER in the grid instead reloaded it every
        # frame and was 72% of the kernel's memory traffic.
        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        j_slot = flat // BLOCK_ROWS
        c_slot = flat % BLOCK_ROWS
        cols = col0 + j_slot
        start = tl.minimum(tl.maximum(key_row - CANDIDATES // 2, 0), ROWS - CANDIDATES)
        cand_rows = start + c_slot
        cand_valid = (cols < COLS) & (c_slot < CANDIDATES)
        cand = tl.load(
            q_ptr + q_base + cand_rows[:, None] * q_stride
            + (cols[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=cand_valid[:, None], other=0.0)

        cand_rows_2d = tl.reshape(cand_rows, (BLOCK_J, BLOCK_ROWS))
        valid2 = tl.reshape(cand_valid, (BLOCK_J, BLOCK_ROWS))
        n_valid = tl.sum(valid2.to(tl.float32), axis=1)
        cand_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        center_index = key_row - start
        slot = tl.arange(0, C)[None, :]
        pslots = tl.arange(0, TOP_PROBES + UNIFORM_PROBES)
        ustride: tl.constexpr = COLS // UNIFORM_PROBES
        uniform_hit = (ls % ustride == 0) & key_valid
        out_cols = col0 + tl.arange(0, BLOCK_J)
        out_mask = (out_cols < COLS)[:, None]

        # Anchor rows depend only on (query_outer, key_row) and are the same
        # for every key frame, so their loads and the reciprocal below hoist
        # out of the K_OUTER loop. Anchor SCORES still need the per-frame key
        # tile and cannot be hoisted.
        # a_i's offset is a compile-time constant (static_range yields plain
        # Python ints), so each anchor row is just a scalar expression -- no
        # need to build and scalar-index a small tensor.
        row0 = tl.minimum(tl.maximum(key_row - 2, 0), ROWS - 1)
        row1 = tl.minimum(tl.maximum(key_row + 0, 0), ROWS - 1)
        row2 = tl.minimum(tl.maximum(key_row + 1, 0), ROWS - 1)
        row3 = tl.minimum(tl.maximum(key_row + 3, 0), ROWS - 1)
        aq0 = tl.load(q_ptr + q_base + row0 * q_stride
                      + (col0 * HEADS + head) * D_HEAD + dims, mask=dims < D_HEAD, other=0.0)
        aq1 = tl.load(q_ptr + q_base + row1 * q_stride
                      + (col0 * HEADS + head) * D_HEAD + dims, mask=dims < D_HEAD, other=0.0)
        aq2 = tl.load(q_ptr + q_base + row2 * q_stride
                      + (col0 * HEADS + head) * D_HEAD + dims, mask=dims < D_HEAD, other=0.0)
        aq3 = tl.load(q_ptr + q_base + row3 * q_stride
                      + (col0 * HEADS + head) * D_HEAD + dims, mask=dims < D_HEAD, other=0.0)
        inv_n_valid = 1.0 / n_valid

        for key_outer in tl.range(0, K_OUTER):
            k_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS * HEADS * D_HEAD
            key_off = k_base + (ls[:, None] * HEADS + head) * D_HEAD + dims[None, :]
            keys = tl.load(k_ptr + key_off, mask=key_valid[:, None], other=0.0)

            # Probe columns, shared across this query-column tile: four fixed
            # anchor rows score every key, and the strongest columns by
            # normalised anchor response become probes. Worth 13% rel-L2 over
            # the fixed uniform columns upstream uses. Kept as four manual
            # reductions rather than one tl.dot: batching the anchors gives
            # N=4, below tl.dot's practical minimum of 16, so it would fall
            # back to scalar FMA and buy nothing.
            resp0 = tl.sum(aq0[None, :] * keys, axis=1) * (SCALE * log2e)
            resp0 = tl.where(key_valid, resp0, -float("inf"))
            rmax0 = tl.max(resp0, axis=0)
            importance = resp0 - (rmax0 + tl.log2(tl.sum(tl.exp2(resp0 - rmax0), axis=0)))

            resp1 = tl.sum(aq1[None, :] * keys, axis=1) * (SCALE * log2e)
            resp1 = tl.where(key_valid, resp1, -float("inf"))
            rmax1 = tl.max(resp1, axis=0)
            importance = tl.maximum(
                importance, resp1 - (rmax1 + tl.log2(tl.sum(tl.exp2(resp1 - rmax1), axis=0))))

            resp2 = tl.sum(aq2[None, :] * keys, axis=1) * (SCALE * log2e)
            resp2 = tl.where(key_valid, resp2, -float("inf"))
            rmax2 = tl.max(resp2, axis=0)
            importance = tl.maximum(
                importance, resp2 - (rmax2 + tl.log2(tl.sum(tl.exp2(resp2 - rmax2), axis=0))))

            resp3 = tl.sum(aq3[None, :] * keys, axis=1) * (SCALE * log2e)
            resp3 = tl.where(key_valid, resp3, -float("inf"))
            rmax3 = tl.max(resp3, axis=0)
            importance = tl.maximum(
                importance, resp3 - (rmax3 + tl.log2(tl.sum(tl.exp2(resp3 - rmax3), axis=0))))

            probe_cols = tl.where(pslots >= TOP_PROBES, (pslots - TOP_PROBES) * ustride, 0)
            work = tl.where(key_valid & ~uniform_hit, importance, -float("inf"))
            for r in tl.static_range(0, TOP_PROBES):
                _, best = tl.max(work, axis=0, return_indices=True)
                probe_cols = tl.where(pslots == r, best, probe_cols)
                work = tl.where(ls == best, -float("inf"), work)

            # probe_cols is drawn from ls (< COLS via key_valid) or from the
            # uniform stride, which floor-divides COLS -- always in bounds --
            # but mask anyway: probe_off is a runtime, data-dependent index,
            # not a value provably bounded at compile time.
            probe_mask = probe_cols[:, None] < COLS
            probe_off = k_base + (probe_cols[:, None] * HEADS + head) * D_HEAD + dims[None, :]
            probe_keys = tl.load(k_ptr + probe_off, mask=probe_mask, other=0.0)
            tl.static_assert(BLOCK_L >= 16 and D_HEAD >= 16 and (TOP_PROBES + UNIFORM_PROBES) >= 16,
                             "tl.dot needs every operand dim >= 16")
            logits = tl.dot(cand, tl.trans(probe_keys), input_precision=INPUT_PRECISION)
            logits = logits * (SCALE * log2e)
            row_max = tl.max(logits, axis=1)
            expd = tl.exp2(logits - row_max[:, None])
            denom = tl.sum(expd, axis=1)
            log_mass = row_max + tl.log2(denom)
            prob = expd * (1.0 / denom)[:, None]

            vals = tl.load(v_ptr + probe_off, mask=probe_mask, other=0.0)
            vals = tl.reshape(vals, (TOP_PROBES + UNIFORM_PROBES, VDIMS, D_HEAD // VDIMS))
            vals = tl.sum(vals, axis=2) * (1.0 / (D_HEAD // VDIMS))
            profile = tl.dot(prob.to(vals.dtype), vals, input_precision=INPUT_PRECISION)
            profile = tl.reshape(profile, (BLOCK_J, BLOCK_ROWS, VDIMS)).to(tl.float32)
            residual = profile - tl.sum(profile, axis=2)[:, :, None] * (1.0 / VDIMS)

            lm = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS))
            mu = tl.sum(tl.where(valid2, lm, 0.0), axis=1) * inv_n_valid
            dev = lm - mu[:, None]
            var = tl.sum(tl.where(valid2, dev * dev, 0.0), axis=1) * inv_n_valid
            z = dev * tl.rsqrt(var[:, None] + 1e-6)
            # Scaling commutes with the whole Gram-Schmidt because every basis
            # is normalised, so the weight applies once to the objective.
            weight = tl.exp2((2.0 * ALPHA * log2e) * z)

            used = cand_ids == center_index
            seed0 = tl.sum(tl.where(used, cand_rows_2d, 0), axis=1)
            seeds = seed0[:, None] * (slot == 0).to(tl.int32)
            basis_src = tl.sum(tl.where(used[:, :, None], residual, 0.0), axis=1)
            for step in tl.static_range(1, C):
                basis = basis_src * tl.rsqrt(tl.sum(basis_src * basis_src, axis=1) + 1e-12)[:, None]
                proj = tl.sum(residual * basis[:, None, :], axis=2)
                residual = residual - proj[:, :, None] * basis[:, None, :]
                obj = tl.sum(residual * residual, axis=2) * weight
                obj = tl.where(valid2 & ~used, obj, -1.0)
                _, pick = tl.max(obj, axis=1, return_indices=True,
                                 return_indices_tie_break_left=True)
                hit = cand_ids == pick[:, None]
                used = used | hit
                seeds = seeds + (start + pick)[:, None] * (slot == step).to(tl.int32)
                basis_src = tl.sum(tl.where(hit[:, :, None], residual, 0.0), axis=1)

            out_off = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer) * ROWS
                          + key_row) * COLS + out_cols[:, None]) * HEADS + head) * C + slot)
            tl.store(seed_row_ptr + out_off, seeds.to(tl.int32), mask=out_mask)

    @triton.jit
    def _span_gs_temporal_consensus2_kernel(
        q_ptr, k_ptr, v_ptr, probe_ptr, seed_row_ptr,
        K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr, TOP_PROBES: tl.constexpr,
        UNIFORM_PROBES: tl.constexpr, VDIMS: tl.constexpr,
        PROBE_PAD: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_ROWS: tl.constexpr, N_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr, C: tl.constexpr,
        ALPHA: tl.constexpr, SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """Choose one seed table from the joint response span of two frames."""
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        j_block = pid % N_GROUPS
        key_row = (pid // N_GROUPS) % ROWS

        dims = tl.arange(0, BLOCK_D)
        q_frame_stride: tl.constexpr = ROWS * COLS * HEADS * D_HEAD
        q_stride: tl.constexpr = COLS * HEADS * D_HEAD
        q_base = batch * 2 * q_frame_stride
        log2e: tl.constexpr = 1.4426950408889634
        P: tl.constexpr = TOP_PROBES + UNIFORM_PROBES

        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        j_slot = flat // BLOCK_ROWS
        c_slot = flat % BLOCK_ROWS
        cols = j_block * BLOCK_J + j_slot
        start = tl.minimum(tl.maximum(key_row - CANDIDATES // 2, 0), ROWS - CANDIDATES)
        cand_rows = start + c_slot
        cand_valid = (cols < COLS) & (c_slot < CANDIDATES)
        cand_offset = (
            cand_rows[:, None] * q_stride
            + (cols[:, None] * HEADS + head) * D_HEAD
            + dims[None, :]
        )
        cand0 = tl.load(
            q_ptr + q_base + cand_offset,
            mask=cand_valid[:, None], other=0.0,
        )
        cand1 = tl.load(
            q_ptr + q_base + q_frame_stride + cand_offset,
            mask=cand_valid[:, None], other=0.0,
        )

        cand_rows_2d = tl.reshape(cand_rows, (BLOCK_J, BLOCK_ROWS))
        valid2 = tl.reshape(cand_valid, (BLOCK_J, BLOCK_ROWS))
        n_valid = tl.sum(valid2.to(tl.float32), axis=1)
        inv_n_valid = 1.0 / n_valid
        cand_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        center_index = key_row - start
        slot = tl.arange(0, C)[None, :]
        pslots = tl.arange(0, PROBE_PAD)
        out_cols = j_block * BLOCK_J + tl.arange(0, BLOCK_J)
        out_mask = (out_cols < COLS)[:, None]

        for key_outer in tl.range(0, K_OUTER):
            k_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS * HEADS * D_HEAD
            probe_tail = (
                (((key_outer * ROWS + key_row) * N_GROUPS + j_block)
                  * HEADS + head) * PROBE_PAD
                + pslots
            )
            probe_batch_stride: tl.constexpr = 2 * K_OUTER * ROWS * N_GROUPS * HEADS * PROBE_PAD
            probe_frame_stride: tl.constexpr = K_OUTER * ROWS * N_GROUPS * HEADS * PROBE_PAD
            probe_cols0 = tl.load(
                probe_ptr + batch * probe_batch_stride + probe_tail
            )
            probe_cols1 = tl.load(
                probe_ptr + batch * probe_batch_stride + probe_frame_stride + probe_tail
            )
            probe_mask0 = (pslots[:, None] < P) & (probe_cols0[:, None] < COLS)
            probe_mask1 = (pslots[:, None] < P) & (probe_cols1[:, None] < COLS)
            probe_off0 = (
                k_base
                + (probe_cols0[:, None] * HEADS + head) * D_HEAD
                + dims[None, :]
            )
            probe_off1 = (
                k_base
                + (probe_cols1[:, None] * HEADS + head) * D_HEAD
                + dims[None, :]
            )
            probe_keys0 = tl.load(
                k_ptr + probe_off0, mask=probe_mask0, other=0.0
            )
            probe_keys1 = tl.load(
                k_ptr + probe_off1, mask=probe_mask1, other=0.0
            )
            logits0 = tl.dot(
                cand0, tl.trans(probe_keys0), input_precision=INPUT_PRECISION
            ) * (SCALE * log2e)
            logits1 = tl.dot(
                cand1, tl.trans(probe_keys1), input_precision=INPUT_PRECISION
            ) * (SCALE * log2e)
            logits0 = tl.where(pslots[None, :] < P, logits0, -float("inf"))
            logits1 = tl.where(pslots[None, :] < P, logits1, -float("inf"))

            row_max0 = tl.max(logits0, axis=1)
            row_max1 = tl.max(logits1, axis=1)
            expd0 = tl.exp2(logits0 - row_max0[:, None])
            expd1 = tl.exp2(logits1 - row_max1[:, None])
            denom0 = tl.sum(expd0, axis=1)
            denom1 = tl.sum(expd1, axis=1)
            log_mass0 = row_max0 + tl.log2(denom0)
            log_mass1 = row_max1 + tl.log2(denom1)
            prob0 = expd0 * (1.0 / denom0)[:, None]
            prob1 = expd1 * (1.0 / denom1)[:, None]

            vals0 = tl.load(v_ptr + probe_off0, mask=probe_mask0, other=0.0)
            vals1 = tl.load(v_ptr + probe_off1, mask=probe_mask1, other=0.0)
            vals0 = tl.reshape(vals0, (PROBE_PAD, VDIMS, D_HEAD // VDIMS))
            vals1 = tl.reshape(vals1, (PROBE_PAD, VDIMS, D_HEAD // VDIMS))
            vals0 = tl.sum(vals0, axis=2) * (1.0 / (D_HEAD // VDIMS))
            vals1 = tl.sum(vals1, axis=2) * (1.0 / (D_HEAD // VDIMS))
            profile0 = tl.dot(
                prob0.to(vals0.dtype), vals0, input_precision=INPUT_PRECISION
            )
            profile1 = tl.dot(
                prob1.to(vals1.dtype), vals1, input_precision=INPUT_PRECISION
            )
            profile0 = tl.reshape(
                profile0, (BLOCK_J, BLOCK_ROWS, VDIMS)
            ).to(tl.float32)
            profile1 = tl.reshape(
                profile1, (BLOCK_J, BLOCK_ROWS, VDIMS)
            ).to(tl.float32)
            residual0 = profile0 - tl.sum(profile0, axis=2)[:, :, None] * (1.0 / VDIMS)
            residual1 = profile1 - tl.sum(profile1, axis=2)[:, :, None] * (1.0 / VDIMS)

            lm0 = tl.reshape(log_mass0, (BLOCK_J, BLOCK_ROWS))
            lm1 = tl.reshape(log_mass1, (BLOCK_J, BLOCK_ROWS))
            mu0 = tl.sum(tl.where(valid2, lm0, 0.0), axis=1) * inv_n_valid
            mu1 = tl.sum(tl.where(valid2, lm1, 0.0), axis=1) * inv_n_valid
            dev0 = lm0 - mu0[:, None]
            dev1 = lm1 - mu1[:, None]
            var0 = tl.sum(tl.where(valid2, dev0 * dev0, 0.0), axis=1) * inv_n_valid
            var1 = tl.sum(tl.where(valid2, dev1 * dev1, 0.0), axis=1) * inv_n_valid
            z0 = dev0 * tl.rsqrt(var0[:, None] + 1e-6)
            z1 = dev1 * tl.rsqrt(var1[:, None] + 1e-6)
            weight = tl.exp2((ALPHA * log2e) * (z0 + z1))

            used = cand_ids == center_index
            seed0 = tl.sum(tl.where(used, cand_rows_2d, 0), axis=1)
            seeds = seed0[:, None] * (slot == 0).to(tl.int32)
            basis_src0 = tl.sum(
                tl.where(used[:, :, None], residual0, 0.0), axis=1
            )
            basis_src1 = tl.sum(
                tl.where(used[:, :, None], residual1, 0.0), axis=1
            )
            for step in tl.static_range(1, C):
                inv_norm = tl.rsqrt(
                    tl.sum(basis_src0 * basis_src0, axis=1)
                    + tl.sum(basis_src1 * basis_src1, axis=1)
                    + 1e-12
                )
                basis0 = basis_src0 * inv_norm[:, None]
                basis1 = basis_src1 * inv_norm[:, None]
                proj = (
                    tl.sum(residual0 * basis0[:, None, :], axis=2)
                    + tl.sum(residual1 * basis1[:, None, :], axis=2)
                )
                residual0 = residual0 - proj[:, :, None] * basis0[:, None, :]
                residual1 = residual1 - proj[:, :, None] * basis1[:, None, :]
                obj = (
                    tl.sum(residual0 * residual0, axis=2)
                    + tl.sum(residual1 * residual1, axis=2)
                ) * weight
                obj = tl.where(valid2 & ~used, obj, -1.0)
                _, pick = tl.max(
                    obj, axis=1, return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit = cand_ids == pick[:, None]
                used = used | hit
                seeds = seeds + (start + pick)[:, None] * (slot == step).to(tl.int32)
                basis_src0 = tl.sum(
                    tl.where(hit[:, :, None], residual0, 0.0), axis=1
                )
                basis_src1 = tl.sum(
                    tl.where(hit[:, :, None], residual1, 0.0), axis=1
                )

            out_off = (
                (((((batch * K_OUTER + key_outer) * ROWS + key_row)
                    * COLS + out_cols[:, None]) * HEADS + head) * C)
                + slot
            )
            tl.store(
                seed_row_ptr + out_off,
                seeds.to(tl.int32),
                mask=out_mask,
            )


def span_seed_rows_triton(q, k, v, scale=None, *, alpha=0.3, candidate_count=30,
                          top_probes=8, uniform_probes=8, value_dims=32, block_j=None):
    """Seed table [B, A, F, K, J, H, 4] from the Triton selector."""
    if triton is None:
        raise RuntimeError("Triton is required for span_seed_rows_triton.")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, A, ROWS, COLS, H, D = q.shape
    F = k.size(1)
    if scale is None:
        scale = D ** -0.5
    table = torch.empty((B, A, F, ROWS, COLS, H, C4), device=q.device, dtype=torch.int32)
    block_rows = triton.next_power_of_2(candidate_count)
    block_l = triton.next_power_of_2(COLS)
    # Hold the candidate tile near 256 rows. BLOCK_ROWS is next_pow2 of the
    # candidate count, so a 45-row pool jumps to 64 and a fixed BLOCK_J=8 then
    # spills: 42 spills and 140 KiB of shared memory, 3.1x slower than the same
    # pool at BLOCK_J=4. Measured on A100 at 720p:
    #   pool 45, J=8  ->  42 spills, 140 KiB, 329.2 ms
    #   pool 45, J=4  ->   0 spills,  74 KiB, 106.2 ms
    #   pool 45, J=2  ->   0 spills,  41 KiB, 172.2 ms  (too little work per program)
    if block_j is None:
        # BLOCK_J is both a tile knob and the probe-sharing width, so it moves
        # speed and quality together. Measured at 480p (block_rows=32, 12
        # heads, isolated selector) and on the 16-point capture:
        #   J=2   168 regs   148.7 ms   rel-L2 0.06669  (-4.8% vs J=8)
        #   J=4   168 regs    99.6 ms   rel-L2 0.06789  (-3.1% vs J=8)  <- best
        #   J=8   255 regs   100.8 ms   rel-L2 0.07005
        #   J=16  spills      224.0 ms
        # Narrower sharing improves seed choice monotonically; J=4 also frees
        # registers (255 -> 168) though occupancy and per-program ILP roughly
        # cancel on speed. At 720p pool45 (block_rows=64) J=4 was also best
        # (J=2 172 ms vs J=4 106 ms), so cap at 4 rather than scale by rows.
        block_j = max(1, min(4, 256 // block_rows))
    grid = (A * ROWS * triton.cdiv(COLS, block_j), B * H)
    _span_seed_kernel[grid](
        q, k, v, table,
        A, F, ROWS, COLS, H, D,
        CANDIDATES=candidate_count, TOP_PROBES=top_probes,
        UNIFORM_PROBES=uniform_probes, VDIMS=value_dims,
        BLOCK_J=block_j, BLOCK_ROWS=block_rows, BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(D), C=C4,
        ALPHA=float(alpha), SCALE=float(scale), INPUT_PRECISION="ieee",
        # num_warps swept 2/4/8, num_stages swept 1-4 at the production tile
        # (BLOCK_J=8, BLOCK_ROWS=32). num_warps=2 spills 52-54 registers and is
        # 2.2x slower despite nominally allowing more resident CTAs -- the
        # kernel is register-bound by tile size, not warp count, so halving
        # warps just removes the ILP that was hiding the spill cost.
        # num_warps=8 is register-starved the other way (fewer per warp).
        # num_stages=1 beats 2 by 2.8% with identical regs/spills (0) at the
        # winning num_warps=4; higher stages cost more shared memory for no
        # gain since the K_OUTER loop body has little to pipeline into.
        num_warps=4, num_stages=1)
    return table


# ---------------------------------------------------------------------------
# Split-probe selector: probe selection and Gram-Schmidt as separate kernels.
#
# The fused _span_seed_kernel repeats its probe-selection front half (key tile
# load, anchor scoring, 8-round top-k) once per column group, which is what
# makes small BLOCK_J expensive: at J=2 that work runs 26x per (frame, key row)
# instead of 13x, and J=2 is the quality-best sharing width (rel-L2 0.06669 vs
# 0.07005 at J=8 on the 16-point capture).
#
# The split removes exactly that duplication and restructures it:
#   kernel 1  one program covers 16 column groups of a (frame, key row): the
#             key tile loads ONCE, anchor scoring for all groups is a single
#             tensor-core matmul ([groups*4, D] x [D, L]), and the 8 top-k
#             rounds run vectorized across the groups in lockstep instead of
#             serially per group. Emits 16 int32 probe columns per group.
#   kernel 2  the fused kernel minus keys/anchors/top-k: candidate scoring,
#             value profiles, importance weight, Gram-Schmidt. Much smaller
#             live state, which is what makes BLOCK_J=2 affordable.
#
# Numerics: kernel 1 scores anchors with tl.dot where the fused kernel used
# elementwise reductions; different summation order can flip near-tied top-k
# picks, so the split path is validated on the capture eval, not by
# bit-comparison against the fused kernel.
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _span_probe_kernel(
        q_ptr, k_ptr, probe_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        TOP_PROBES: tl.constexpr, UNIFORM_PROBES: tl.constexpr,
        PROBE_PAD: tl.constexpr,
        BLOCK_J: tl.constexpr, N_GROUPS: tl.constexpr, GROUP_TILE: tl.constexpr,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        m_tiles: tl.constexpr = (N_GROUPS + GROUP_TILE - 1) // GROUP_TILE
        m_tile = pid % m_tiles
        pid = pid // m_tiles
        key_row = pid % ROWS
        pid = pid // ROWS
        key_outer = pid % K_OUTER
        query_outer = pid // K_OUTER

        dims = tl.arange(0, BLOCK_D)
        ls = tl.arange(0, BLOCK_L)
        key_valid = ls < COLS
        log2e: tl.constexpr = 1.4426950408889634

        k_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS * HEADS * D_HEAD
        keys = tl.load(
            k_ptr + k_base + (ls[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=key_valid[:, None], other=0.0)

        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
        q_stride = COLS * HEADS * D_HEAD

        # 16 groups x 4 anchors stacked into one [64, D] tile. Anchor rows
        # depend only on key_row; the column differs per group (col0).
        rows_m = tl.arange(0, GROUP_TILE * 4)
        g_local = rows_m // 4
        a_i = rows_m % 4
        g_global_rows = m_tile * GROUP_TILE + g_local
        group_ok_rows = g_global_rows < N_GROUPS
        col0 = g_global_rows * BLOCK_J
        off_a = tl.where(a_i == 0, -2, tl.where(a_i == 1, 0, tl.where(a_i == 2, 1, 3)))
        arow = tl.minimum(tl.maximum(key_row + off_a, 0), ROWS - 1)
        anchors = tl.load(
            q_ptr + q_base + arow[:, None] * q_stride
            + (col0[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=group_ok_rows[:, None], other=0.0)

        resp = tl.dot(anchors, tl.trans(keys), input_precision=INPUT_PRECISION)
        resp = resp * (SCALE * log2e)
        resp = tl.where(key_valid[None, :], resp, -float("inf"))
        rmax = tl.max(resp, axis=1)
        lse = rmax + tl.log2(tl.sum(tl.exp2(resp - rmax[:, None]), axis=1))
        norm = tl.reshape(resp - lse[:, None], (GROUP_TILE, 4, BLOCK_L))
        importance = tl.max(norm, axis=1)                       # [GROUP_TILE, L]

        ustride: tl.constexpr = COLS // UNIFORM_PROBES
        uniform_hit = (ls % ustride == 0) & key_valid
        work = tl.where(
            (key_valid & ~uniform_hit)[None, :], importance, -float("inf"))
        P: tl.constexpr = TOP_PROBES + UNIFORM_PROBES
        pslots = tl.arange(0, PROBE_PAD)
        probe_base = tl.where(
            (pslots >= TOP_PROBES) & (pslots < P),
            (pslots - TOP_PROBES) * ustride,
            0,
        )
        probe_cols = (
            tl.zeros((GROUP_TILE, PROBE_PAD), tl.int32) + probe_base[None, :]
        )
        # 8 top-k rounds for all 16 groups advance in lockstep: one argmax per
        # round instead of one per (group, round).
        for r in tl.static_range(0, TOP_PROBES):
            _, best = tl.max(work, axis=1, return_indices=True,
                             return_indices_tie_break_left=True)
            probe_cols = tl.where(pslots[None, :] == r, best[:, None], probe_cols)
            work = tl.where(ls[None, :] == best[:, None], -float("inf"), work)

        g_vec = m_tile * GROUP_TILE + tl.arange(0, GROUP_TILE)
        out_off = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                      * ROWS + key_row) * N_GROUPS + g_vec[:, None]) * HEADS + head)
                   * PROBE_PAD + pslots[None, :])
        tl.store(probe_ptr + out_off, probe_cols,
                 mask=(g_vec < N_GROUPS)[:, None])

    @triton.jit
    def _span_gs_kernel(
        q_ptr, k_ptr, v_ptr, probe_ptr, seed_row_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr, TOP_PROBES: tl.constexpr,
        UNIFORM_PROBES: tl.constexpr, VDIMS: tl.constexpr,
        PROBE_PAD: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_ROWS: tl.constexpr, N_GROUPS: tl.constexpr,
        KEY_OUTER_TILE: tl.constexpr, N_KEY_OUTER_TILES: tl.constexpr,
        BLOCK_D: tl.constexpr, C: tl.constexpr,
        ALPHA: tl.constexpr, SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """The fused kernel's scoring + Gram-Schmidt half, probes precomputed."""
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        j_block = pid % N_GROUPS
        key_outer_tile = (pid // N_GROUPS) % N_KEY_OUTER_TILES
        key_row = (pid // (N_GROUPS * N_KEY_OUTER_TILES)) % ROWS
        query_outer = (pid // (N_GROUPS * N_KEY_OUTER_TILES)) // ROWS

        dims = tl.arange(0, BLOCK_D)
        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
        q_stride = COLS * HEADS * D_HEAD
        log2e: tl.constexpr = 1.4426950408889634
        P: tl.constexpr = TOP_PROBES + UNIFORM_PROBES

        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        j_slot = flat // BLOCK_ROWS
        c_slot = flat % BLOCK_ROWS
        cols = j_block * BLOCK_J + j_slot
        start = tl.minimum(tl.maximum(key_row - CANDIDATES // 2, 0), ROWS - CANDIDATES)
        cand_rows = start + c_slot
        cand_valid = (cols < COLS) & (c_slot < CANDIDATES)
        cand = tl.load(
            q_ptr + q_base + cand_rows[:, None] * q_stride
            + (cols[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=cand_valid[:, None], other=0.0)

        cand_rows_2d = tl.reshape(cand_rows, (BLOCK_J, BLOCK_ROWS))
        valid2 = tl.reshape(cand_valid, (BLOCK_J, BLOCK_ROWS))
        n_valid = tl.sum(valid2.to(tl.float32), axis=1)
        inv_n_valid = 1.0 / n_valid
        cand_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        center_index = key_row - start
        slot = tl.arange(0, C)[None, :]
        pslots = tl.arange(0, PROBE_PAD)
        out_cols = j_block * BLOCK_J + tl.arange(0, BLOCK_J)
        out_mask = (out_cols < COLS)[:, None]

        for key_outer_slot in tl.static_range(0, KEY_OUTER_TILE):
            key_outer = key_outer_tile * KEY_OUTER_TILE + key_outer_slot
            k_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS * HEADS * D_HEAD
            poff = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                       * ROWS + key_row) * N_GROUPS + j_block) * HEADS + head)
                    * PROBE_PAD + pslots)
            probe_cols = tl.load(probe_ptr + poff)

            probe_mask = (pslots[:, None] < P) & (probe_cols[:, None] < COLS)
            probe_off = k_base + (probe_cols[:, None] * HEADS + head) * D_HEAD + dims[None, :]
            probe_keys = tl.load(k_ptr + probe_off, mask=probe_mask, other=0.0)
            logits = tl.dot(cand, tl.trans(probe_keys), input_precision=INPUT_PRECISION)
            logits = logits * (SCALE * log2e)
            logits = tl.where(pslots[None, :] < P, logits, -float("inf"))
            row_max = tl.max(logits, axis=1)
            expd = tl.exp2(logits - row_max[:, None])
            denom = tl.sum(expd, axis=1)
            log_mass = row_max + tl.log2(denom)
            prob = expd * (1.0 / denom)[:, None]

            vals = tl.load(v_ptr + probe_off, mask=probe_mask, other=0.0)
            vals = tl.reshape(vals, (PROBE_PAD, VDIMS, D_HEAD // VDIMS))
            vals = tl.sum(vals, axis=2) * (1.0 / (D_HEAD // VDIMS))
            profile = tl.dot(prob.to(vals.dtype), vals, input_precision=INPUT_PRECISION)
            profile = tl.reshape(profile, (BLOCK_J, BLOCK_ROWS, VDIMS)).to(tl.float32)
            residual = profile - tl.sum(profile, axis=2)[:, :, None] * (1.0 / VDIMS)

            lm = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS))
            mu = tl.sum(tl.where(valid2, lm, 0.0), axis=1) * inv_n_valid
            dev = lm - mu[:, None]
            var = tl.sum(tl.where(valid2, dev * dev, 0.0), axis=1) * inv_n_valid
            z = dev * tl.rsqrt(var[:, None] + 1e-6)
            weight = tl.exp2((2.0 * ALPHA * log2e) * z)

            used = cand_ids == center_index
            seed0 = tl.sum(tl.where(used, cand_rows_2d, 0), axis=1)
            seeds = seed0[:, None] * (slot == 0).to(tl.int32)
            basis_src = tl.sum(tl.where(used[:, :, None], residual, 0.0), axis=1)
            for step in tl.static_range(1, C):
                basis = basis_src * tl.rsqrt(
                    tl.sum(basis_src * basis_src, axis=1) + 1e-12)[:, None]
                proj = tl.sum(residual * basis[:, None, :], axis=2)
                residual = residual - proj[:, :, None] * basis[:, None, :]
                obj = tl.sum(residual * residual, axis=2) * weight
                obj = tl.where(valid2 & ~used, obj, -1.0)
                _, pick = tl.max(obj, axis=1, return_indices=True,
                                 return_indices_tie_break_left=True)
                hit = cand_ids == pick[:, None]
                used = used | hit
                seeds = seeds + (start + pick)[:, None] * (slot == step).to(tl.int32)
                basis_src = tl.sum(tl.where(hit[:, :, None], residual, 0.0), axis=1)

            out_off = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                          * ROWS + key_row) * COLS + out_cols[:, None]) * HEADS + head)
                       * C + slot)
            tl.store(seed_row_ptr + out_off, seeds.to(tl.int32), mask=out_mask)

    @triton.jit
    def _span_gs_temporal_probe_shared_kernel(
        q_ptr, k_ptr, v_ptr, probe_ptr, seed_row_ptr,
        Q_OUTER: tl.constexpr, ROUTER_FRAMES: tl.constexpr,
        K_OUTER: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr, TOP_PROBES: tl.constexpr,
        UNIFORM_PROBES: tl.constexpr, VDIMS: tl.constexpr,
        PROBE_PAD: tl.constexpr, BLOCK_J: tl.constexpr,
        BLOCK_ROWS: tl.constexpr, N_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr, C: tl.constexpr,
        ALPHA: tl.constexpr, SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        """Run per-frame GS while sharing only probe columns over time."""

        tl.static_assert(Q_OUTER % ROUTER_FRAMES == 0)
        frames_per_router: tl.constexpr = Q_OUTER // ROUTER_FRAMES
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        j_block = pid % N_GROUPS
        key_row = (pid // N_GROUPS) % ROWS
        query_outer_program = (pid // N_GROUPS) // ROWS
        router_slot = query_outer_program // frames_per_router

        dims = tl.arange(0, BLOCK_D)
        q_stride = COLS * HEADS * D_HEAD
        log2e: tl.constexpr = 1.4426950408889634
        P: tl.constexpr = TOP_PROBES + UNIFORM_PROBES
        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        j_slot = flat // BLOCK_ROWS
        c_slot = flat % BLOCK_ROWS
        cols = j_block * BLOCK_J + j_slot
        start = tl.minimum(
            tl.maximum(key_row - CANDIDATES // 2, 0), ROWS - CANDIDATES
        )
        cand_rows = start + c_slot
        cand_valid = (cols < COLS) & (c_slot < CANDIDATES)
        cand_rows_2d = tl.reshape(cand_rows, (BLOCK_J, BLOCK_ROWS))
        valid2 = tl.reshape(cand_valid, (BLOCK_J, BLOCK_ROWS))
        n_valid = tl.sum(valid2.to(tl.float32), axis=1)
        inv_n_valid = 1.0 / n_valid
        cand_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        center_index = key_row - start
        seed_slot = tl.arange(0, C)[None, :]
        probe_slot = tl.arange(0, PROBE_PAD)
        out_cols = j_block * BLOCK_J + tl.arange(0, BLOCK_J)
        out_mask = (out_cols < COLS)[:, None]

        # One target frame per program exposes all 21 frames to the grid. The
        # previous seven-frame loop serialized independent GS problems and
        # left the H200 under-occupied.
        for frame_unroll in tl.static_range(0, 1):
            query_outer = query_outer_program
            q_base = (
                (batch * Q_OUTER + query_outer)
                * ROWS
                * COLS
                * HEADS
                * D_HEAD
            )
            candidate = tl.load(
                q_ptr
                + q_base
                + cand_rows[:, None] * q_stride
                + (cols[:, None] * HEADS + head) * D_HEAD
                + dims[None, :],
                mask=cand_valid[:, None],
                other=0.0,
            )

            for key_outer in tl.range(0, K_OUTER):
                k_base = (
                    ((batch * K_OUTER + key_outer) * ROWS + key_row)
                    * COLS
                    * HEADS
                    * D_HEAD
                )
                probe_offset = (
                    ((((((batch * ROUTER_FRAMES + router_slot) * K_OUTER
                         + key_outer) * ROWS + key_row) * N_GROUPS + j_block)
                       * HEADS + head) * PROBE_PAD)
                    + probe_slot
                )
                probe_cols = tl.load(probe_ptr + probe_offset)
                probe_mask = (
                    (probe_slot[:, None] < P)
                    & (probe_cols[:, None] < COLS)
                )
                probe_kv_offset = (
                    k_base
                    + (probe_cols[:, None] * HEADS + head) * D_HEAD
                    + dims[None, :]
                )
                probe_keys = tl.load(
                    k_ptr + probe_kv_offset, mask=probe_mask, other=0.0
                )
                logits = tl.dot(
                    candidate,
                    tl.trans(probe_keys),
                    input_precision=INPUT_PRECISION,
                )
                logits = logits * (SCALE * log2e)
                logits = tl.where(
                    probe_slot[None, :] < P, logits, -float("inf")
                )
                row_max = tl.max(logits, axis=1)
                exponential = tl.exp2(logits - row_max[:, None])
                denominator = tl.sum(exponential, axis=1)
                log_mass = row_max + tl.log2(denominator)
                probability = exponential * (1.0 / denominator)[:, None]

                probe_values = tl.load(
                    v_ptr + probe_kv_offset, mask=probe_mask, other=0.0
                )
                probe_values = tl.reshape(
                    probe_values, (PROBE_PAD, VDIMS, D_HEAD // VDIMS)
                )
                probe_values = tl.sum(probe_values, axis=2) * (
                    1.0 / (D_HEAD // VDIMS)
                )
                profile = tl.dot(
                    probability.to(probe_values.dtype),
                    probe_values,
                    input_precision=INPUT_PRECISION,
                )
                profile = tl.reshape(
                    profile, (BLOCK_J, BLOCK_ROWS, VDIMS)
                ).to(tl.float32)
                residual = profile - tl.sum(
                    profile, axis=2
                )[:, :, None] * (1.0 / VDIMS)

                mass = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS))
                mean = tl.sum(tl.where(valid2, mass, 0.0), axis=1) * inv_n_valid
                deviation = mass - mean[:, None]
                variance = tl.sum(
                    tl.where(valid2, deviation * deviation, 0.0), axis=1
                ) * inv_n_valid
                z_score = deviation * tl.rsqrt(variance[:, None] + 1e-6)
                weight = tl.exp2((2.0 * ALPHA * log2e) * z_score)

                used = cand_ids == center_index
                seed0 = tl.sum(
                    tl.where(used, cand_rows_2d, 0), axis=1
                )
                seeds = seed0[:, None] * (seed_slot == 0).to(tl.int32)
                basis_source = tl.sum(
                    tl.where(used[:, :, None], residual, 0.0), axis=1
                )
                for step in tl.static_range(1, C):
                    basis = basis_source * tl.rsqrt(
                        tl.sum(basis_source * basis_source, axis=1) + 1e-12
                    )[:, None]
                    projection = tl.sum(
                        residual * basis[:, None, :], axis=2
                    )
                    residual = (
                        residual
                        - projection[:, :, None] * basis[:, None, :]
                    )
                    objective = tl.sum(residual * residual, axis=2) * weight
                    objective = tl.where(valid2 & ~used, objective, -1.0)
                    _, pick = tl.max(
                        objective,
                        axis=1,
                        return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    hit = cand_ids == pick[:, None]
                    used = used | hit
                    seeds = seeds + (start + pick)[:, None] * (
                        seed_slot == step
                    ).to(tl.int32)
                    basis_source = tl.sum(
                        tl.where(hit[:, :, None], residual, 0.0), axis=1
                    )

                output_offset = (
                    ((((((batch * Q_OUTER + query_outer) * K_OUTER
                         + key_outer) * ROWS + key_row) * COLS
                       + out_cols[:, None]) * HEADS + head) * C)
                    + seed_slot
                )
                tl.store(
                    seed_row_ptr + output_offset,
                    seeds.to(tl.int32),
                    mask=out_mask,
                )

    @triton.jit
    def _temporal_query_row_map_kernel(
        q_ptr,
        row_map_ptr,
        Q_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        ROUTER_FRAME: tl.constexpr,
        RADIUS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        """Track router-frame rows with a banded cosine-similarity map."""

        frame_column = tl.program_id(0)
        batch_head = tl.program_id(1)
        query_outer = frame_column // COLS
        query_column = frame_column % COLS
        head = batch_head % HEADS
        batch = batch_head // HEADS

        rows = tl.arange(0, BLOCK_ROWS)
        dims = tl.arange(0, BLOCK_D)
        row_valid = rows < ROWS
        dim_valid = dims < D_HEAD
        router_base = (
            (((batch * Q_OUTER + ROUTER_FRAME) * ROWS + rows[:, None]) * COLS
              + query_column)
            * HEADS
            + head
        ) * D_HEAD
        target_base = (
            (((batch * Q_OUTER + query_outer) * ROWS + rows[:, None]) * COLS
              + query_column)
            * HEADS
            + head
        ) * D_HEAD
        router_query = tl.load(
            q_ptr + router_base + dims[None, :],
            mask=row_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        target_query = tl.load(
            q_ptr + target_base + dims[None, :],
            mask=row_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        router_float = router_query.to(tl.float32)
        target_float = target_query.to(tl.float32)
        router_inv_norm = tl.rsqrt(
            tl.sum(router_float * router_float, axis=1) + 1.0e-6
        )
        target_inv_norm = tl.rsqrt(
            tl.sum(target_float * target_float, axis=1) + 1.0e-6
        )
        router_unit = (router_float * router_inv_norm[:, None]).to(
            router_query.dtype
        )
        target_unit = (target_float * target_inv_norm[:, None]).to(
            target_query.dtype
        )
        similarity = tl.dot(
            router_unit,
            tl.trans(target_unit),
            input_precision=INPUT_PRECISION,
        )
        target_rows = tl.arange(0, BLOCK_ROWS)[None, :]
        band = tl.abs(rows[:, None] - target_rows) <= RADIUS
        valid = row_valid[:, None] & (target_rows < ROWS) & band
        similarity = tl.where(valid, similarity, -float("inf"))
        _, best_row = tl.max(
            similarity,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        best_row = tl.where(query_outer == ROUTER_FRAME, rows, best_row)
        output_base = (
            (((batch * Q_OUTER + query_outer) * COLS + query_column) * HEADS
              + head)
            * ROWS
        )
        tl.store(
            row_map_ptr + output_base + rows,
            best_row.to(tl.uint8),
            mask=row_valid,
        )

    @triton.jit
    def _hellinger_probe_kernel(
        q_ptr, k_ptr, probe_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        TOP_PROBES: tl.constexpr, UNIFORM_PROBES: tl.constexpr,
        BLOCK_J: tl.constexpr, N_GROUPS: tl.constexpr, GROUP_TILE: tl.constexpr,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """Select probes with the two-anchor (-1, +1) screened configuration."""
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        m_tiles: tl.constexpr = (N_GROUPS + GROUP_TILE - 1) // GROUP_TILE
        m_tile = pid % m_tiles
        pid = pid // m_tiles
        key_row = pid % ROWS
        pid = pid // ROWS
        key_outer = pid % K_OUTER
        query_outer = pid // K_OUTER

        dims = tl.arange(0, BLOCK_D)
        ls = tl.arange(0, BLOCK_L)
        key_valid = ls < COLS
        log2e: tl.constexpr = 1.4426950408889634

        k_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS * HEADS * D_HEAD
        keys = tl.load(
            k_ptr + k_base + (ls[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=key_valid[:, None], other=0.0)

        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
        q_stride = COLS * HEADS * D_HEAD
        rows_m = tl.arange(0, GROUP_TILE * 2)
        g_local = rows_m // 2
        anchor_id = rows_m % 2
        g_global_rows = m_tile * GROUP_TILE + g_local
        group_ok_rows = g_global_rows < N_GROUPS
        col0 = g_global_rows * BLOCK_J
        anchor_row = tl.minimum(
            tl.maximum(key_row + tl.where(anchor_id == 0, -1, 1), 0), ROWS - 1
        )
        anchors = tl.load(
            q_ptr + q_base + anchor_row[:, None] * q_stride
            + (col0[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=group_ok_rows[:, None], other=0.0)

        response = tl.dot(anchors, tl.trans(keys), input_precision=INPUT_PRECISION)
        response = response * (SCALE * log2e)
        response = tl.where(key_valid[None, :], response, -float("inf"))
        response_max = tl.max(response, axis=1)
        log_norm = response_max + tl.log2(
            tl.sum(tl.exp2(response - response_max[:, None]), axis=1)
        )
        normalized = tl.reshape(
            response - log_norm[:, None], (GROUP_TILE, 2, BLOCK_L)
        )
        importance = tl.max(normalized, axis=1)

        uniform_stride: tl.constexpr = COLS // UNIFORM_PROBES
        uniform_hit = (ls % uniform_stride == 0) & key_valid
        work = tl.where(
            (key_valid & ~uniform_hit)[None, :], importance, -float("inf")
        )
        P: tl.constexpr = TOP_PROBES + UNIFORM_PROBES
        probe_slot = tl.arange(0, P)
        uniform_probe = (probe_slot - TOP_PROBES) * uniform_stride
        probe_columns = tl.where(
            probe_slot[None, :] >= TOP_PROBES, uniform_probe[None, :], 0
        )
        for rank in tl.static_range(0, TOP_PROBES):
            _, best = tl.max(
                work,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            probe_columns = tl.where(
                probe_slot[None, :] == rank, best[:, None], probe_columns
            )
            work = tl.where(ls[None, :] == best[:, None], -float("inf"), work)

        group = m_tile * GROUP_TILE + tl.arange(0, GROUP_TILE)
        output_offset = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                            * ROWS + key_row) * N_GROUPS + group[:, None])
                          * HEADS + head) * P + probe_slot[None, :])
        tl.store(
            probe_ptr + output_offset,
            probe_columns,
            mask=(group < N_GROUPS)[:, None],
        )

    @triton.jit
    def _hellinger_gs_kernel(
        q_ptr, k_ptr, probe_ptr, candidate_ptr, anchor_ptr, seed_row_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr, TOP_PROBES: tl.constexpr,
        UNIFORM_PROBES: tl.constexpr, P_PAD: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_ROWS: tl.constexpr, N_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr, C: tl.constexpr, C_PAD: tl.constexpr,
        SELECTION_MODE: tl.constexpr,
        FORCE_LOCAL_M1: tl.constexpr,
        FULL_KEYS: tl.constexpr,
        ALPHA: tl.constexpr, SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """Select C rows by weighted Gram-Schmidt in Hellinger space."""
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        group = pid % N_GROUPS
        key_row = (pid // N_GROUPS) % ROWS
        query_outer = (pid // N_GROUPS) // ROWS

        dims = tl.arange(0, BLOCK_D)
        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
        q_stride = COLS * HEADS * D_HEAD
        log2e: tl.constexpr = 1.4426950408889634
        P: tl.constexpr = TOP_PROBES + UNIFORM_PROBES

        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        column_slot = flat // BLOCK_ROWS
        candidate_slot = flat % BLOCK_ROWS
        columns = group * BLOCK_J + column_slot
        candidate_rows = tl.load(
            candidate_ptr + key_row * CANDIDATES + candidate_slot,
            mask=candidate_slot < CANDIDATES,
            other=0,
        )
        candidate_valid = (columns < COLS) & (candidate_slot < CANDIDATES)
        candidate = tl.load(
            q_ptr + q_base + candidate_rows[:, None] * q_stride
            + (columns[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=candidate_valid[:, None], other=0.0)

        candidate_rows_2d = tl.reshape(candidate_rows, (BLOCK_J, BLOCK_ROWS))
        valid_2d = tl.reshape(candidate_valid, (BLOCK_J, BLOCK_ROWS))
        valid_count = tl.sum(valid_2d.to(tl.float32), axis=1)
        candidate_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        anchor_index = tl.load(anchor_ptr + key_row)
        seed_slot = tl.arange(0, C_PAD)[None, :]
        probe_slot = tl.arange(0, P_PAD)
        probe_valid = probe_slot < P
        output_columns = group * BLOCK_J + tl.arange(0, BLOCK_J)
        output_mask = (output_columns < COLS)[:, None]

        for key_outer in tl.range(0, K_OUTER):
            key_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS * HEADS * D_HEAD
            if FULL_KEYS:
                probe_columns = probe_slot
            else:
                probe_offset = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                                   * ROWS + key_row) * N_GROUPS + group)
                                 * HEADS + head) * P + probe_slot)
                probe_columns = tl.load(
                    probe_ptr + probe_offset,
                    mask=probe_valid,
                    other=COLS,
                )
            probe_mask = probe_valid[:, None] & (probe_columns[:, None] < COLS)
            probe_key_offset = (
                key_base
                + (probe_columns[:, None] * HEADS + head) * D_HEAD
                + dims[None, :]
            )
            probe_keys = tl.load(
                k_ptr + probe_key_offset, mask=probe_mask, other=0.0
            )
            logits = tl.dot(
                candidate, tl.trans(probe_keys), input_precision=INPUT_PRECISION
            )
            logits = logits * (SCALE * log2e)
            logits = tl.where(probe_valid[None, :], logits, -float("inf"))
            row_max = tl.max(logits, axis=1)
            exponential = tl.exp2(logits - row_max[:, None])
            denominator = tl.sum(exponential, axis=1)
            log_mass = row_max + tl.log2(denominator)
            probability = exponential * (1.0 / denominator)[:, None]

            profile = tl.sqrt(probability)
            profile = tl.reshape(profile, (BLOCK_J, BLOCK_ROWS, P_PAD)).to(tl.float32)
            residual = tl.where(
                probe_valid[None, None, :],
                profile - tl.sum(profile, axis=2)[:, :, None] * (1.0 / P),
                0.0,
            )

            mass = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS))
            mean = tl.sum(tl.where(valid_2d, mass, 0.0), axis=1) / valid_count
            deviation = mass - mean[:, None]
            variance = (
                tl.sum(tl.where(valid_2d, deviation * deviation, 0.0), axis=1)
                / valid_count
            )
            z_score = deviation * tl.rsqrt(variance[:, None] + 1e-6)
            weight = tl.exp2((2.0 * ALPHA * log2e) * z_score)

            used = candidate_ids == anchor_index
            seed0 = tl.sum(
                tl.where(used, candidate_rows_2d, 0), axis=1
            )
            seeds = seed0[:, None] * (seed_slot == 0).to(tl.int32)
            basis_source = tl.sum(
                tl.where(used[:, :, None], residual, 0.0), axis=1
            )

            if SELECTION_MODE == 4 or SELECTION_MODE == 5:
                # Pair-first volume selection.  Project the fixed anchor once,
                # shortlist eight candidates, score all shortlist pairs in
                # parallel by their weighted Gram area, then score every
                # remaining candidate by the anchored 3x3 Gram determinant.
                # There is no iterative vector orthogonalization.
                tl.static_assert(C == 4)
                shortlist_count: tl.constexpr = (
                    4 if SELECTION_MODE == 5 else 8
                )
                shortlist_slot = tl.arange(0, shortlist_count)
                basis = basis_source * tl.rsqrt(
                    tl.sum(basis_source * basis_source, axis=1) + 1e-12
                )[:, None]
                projection = tl.sum(residual * basis[:, None, :], axis=2)
                residual = residual - projection[:, :, None] * basis[:, None, :]
                diagonal = tl.sum(residual * residual, axis=2)
                shortlist_objective = tl.where(
                    valid_2d & ~used,
                    diagonal * weight,
                    -1.0,
                )
                shortlist = tl.zeros(
                    (BLOCK_J, shortlist_count),
                    tl.int32,
                )
                shortlist_profile = tl.zeros(
                    (BLOCK_J, shortlist_count, P_PAD),
                    tl.float32,
                )
                shortlist_weight = tl.zeros(
                    (BLOCK_J, shortlist_count),
                    tl.float32,
                )
                for rank in tl.static_range(0, shortlist_count):
                    _, pick = tl.max(
                        shortlist_objective,
                        axis=1,
                        return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    hit = candidate_ids == pick[:, None]
                    selected_profile = tl.sum(
                        tl.where(hit[:, :, None], residual, 0.0),
                        axis=1,
                    )
                    selected_weight = tl.sum(
                        tl.where(hit, weight, 0.0),
                        axis=1,
                    )
                    rank_hit = shortlist_slot == rank
                    shortlist = shortlist + pick[:, None] * rank_hit[None, :].to(
                        tl.int32
                    )
                    shortlist_profile = shortlist_profile + (
                        selected_profile[:, None, :]
                        * rank_hit[None, :, None].to(tl.float32)
                    )
                    shortlist_weight = shortlist_weight + (
                        selected_weight[:, None]
                        * rank_hit[None, :].to(tl.float32)
                    )
                    shortlist_objective = tl.where(
                        hit,
                        -1.0,
                        shortlist_objective,
                    )

                pair_gram = tl.sum(
                    shortlist_profile[:, :, None, :]
                    * shortlist_profile[:, None, :, :],
                    axis=3,
                )
                shortlist_diagonal = tl.sum(
                    shortlist_profile * shortlist_profile,
                    axis=2,
                )
                pair_area = tl.maximum(
                    shortlist_diagonal[:, :, None]
                    * shortlist_diagonal[:, None, :]
                    - pair_gram * pair_gram,
                    0.0,
                )
                pair_score = (
                    pair_area
                    * shortlist_weight[:, :, None]
                    * shortlist_weight[:, None, :]
                )
                pair_valid = shortlist_slot[:, None] < shortlist_slot[None, :]
                pair_score = tl.where(pair_valid[None, :, :], pair_score, -1.0)
                pair_score_flat = tl.reshape(
                    pair_score,
                    (BLOCK_J, shortlist_count * shortlist_count),
                )
                _, pair_pick = tl.max(
                    pair_score_flat,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                first_slot = pair_pick // shortlist_count
                second_slot = pair_pick % shortlist_count
                first_hit = shortlist_slot == first_slot[:, None]
                second_hit = shortlist_slot == second_slot[:, None]
                first = tl.sum(tl.where(first_hit, shortlist, 0), axis=1)
                second = tl.sum(tl.where(second_hit, shortlist, 0), axis=1)
                first_profile = tl.sum(
                    tl.where(first_hit[:, :, None], shortlist_profile, 0.0),
                    axis=1,
                )
                second_profile = tl.sum(
                    tl.where(second_hit[:, :, None], shortlist_profile, 0.0),
                    axis=1,
                )
                g11 = tl.sum(first_profile * first_profile, axis=1)
                g22 = tl.sum(second_profile * second_profile, axis=1)
                g12 = tl.sum(first_profile * second_profile, axis=1)
                g13 = tl.sum(
                    residual * first_profile[:, None, :],
                    axis=2,
                )
                g23 = tl.sum(
                    residual * second_profile[:, None, :],
                    axis=2,
                )
                determinant = (
                    g11[:, None] * g22[:, None] * diagonal
                    + 2.0 * g12[:, None] * g13 * g23
                    - g11[:, None] * g23 * g23
                    - g22[:, None] * g13 * g13
                    - diagonal * g12[:, None] * g12[:, None]
                )
                first_candidate_hit = candidate_ids == first[:, None]
                second_candidate_hit = candidate_ids == second[:, None]
                used = used | first_candidate_hit | second_candidate_hit
                third_objective = tl.where(
                    valid_2d & ~used,
                    tl.maximum(determinant, 0.0) * weight,
                    -1.0,
                )
                _, third = tl.max(
                    third_objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                third_hit = candidate_ids == third[:, None]
                first_row = tl.sum(
                    tl.where(first_candidate_hit, candidate_rows_2d, 0),
                    axis=1,
                )
                second_row = tl.sum(
                    tl.where(second_candidate_hit, candidate_rows_2d, 0),
                    axis=1,
                )
                third_row = tl.sum(
                    tl.where(third_hit, candidate_rows_2d, 0),
                    axis=1,
                )
                seeds = (
                    seeds
                    + first_row[:, None] * (seed_slot == 1).to(tl.int32)
                    + second_row[:, None] * (seed_slot == 2).to(tl.int32)
                    + third_row[:, None] * (seed_slot == 3).to(tl.int32)
                )
            elif SELECTION_MODE == 3:
                # Pivoted Cholesky on the Hellinger Gram matrix.  This is
                # algebraically equivalent to weighted pivoted GS, but it
                # updates a clipped Schur-complement diagonal instead of
                # repeatedly mutating the feature vectors.  C4 is unrolled so
                # only the two prior Cholesky columns that are still needed
                # remain live; no [candidate, C] factor workspace is built.
                tl.static_assert(C == 4)
                diagonal = tl.sum(residual * residual, axis=2)
                gram0 = tl.sum(
                    residual * basis_source[:, None, :],
                    axis=2,
                )
                pivot0_diagonal = tl.sum(
                    tl.where(used, diagonal, 0.0),
                    axis=1,
                )
                column0 = gram0 * tl.rsqrt(
                    tl.maximum(pivot0_diagonal, 1e-12)
                )[:, None]
                diagonal = tl.maximum(diagonal - column0 * column0, 0.0)
                objective = tl.where(
                    valid_2d & ~used,
                    diagonal * weight,
                    -1.0,
                )
                _, pick1 = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit1 = candidate_ids == pick1[:, None]
                used = used | hit1
                row1 = tl.sum(
                    tl.where(hit1, candidate_rows_2d, 0),
                    axis=1,
                )
                seeds = seeds + row1[:, None] * (seed_slot == 1).to(tl.int32)

                profile1 = tl.sum(
                    tl.where(hit1[:, :, None], residual, 0.0),
                    axis=1,
                )
                gram1 = tl.sum(
                    residual * profile1[:, None, :],
                    axis=2,
                )
                column0_p1 = tl.sum(tl.where(hit1, column0, 0.0), axis=1)
                pivot1_diagonal = tl.sum(
                    tl.where(hit1, diagonal, 0.0),
                    axis=1,
                )
                column1 = (gram1 - column0 * column0_p1[:, None]) * tl.rsqrt(
                    tl.maximum(pivot1_diagonal, 1e-12)
                )[:, None]
                diagonal = tl.maximum(diagonal - column1 * column1, 0.0)
                objective = tl.where(
                    valid_2d & ~used,
                    diagonal * weight,
                    -1.0,
                )
                _, pick2 = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit2 = candidate_ids == pick2[:, None]
                used = used | hit2
                row2 = tl.sum(
                    tl.where(hit2, candidate_rows_2d, 0),
                    axis=1,
                )
                seeds = seeds + row2[:, None] * (seed_slot == 2).to(tl.int32)

                profile2 = tl.sum(
                    tl.where(hit2[:, :, None], residual, 0.0),
                    axis=1,
                )
                gram2 = tl.sum(
                    residual * profile2[:, None, :],
                    axis=2,
                )
                column0_p2 = tl.sum(tl.where(hit2, column0, 0.0), axis=1)
                column1_p2 = tl.sum(tl.where(hit2, column1, 0.0), axis=1)
                pivot2_diagonal = tl.sum(
                    tl.where(hit2, diagonal, 0.0),
                    axis=1,
                )
                column2 = (
                    gram2
                    - column0 * column0_p2[:, None]
                    - column1 * column1_p2[:, None]
                ) * tl.rsqrt(tl.maximum(pivot2_diagonal, 1e-12))[:, None]
                diagonal = tl.maximum(diagonal - column2 * column2, 0.0)
                objective = tl.where(
                    valid_2d & ~used,
                    diagonal * weight,
                    -1.0,
                )
                _, pick3 = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit3 = candidate_ids == pick3[:, None]
                row3 = tl.sum(
                    tl.where(hit3, candidate_rows_2d, 0),
                    axis=1,
                )
                seeds = seeds + row3[:, None] * (seed_slot == 3).to(tl.int32)
            elif SELECTION_MODE == 0:
                # Full weighted Gram-Schmidt: maximize the residual volume at
                # every step. This remains the default/reference selector.
                for step in tl.static_range(1, C):
                    basis = basis_source * tl.rsqrt(
                        tl.sum(basis_source * basis_source, axis=1) + 1e-12
                    )[:, None]
                    projection = tl.sum(residual * basis[:, None, :], axis=2)
                    residual = residual - projection[:, :, None] * basis[:, None, :]
                    objective = tl.sum(residual * residual, axis=2) * weight
                    objective = tl.where(valid_2d & ~used, objective, -1.0)
                    _, pick = tl.max(
                        objective,
                        axis=1,
                        return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    hit = candidate_ids == pick[:, None]
                    used = used | hit
                    picked_row = tl.sum(
                        tl.where(hit, candidate_rows_2d, 0), axis=1
                    )
                    seeds = seeds + picked_row[:, None] * (seed_slot == step).to(tl.int32)
                    basis_source = tl.sum(
                        tl.where(hit[:, :, None], residual, 0.0), axis=1
                    )
            elif SELECTION_MODE == 1:
                # Two-stage span: orthogonalize against anchor and one pivot,
                # then take the top two remaining rows in parallel. Offline
                # real-QKV screening retains 99% of full-GS output accuracy.
                tl.static_assert(C == 4)
                basis = basis_source * tl.rsqrt(
                    tl.sum(basis_source * basis_source, axis=1) + 1e-12
                )[:, None]
                projection = tl.sum(residual * basis[:, None, :], axis=2)
                residual = residual - projection[:, :, None] * basis[:, None, :]
                objective = tl.sum(residual * residual, axis=2) * weight
                objective = tl.where(valid_2d & ~used, objective, -1.0)
                _, pick = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit = candidate_ids == pick[:, None]
                used = used | hit
                picked_row = tl.sum(
                    tl.where(hit, candidate_rows_2d, 0), axis=1
                )
                seeds = seeds + picked_row[:, None] * (seed_slot == 1).to(tl.int32)
                basis_source = tl.sum(
                    tl.where(hit[:, :, None], residual, 0.0), axis=1
                )

                basis = basis_source * tl.rsqrt(
                    tl.sum(basis_source * basis_source, axis=1) + 1e-12
                )[:, None]
                projection = tl.sum(residual * basis[:, None, :], axis=2)
                residual = residual - projection[:, :, None] * basis[:, None, :]
                objective = tl.sum(residual * residual, axis=2) * weight
                objective = tl.where(valid_2d & ~used, objective, -1.0)
                _, pick2 = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit2 = candidate_ids == pick2[:, None]
                used = used | hit2
                row2 = tl.sum(tl.where(hit2, candidate_rows_2d, 0), axis=1)
                seeds = seeds + row2[:, None] * (seed_slot == 2).to(tl.int32)
                objective = tl.where(valid_2d & ~used, objective, -1.0)
                _, pick3 = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit3 = candidate_ids == pick3[:, None]
                row3 = tl.sum(tl.where(hit3, candidate_rows_2d, 0), axis=1)
                seeds = seeds + row3[:, None] * (seed_slot == 3).to(tl.int32)
            else:
                # Anchor-top3: one projection followed by three ranked picks.
                tl.static_assert(C == 4)
                basis = basis_source * tl.rsqrt(
                    tl.sum(basis_source * basis_source, axis=1) + 1e-12
                )[:, None]
                projection = tl.sum(residual * basis[:, None, :], axis=2)
                residual = residual - projection[:, :, None] * basis[:, None, :]
                objective = tl.sum(residual * residual, axis=2) * weight
                objective = tl.where(valid_2d & ~used, objective, -1.0)
                for step in tl.static_range(1, C):
                    _, pick = tl.max(
                        objective,
                        axis=1,
                        return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    hit = candidate_ids == pick[:, None]
                    picked_row = tl.sum(
                        tl.where(hit, candidate_rows_2d, 0), axis=1
                    )
                    seeds = seeds + picked_row[:, None] * (seed_slot == step).to(tl.int32)
                    objective = tl.where(hit, -1.0, objective)

            if FORCE_LOCAL_M1:
                seed0 = tl.sum(tl.where(seed_slot == 0, seeds, 0), axis=1)
                seed1 = tl.sum(tl.where(seed_slot == 1, seeds, 0), axis=1)
                seed2 = tl.sum(tl.where(seed_slot == 2, seeds, 0), axis=1)
                seed3 = tl.sum(tl.where(seed_slot == 3, seeds, 0), axis=1)
                local = tl.maximum(key_row - 1, 0)
                force = local != seed0
                new1 = tl.where(force, local, seed1)
                new2 = tl.where(force, tl.where(seed1 != local, seed1, seed2), seed2)
                new3 = tl.where(
                    force,
                    tl.where(seed1 == local, seed3, tl.where(seed2 == local, seed3, seed2)),
                    seed3,
                )
                seeds = (
                    seed0[:, None] * (seed_slot == 0).to(tl.int32)
                    + new1[:, None] * (seed_slot == 1).to(tl.int32)
                    + new2[:, None] * (seed_slot == 2).to(tl.int32)
                    + new3[:, None] * (seed_slot == 3).to(tl.int32)
                )

            output_offset = ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                                 * ROWS + key_row) * COLS + output_columns[:, None])
                               * HEADS + head) * C + seed_slot)
            tl.store(
                seed_row_ptr + output_offset,
                seeds.to(tl.int32),
                mask=output_mask & (seed_slot < C),
            )

    @triton.jit
    def _hellinger_gs_keytile_kernel(
        q_ptr, k_ptr, probe_ptr, candidate_ptr, anchor_ptr, seed_row_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr, TOP_PROBES: tl.constexpr,
        UNIFORM_PROBES: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_ROWS: tl.constexpr, N_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr, C: tl.constexpr, C_PAD: tl.constexpr,
        KEY_TILE: tl.constexpr, N_KEY_TILES: tl.constexpr,
        FORCE_LOCAL_M1: tl.constexpr,
        ALPHA: tl.constexpr, SCALE: tl.constexpr, INPUT_PRECISION: tl.constexpr,
    ):
        """Run unchanged Hellinger-GS selection for several key frames per dot."""
        tl.static_assert(C == 4)

        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        key_tile = pid % N_KEY_TILES
        pid = pid // N_KEY_TILES
        group = pid % N_GROUPS
        key_row = (pid // N_GROUPS) % ROWS
        query_outer = (pid // N_GROUPS) // ROWS

        dims = tl.arange(0, BLOCK_D)
        q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
        q_stride = COLS * HEADS * D_HEAD
        log2e: tl.constexpr = 1.4426950408889634
        P: tl.constexpr = TOP_PROBES + UNIFORM_PROBES

        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        column_slot = flat // BLOCK_ROWS
        candidate_slot = flat % BLOCK_ROWS
        columns = group * BLOCK_J + column_slot
        candidate_rows = tl.load(
            candidate_ptr + key_row * CANDIDATES + candidate_slot,
            mask=candidate_slot < CANDIDATES,
            other=0,
        )
        candidate_valid = (columns < COLS) & (candidate_slot < CANDIDATES)
        candidate = tl.load(
            q_ptr + q_base + candidate_rows[:, None] * q_stride
            + (columns[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=candidate_valid[:, None], other=0.0,
        )

        key_probe = tl.arange(0, KEY_TILE * P)
        key_slot = key_probe // P
        probe_slot = key_probe % P
        key_outer = key_tile * KEY_TILE + key_slot
        key_valid = key_outer < K_OUTER
        probe_offset = (
            (((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                * ROWS + key_row) * N_GROUPS + group) * HEADS + head)
            * P + probe_slot
        )
        probe_columns = tl.load(
            probe_ptr + probe_offset,
            mask=key_valid,
            other=0,
        )
        probe_key_offset = (
            ((batch * K_OUTER + key_outer[:, None]) * ROWS + key_row)
            * COLS * HEADS * D_HEAD
            + (probe_columns[:, None] * HEADS + head) * D_HEAD
            + dims[None, :]
        )
        probe_keys = tl.load(
            k_ptr + probe_key_offset,
            mask=key_valid[:, None],
            other=0.0,
        )
        logits = tl.dot(
            candidate,
            tl.trans(probe_keys),
            input_precision=INPUT_PRECISION,
        )
        logits = logits * (SCALE * log2e)
        logits = tl.reshape(
            logits,
            (BLOCK_J, BLOCK_ROWS, KEY_TILE, P),
        )
        logits = tl.permute(logits, (0, 2, 1, 3))
        logits = tl.reshape(logits, (BLOCK_J * KEY_TILE, BLOCK_ROWS, P))

        pair = tl.arange(0, BLOCK_J * KEY_TILE)
        pair_column_slot = pair // KEY_TILE
        pair_key_slot = pair % KEY_TILE
        output_columns = group * BLOCK_J + pair_column_slot
        output_key_outer = key_tile * KEY_TILE + pair_key_slot
        pair_valid = (output_columns < COLS) & (output_key_outer < K_OUTER)
        candidate_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        candidate_rows_2d = tl.reshape(candidate_rows, (BLOCK_J, BLOCK_ROWS))
        candidate_rows_pair = candidate_rows_2d[:, None, :] + tl.zeros(
            (1, KEY_TILE, 1), tl.int32
        )
        candidate_rows_pair = tl.reshape(
            candidate_rows_pair, (BLOCK_J * KEY_TILE, BLOCK_ROWS)
        )
        row_valid = pair_valid[:, None] & (candidate_ids < CANDIDATES)
        valid_count = tl.sum(row_valid.to(tl.float32), axis=1)
        valid_count = tl.maximum(valid_count, 1.0)

        row_max = tl.max(logits, axis=2)
        exponential = tl.exp2(logits - row_max[:, :, None])
        denominator = tl.sum(exponential, axis=2)
        log_mass = row_max + tl.log2(denominator)
        probability = exponential * (1.0 / denominator)[:, :, None]
        residual = tl.sqrt(probability)
        residual = residual - tl.sum(residual, axis=2)[:, :, None] * (1.0 / P)

        mean = tl.sum(tl.where(row_valid, log_mass, 0.0), axis=1) / valid_count
        deviation = log_mass - mean[:, None]
        variance = (
            tl.sum(tl.where(row_valid, deviation * deviation, 0.0), axis=1)
            / valid_count
        )
        z_score = deviation * tl.rsqrt(variance[:, None] + 1e-6)
        weight = tl.exp2((2.0 * ALPHA * log2e) * z_score)

        anchor_index = tl.load(anchor_ptr + key_row)
        used = candidate_ids == anchor_index
        seed_slot = tl.arange(0, C_PAD)[None, :]
        seed0 = tl.sum(tl.where(used, candidate_rows_pair, 0), axis=1)
        seeds = seed0[:, None] * (seed_slot == 0).to(tl.int32)
        basis_source = tl.sum(
            tl.where(used[:, :, None], residual, 0.0), axis=1
        )
        for step in tl.static_range(1, C):
            basis = basis_source * tl.rsqrt(
                tl.sum(basis_source * basis_source, axis=1) + 1e-12
            )[:, None]
            projection = tl.sum(residual * basis[:, None, :], axis=2)
            residual = residual - projection[:, :, None] * basis[:, None, :]
            objective = tl.sum(residual * residual, axis=2) * weight
            objective = tl.where(row_valid & ~used, objective, -1.0)
            _, pick = tl.max(
                objective,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            hit = candidate_ids == pick[:, None]
            used = used | hit
            picked_row = tl.sum(
                tl.where(hit, candidate_rows_pair, 0), axis=1
            )
            seeds = seeds + picked_row[:, None] * (seed_slot == step).to(tl.int32)
            basis_source = tl.sum(
                tl.where(hit[:, :, None], residual, 0.0), axis=1
            )

        if FORCE_LOCAL_M1:
            seed0 = tl.sum(tl.where(seed_slot == 0, seeds, 0), axis=1)
            seed1 = tl.sum(tl.where(seed_slot == 1, seeds, 0), axis=1)
            seed2 = tl.sum(tl.where(seed_slot == 2, seeds, 0), axis=1)
            seed3 = tl.sum(tl.where(seed_slot == 3, seeds, 0), axis=1)
            local = tl.maximum(key_row - 1, 0)
            force = local != seed0
            new1 = tl.where(force, local, seed1)
            new2 = tl.where(force, tl.where(seed1 != local, seed1, seed2), seed2)
            new3 = tl.where(
                force,
                tl.where(seed1 == local, seed3, tl.where(seed2 == local, seed3, seed2)),
                seed3,
            )
            seeds = (
                seed0[:, None] * (seed_slot == 0).to(tl.int32)
                + new1[:, None] * (seed_slot == 1).to(tl.int32)
                + new2[:, None] * (seed_slot == 2).to(tl.int32)
                + new3[:, None] * (seed_slot == 3).to(tl.int32)
            )

        output_offset = (
            (((((batch * Q_OUTER + query_outer) * K_OUTER + output_key_outer[:, None])
                * ROWS + key_row) * COLS + output_columns[:, None]) * HEADS + head)
            * C + seed_slot
        )
        tl.store(
            seed_row_ptr + output_offset,
            seeds.to(tl.int32),
            mask=pair_valid[:, None] & (seed_slot < C),
        )

    @triton.jit
    def _temporal_local_refine_kernel(
        q_ptr, k_ptr, coarse_ptr, seed_row_ptr,
        Q_OUTER: tl.constexpr, K_OUTER: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        RADIUS: tl.constexpr, PROBES: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_ROWS: tl.constexpr,
        N_GROUPS: tl.constexpr, BLOCK_PROBES: tl.constexpr,
        BLOCK_D: tl.constexpr, SKETCH_D: tl.constexpr, C: tl.constexpr,
        ALPHA: tl.constexpr, SCALE: tl.constexpr,
        COMPACT: tl.constexpr, DEDUPLICATE: tl.constexpr,
        FAST_LOGIT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        """Refine one temporal seed table per target frame and attention head."""
        tl.static_assert(C == 4, "temporal refinement currently targets C4")
        tl.static_assert(RADIUS == 2, "the production local pool uses radius two")
        tl.static_assert(PROBES == 8, "the production refinement uses eight probes")
        if COMPACT:
            tl.static_assert(BLOCK_ROWS == 8, "compact refinement uses eight rows")
        else:
            tl.static_assert(BLOCK_ROWS == 16, "full refinement uses sixteen rows")

        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        group = pid % N_GROUPS
        key_row = (pid // N_GROUPS) % ROWS
        query_outer = (pid // N_GROUPS) // ROWS

        dims = tl.arange(0, SKETCH_D)
        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        column_slot = flat // BLOCK_ROWS
        candidate_slot = flat % BLOCK_ROWS
        columns = group * BLOCK_J + column_slot
        column_valid = columns < COLS
        candidate_valid = column_valid

        if COMPACT:
            # anchor + radius-two neighborhood of the strongest coarse row +
            # the remaining two coarse rows: exactly eight candidates.
            coarse_seed_slot = tl.where(
                candidate_slot <= 5,
                1,
                candidate_slot - 4,
            )
            neighbor_offset = tl.where(
                candidate_slot <= 5,
                candidate_slot - 3,
                0,
            )
        else:
            neighborhood = 2 * RADIUS + 1
            dynamic_slot = (candidate_slot - 1) // neighborhood
            neighbor_offset = (candidate_slot - 1) % neighborhood - RADIUS
            coarse_seed_slot = dynamic_slot + 1
        coarse_offset = (
            (((((batch * K_OUTER) * ROWS + key_row) * COLS + columns)
               * HEADS + head) * C)
            + coarse_seed_slot
        )

        for key_outer in tl.range(0, K_OUTER):
            key_coarse_offset = coarse_offset + key_outer * ROWS * COLS * HEADS * C
            coarse_row = tl.load(
                coarse_ptr + key_coarse_offset,
                mask=column_valid & (candidate_slot > 0),
                other=key_row,
            ).to(tl.int32)
            candidate_row = tl.where(
                candidate_slot == 0,
                key_row,
                tl.minimum(tl.maximum(coarse_row + neighbor_offset, 0), ROWS - 1),
            )

            q_base = (batch * Q_OUTER + query_outer) * ROWS * COLS * HEADS * D_HEAD
            candidate_q = tl.load(
                q_ptr + q_base
                + (candidate_row[:, None] * COLS + columns[:, None]) * HEADS * D_HEAD
                + head * D_HEAD + dims[None, :],
                mask=candidate_valid[:, None],
                other=0.0,
            )

            probe_slot = tl.arange(0, BLOCK_PROBES)
            probe_valid = probe_slot < PROBES
            probe_column = probe_slot * (COLS // PROBES)
            key_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS * HEADS * D_HEAD
            probe_key = tl.load(
                k_ptr + key_base
                + (probe_column[:, None] * HEADS + head) * D_HEAD
                + dims[None, :],
                mask=probe_valid[:, None],
                other=0.0,
            )

            logits = tl.dot(
                candidate_q, tl.trans(probe_key), input_precision=INPUT_PRECISION
            ) * (SCALE * 1.4426950408889634)
            if FAST_LOGIT:
                profile = tl.where(probe_valid[None, :], logits, 0.0)
            else:
                logits = tl.where(probe_valid[None, :], logits, -float("inf"))
                row_max = tl.max(logits, axis=1)
                exponential = tl.exp2(logits - row_max[:, None])
                denominator = tl.sum(exponential, axis=1)
                log_mass = row_max + tl.log2(denominator)
                probability = exponential / denominator[:, None]
                profile = tl.where(
                    probe_valid[None, :], tl.sqrt(probability), 0.0
                )
            profile = tl.reshape(
                profile, (BLOCK_J, BLOCK_ROWS, BLOCK_PROBES)
            ).to(tl.float32)
            profile_mean = tl.sum(profile, axis=2) * (1.0 / PROBES)
            residual = tl.where(
                probe_valid[None, None, :],
                profile - profile_mean[:, :, None],
                0.0,
            )

            candidate_rows = tl.reshape(candidate_row, (BLOCK_J, BLOCK_ROWS))
            valid = tl.reshape(candidate_valid, (BLOCK_J, BLOCK_ROWS))
            candidate_ids = tl.arange(0, BLOCK_ROWS)[None, :]
            if DEDUPLICATE:
                for previous in tl.static_range(0, BLOCK_ROWS):
                    previous_row = tl.sum(
                        tl.where(candidate_ids == previous, candidate_rows, 0), axis=1
                    )
                    valid = valid & (
                        (candidate_ids <= previous)
                        | (candidate_rows != previous_row[:, None])
                    )

            if FAST_LOGIT:
                weight = tl.full((BLOCK_J, BLOCK_ROWS), 1.0, tl.float32)
            else:
                mass = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS))
                valid_count = tl.sum(valid.to(tl.float32), axis=1)
                mean = tl.sum(tl.where(valid, mass, 0.0), axis=1) / valid_count
                deviation = mass - mean[:, None]
                variance = (
                    tl.sum(tl.where(valid, deviation * deviation, 0.0), axis=1)
                    / valid_count
                )
                z_score = deviation * tl.rsqrt(variance[:, None] + 1e-6)
                weight = tl.exp2(
                    (2.0 * ALPHA * 1.4426950408889634) * z_score
                )

            used = candidate_rows == key_row
            basis_source = tl.sum(
                tl.where(candidate_ids[:, :, None] == 0, residual, 0.0), axis=1
            )
            pick1 = tl.full((BLOCK_J,), key_row, tl.int32)
            pick2 = tl.full((BLOCK_J,), key_row, tl.int32)
            pick3 = tl.full((BLOCK_J,), key_row, tl.int32)
            for step in tl.static_range(0, 3):
                basis = basis_source * tl.rsqrt(
                    tl.sum(basis_source * basis_source, axis=1) + 1e-12
                )[:, None]
                projection = tl.sum(residual * basis[:, None, :], axis=2)
                residual = residual - projection[:, :, None] * basis[:, None, :]
                objective = tl.sum(residual * residual, axis=2) * weight
                objective = tl.where(valid & ~used, objective, -1.0)
                _, position = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit = candidate_ids == position[:, None]
                picked_row = tl.sum(tl.where(hit, candidate_rows, 0), axis=1)
                used = used | (candidate_rows == picked_row[:, None])
                basis_source = tl.sum(
                    tl.where(hit[:, :, None], residual, 0.0), axis=1
                )
                pick1 = tl.where(step == 0, picked_row, pick1)
                pick2 = tl.where(step == 1, picked_row, pick2)
                pick3 = tl.where(step == 2, picked_row, pick3)

            local_row = tl.maximum(key_row - 1, 0)
            use_local = key_row > 0
            dynamic0 = tl.where(pick1 != local_row, pick1, pick2)
            dynamic1 = tl.where(
                pick1 == local_row,
                pick3,
                tl.where(pick2 != local_row, pick2, pick3),
            )
            seed_slot = tl.arange(0, C)[None, :]
            local_seeds = tl.where(
                seed_slot == 0, key_row,
                tl.where(
                    seed_slot == 1, local_row,
                    tl.where(seed_slot == 2, dynamic0[:, None], dynamic1[:, None]),
                ),
            )
            boundary_seeds = tl.where(
                seed_slot == 0, key_row,
                tl.where(
                    seed_slot == 1, pick1[:, None],
                    tl.where(seed_slot == 2, pick2[:, None], pick3[:, None]),
                ),
            )
            seeds = tl.where(use_local, local_seeds, boundary_seeds)
            output_columns = group * BLOCK_J + tl.arange(0, BLOCK_J)
            output_offset = (
                ((((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                     * ROWS + key_row) * COLS + output_columns[:, None])
                   * HEADS + head) * C)
                + seed_slot
            )
            tl.store(
                seed_row_ptr + output_offset,
                seeds.to(tl.int32),
                mask=(output_columns < COLS)[:, None],
            )


def span_seed_rows_triton_split(q, k, v, scale=None, *, alpha=0.3,
                                candidate_count=30, top_probes=8,
                                uniform_probes=8, value_dims=32, block_j=2,
                                group_tile=16, key_outer_tile=None,
                                output_dtype=torch.int32,
                                probe_num_warps=4, gs_num_warps=4):
    """Seed table via the split probe/Gram-Schmidt kernels.

    Defaults BLOCK_J=2, the quality-best sharing width the split exists to
    make affordable (rel-L2 0.06670, selector 63.1 ms isolated at 12 heads).
    block_j=4 is the speed-frontier option: 53-57 ms at rel-L2 ~0.0679, the
    same probe sharing as the previous fused production default. Pool width
    is free here -- the candidate tile is next_pow2(candidate_count), so
    pool 21 and pool 30 cost the same and 30 should always be used.
    """
    if triton is None:
        raise RuntimeError("Triton is required for span_seed_rows_triton_split.")
    if output_dtype not in (torch.int32, torch.uint8):
        raise ValueError("Weighted-Span seed rows must use int32 or uint8 output")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, A, ROWS, COLS, H, D = q.shape
    F = k.size(1)
    if key_outer_tile is None:
        key_outer_tile = F
    key_outer_tile = int(key_outer_tile)
    if key_outer_tile <= 0 or F % key_outer_tile:
        raise ValueError(
            f"key_outer_tile must be a positive divisor of {F}, got "
            f"{key_outer_tile}."
        )
    if scale is None:
        scale = D ** -0.5
    P = top_probes + uniform_probes
    probe_pad = max(16, triton.next_power_of_2(P))
    n_groups = triton.cdiv(COLS, block_j)
    # Keep probe indices in int32. Although every column fits in uint8, the
    # packed uint8 layout changes the selector result for the production
    # 40-head grid; int32 is bit-identical to the original Weighted-Span path.
    probes = torch.empty((B, A, F, ROWS, n_groups, H, probe_pad),
                         device=q.device, dtype=torch.int32)
    table = torch.empty((B, A, F, ROWS, COLS, H, C4),
                        device=q.device, dtype=output_dtype)
    block_l = triton.next_power_of_2(COLS)
    block_rows = triton.next_power_of_2(candidate_count)
    m_tiles = triton.cdiv(n_groups, group_tile)
    n_key_outer_tiles = F // key_outer_tile
    _span_probe_kernel[(A * F * ROWS * m_tiles, B * H)](
        q, k, probes, A, F, ROWS, COLS, H, D,
        TOP_PROBES=top_probes, UNIFORM_PROBES=uniform_probes,
        PROBE_PAD=probe_pad,
        BLOCK_J=block_j, N_GROUPS=n_groups, GROUP_TILE=group_tile,
        BLOCK_L=block_l, BLOCK_D=triton.next_power_of_2(D),
        SCALE=float(scale), INPUT_PRECISION="ieee",
        num_warps=int(probe_num_warps), num_stages=1)
    _span_gs_kernel[(A * ROWS * n_groups * n_key_outer_tiles, B * H)](
        q, k, v, probes, table, A, F, ROWS, COLS, H, D,
        CANDIDATES=candidate_count, TOP_PROBES=top_probes,
        UNIFORM_PROBES=uniform_probes, VDIMS=value_dims,
        PROBE_PAD=probe_pad,
        BLOCK_J=block_j, BLOCK_ROWS=block_rows, N_GROUPS=n_groups,
        KEY_OUTER_TILE=key_outer_tile,
        N_KEY_OUTER_TILES=n_key_outer_tiles,
        BLOCK_D=triton.next_power_of_2(D), C=C4,
        ALPHA=float(alpha), SCALE=float(scale), INPUT_PRECISION="ieee",
        num_warps=int(gs_num_warps), num_stages=1)
    return table


def span_seed_rows_triton_temporal_probes_per_frame(
    q, k, v, scale=None, *, alpha=0.3, candidate_count=45,
    top_probes=8, uniform_probes=8, value_dims=8, block_j=4,
    group_tile=16, router_frames=3,
):
    """Run per-frame span selection with temporally shared probe columns.

    Probe columns are selected from centered router frames, but every target
    frame retains its own candidate Q profiles and Gram-Schmidt seed rows.
    The native uint8 output avoids the multi-gigabyte int32 table permutation
    that otherwise sits between the selector and the C4 merge kernel.
    """
    if triton is None:
        raise RuntimeError(
            "Triton is required for temporal-probe per-frame span selection."
        )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if q.ndim != 6:
        raise ValueError("temporal-probe span selection expects 6D Q/K/V blocks")
    batch, query_outer, rows, columns, heads, head_dim = q.shape
    key_outer = k.size(1)
    router_frames = int(router_frames)
    if query_outer != 21 or router_frames != 3:
        raise ValueError(
            "the production temporal-probe selector requires 21 frames and 3 routers"
        )
    if query_outer % router_frames:
        raise ValueError("query frame count must be divisible by router count")
    if rows != 45 or columns != 80:
        raise ValueError("the production temporal-probe selector requires 45x80")
    if scale is None:
        scale = head_dim ** -0.5

    router_indices = torch.tensor(
        [3, 10, 17], device=q.device, dtype=torch.long
    )
    router_query = q.index_select(1, router_indices).contiguous()
    probe_count = top_probes + uniform_probes
    probe_pad = max(16, triton.next_power_of_2(probe_count))
    group_count = triton.cdiv(columns, block_j)
    probes = torch.empty(
        (
            batch, router_frames, key_outer, rows,
            group_count, heads, probe_pad,
        ),
        device=q.device,
        dtype=torch.int32,
    )
    table = torch.empty(
        (
            batch, query_outer, key_outer, rows,
            columns, heads, C4,
        ),
        device=q.device,
        dtype=torch.uint8,
    )
    block_l = triton.next_power_of_2(columns)
    block_rows = triton.next_power_of_2(candidate_count)
    group_tiles = triton.cdiv(group_count, group_tile)
    _span_probe_kernel[
        (router_frames * key_outer * rows * group_tiles, batch * heads)
    ](
        router_query, k, probes,
        router_frames, key_outer, rows, columns, heads, head_dim,
        TOP_PROBES=top_probes,
        UNIFORM_PROBES=uniform_probes,
        PROBE_PAD=probe_pad,
        BLOCK_J=block_j,
        N_GROUPS=group_count,
        GROUP_TILE=group_tile,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(head_dim),
        SCALE=float(scale),
        INPUT_PRECISION="ieee",
        num_warps=4,
        num_stages=1,
    )
    _span_gs_temporal_probe_shared_kernel[
        (query_outer * rows * group_count, batch * heads)
    ](
        q, k, v, probes, table,
        query_outer, router_frames, key_outer,
        rows, columns, heads, head_dim,
        CANDIDATES=candidate_count,
        TOP_PROBES=top_probes,
        UNIFORM_PROBES=uniform_probes,
        VDIMS=value_dims,
        PROBE_PAD=probe_pad,
        BLOCK_J=block_j,
        BLOCK_ROWS=block_rows,
        N_GROUPS=group_count,
        BLOCK_D=triton.next_power_of_2(head_dim),
        C=C4,
        ALPHA=float(alpha),
        SCALE=float(scale),
        INPUT_PRECISION="ieee",
        num_warps=4,
        num_stages=1,
    )
    return table


def span_seed_rows_triton_temporal_consensus2(
    q, k, v, scale=None, *, alpha=0.3, candidate_count=30,
    top_probes=8, uniform_probes=8, value_dims=8, block_j=4,
    group_tile=16,
):
    """Select one shared seed table from two temporal response profiles."""
    if triton is None:
        raise RuntimeError(
            "Triton is required for span_seed_rows_triton_temporal_consensus2."
        )
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, A, ROWS, COLS, H, D = q.shape
    if A != 2:
        raise ValueError("temporal consensus requires exactly two router frames")
    F = k.size(1)
    if scale is None:
        scale = D ** -0.5
    P = top_probes + uniform_probes
    probe_pad = max(16, triton.next_power_of_2(P))
    n_groups = triton.cdiv(COLS, block_j)
    probes = torch.empty(
        (B, A, F, ROWS, n_groups, H, probe_pad),
        device=q.device, dtype=torch.int32,
    )
    table = torch.empty(
        (B, 1, F, ROWS, COLS, H, C4),
        device=q.device, dtype=torch.int32,
    )
    block_l = triton.next_power_of_2(COLS)
    block_rows = triton.next_power_of_2(candidate_count)
    m_tiles = triton.cdiv(n_groups, group_tile)
    _span_probe_kernel[(A * F * ROWS * m_tiles, B * H)](
        q, k, probes, A, F, ROWS, COLS, H, D,
        TOP_PROBES=top_probes, UNIFORM_PROBES=uniform_probes,
        PROBE_PAD=probe_pad,
        BLOCK_J=block_j, N_GROUPS=n_groups, GROUP_TILE=group_tile,
        BLOCK_L=block_l, BLOCK_D=triton.next_power_of_2(D),
        SCALE=float(scale), INPUT_PRECISION="ieee",
        num_warps=4, num_stages=1,
    )
    _span_gs_temporal_consensus2_kernel[(ROWS * n_groups, B * H)](
        q, k, v, probes, table, F, ROWS, COLS, H, D,
        CANDIDATES=candidate_count, TOP_PROBES=top_probes,
        UNIFORM_PROBES=uniform_probes, VDIMS=value_dims,
        PROBE_PAD=probe_pad,
        BLOCK_J=block_j, BLOCK_ROWS=block_rows, N_GROUPS=n_groups,
        BLOCK_D=triton.next_power_of_2(D), C=C4,
        ALPHA=float(alpha), SCALE=float(scale), INPUT_PRECISION="ieee",
        num_warps=4, num_stages=1,
    )
    return table


def temporal_query_row_map_triton(q, *, router_frame=10, radius=2):
    """Map router-frame rows to nearby rows in every frame using Q only.

    The output layout is ``[B, T, column, head, router_row]``. It lets the
    merge kernel follow temporal feature motion without rerunning the
    key-conditioned span selector.
    """

    if triton is None:
        raise RuntimeError("Triton is required for temporal query row tracking.")
    q = q.contiguous()
    if q.ndim != 6:
        raise ValueError("temporal row tracking expects [B, T, row, column, H, D]")
    batch, query_outer, rows, columns, heads, dim = q.shape
    router_frame = int(router_frame)
    radius = int(radius)
    if not 0 <= router_frame < query_outer:
        raise ValueError("temporal row-tracking router frame is outside the video")
    if radius not in {1, 2, 3, 4}:
        raise ValueError("temporal row-tracking radius must be 1, 2, 3, or 4")
    if rows > 64 or dim > 128:
        raise ValueError("temporal row tracking supports at most 64 rows and D=128")
    row_map = torch.empty(
        (batch, query_outer, columns, heads, rows),
        device=q.device,
        dtype=torch.uint8,
    )
    _temporal_query_row_map_kernel[(query_outer * columns, batch * heads)](
        q,
        row_map,
        query_outer,
        rows,
        columns,
        heads,
        dim,
        ROUTER_FRAME=router_frame,
        RADIUS=radius,
        BLOCK_ROWS=64,
        BLOCK_D=triton.next_power_of_2(dim),
        INPUT_PRECISION="ieee",
        num_warps=8,
        num_stages=1,
    )
    return row_map


def hellinger_span_seed_rows_triton(
    q,
    k,
    scale=None,
    *,
    alpha=0.3,
    candidate_count=22,
    candidate_geometry_count=None,
    top_probes=8,
    uniform_probes=8,
    probe_anchor_count=2,
    block_j=4,
    group_tile=16,
    output_dtype=torch.int32,
    probe_num_warps=4,
    gs_num_warps=4,
    probe_num_stages=1,
    gs_num_stages=1,
    gs_key_tile=1,
    num_seeds=4,
    selection_mode="gs",
    force_local_minus_one=False,
    timings=None,
):
    """Hardware-aligned Hellinger-GS seed selector for all-layer Hard C4/C5."""
    if triton is None:
        raise RuntimeError("Triton is required for hellinger_span_seed_rows_triton.")
    if probe_anchor_count != 2:
        raise ValueError("the production Hellinger selector currently requires two anchors")
    requested_probe_count = int(top_probes) + int(uniform_probes)
    if uniform_probes <= 0:
        raise ValueError("the production Hellinger selector requires uniform probes")
    if candidate_count > 45:
        raise ValueError("the production Hellinger selector supports at most 45 candidates")
    candidate_geometry_count = int(candidate_geometry_count or candidate_count)
    if not candidate_count <= candidate_geometry_count <= 45:
        raise ValueError(
            "candidate_geometry_count must be between candidate_count and 45"
        )
    num_seeds = int(num_seeds)
    if num_seeds not in {4, 5, 6, 8}:
        raise ValueError(
            "the production Hellinger selector supports 4, 5, 6, or 8 seeds"
        )
    if candidate_count < num_seeds:
        raise ValueError("candidate_count must be at least num_seeds")
    selection_modes = {
        "gs": 0,
        "two_stage": 1,
        "anchor_top3": 2,
        "pivoted_cholesky": 3,
        "pair_first_m8": 4,
        "pair_first_m4": 5,
    }
    try:
        selection_mode_id = selection_modes[str(selection_mode)]
    except KeyError as exc:
        raise ValueError(
            "Hellinger selection_mode must be 'gs', 'two_stage', "
            "'anchor_top3', 'pivoted_cholesky', 'pair_first_m8', or "
            "'pair_first_m4'"
        ) from exc
    if selection_mode_id and num_seeds != 4:
        raise ValueError("simplified Hellinger selectors require exactly four seeds")
    if output_dtype not in (torch.int32, torch.uint8):
        raise ValueError("Hellinger seed rows support int32 or uint8 output")
    gs_key_tile = int(gs_key_tile)
    if gs_key_tile not in {1, 2, 4, 8}:
        raise ValueError("Hellinger GS key tile must be 1, 2, 4, or 8")
    if gs_key_tile > 1 and num_seeds != 4:
        raise ValueError("key-tiled Hellinger GS currently supports exactly four seeds")
    if gs_key_tile > 1 and selection_mode_id:
        raise ValueError("simplified Hellinger selectors currently require gs_key_tile=1")
    force_local_minus_one = bool(force_local_minus_one)
    if force_local_minus_one and num_seeds != 4:
        raise ValueError("local-minus-one Hellinger routing requires exactly four seeds")

    from .adaptive_seed_pytorch import multiscale_candidate_table

    q, k = q.contiguous(), k.contiguous()
    batch, query_outer, rows, columns, heads, head_dim = q.shape
    key_outer = k.size(1)
    full_keys = requested_probe_count == columns
    if requested_probe_count != 16 and not full_keys:
        raise ValueError(
            "the production Hellinger selector requires 16 probes or every key column"
        )
    if full_keys and gs_key_tile != 1:
        raise ValueError("full-key Hellinger routing currently requires gs_key_tile=1")
    if scale is None:
        scale = head_dim**-0.5
    probe_count = requested_probe_count
    group_count = triton.cdiv(columns, block_j)
    if full_keys:
        # The kernel derives 0..COLS-1 directly. Avoid materializing the
        # [Q,K,row,column-group,head,COLS] index tensor (about 242 MiB at 720p).
        probes = torch.empty((1,), device=q.device, dtype=torch.int32)
    else:
        probes = torch.empty(
            (
                batch,
                query_outer,
                key_outer,
                rows,
                group_count,
                heads,
                probe_count,
            ),
            device=q.device,
            dtype=torch.int32,
        )
    table = torch.empty(
        (batch, query_outer, key_outer, rows, columns, heads, num_seeds),
        device=q.device,
        dtype=output_dtype,
    )
    candidate_rows, anchor_indices = multiscale_candidate_table(
        rows,
        candidate_count,
        q.device,
        geometry_count=candidate_geometry_count,
    )
    block_l = triton.next_power_of_2(columns)
    block_rows = triton.next_power_of_2(candidate_count)
    group_tiles = triton.cdiv(group_count, group_tile)
    timing_events = None
    if timings is not None:
        timing_events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        timing_events[0].record()
    if not full_keys:
        _hellinger_probe_kernel[
            (query_outer * key_outer * rows * group_tiles, batch * heads)
        ](
            q,
            k,
            probes,
            query_outer,
            key_outer,
            rows,
            columns,
            heads,
            head_dim,
            TOP_PROBES=top_probes,
            UNIFORM_PROBES=uniform_probes,
            BLOCK_J=block_j,
            N_GROUPS=group_count,
            GROUP_TILE=group_tile,
            BLOCK_L=block_l,
            BLOCK_D=triton.next_power_of_2(head_dim),
            SCALE=float(scale),
            INPUT_PRECISION="ieee",
            num_warps=probe_num_warps,
            num_stages=probe_num_stages,
        )
    if timing_events is not None:
        timing_events[1].record()
    if gs_key_tile == 1:
        _hellinger_gs_kernel[(query_outer * rows * group_count, batch * heads)](
            q,
            k,
            probes,
            candidate_rows,
            anchor_indices,
            table,
            query_outer,
            key_outer,
            rows,
            columns,
            heads,
            head_dim,
            CANDIDATES=candidate_count,
            TOP_PROBES=top_probes,
            UNIFORM_PROBES=uniform_probes,
            P_PAD=triton.next_power_of_2(probe_count),
            BLOCK_J=block_j,
            BLOCK_ROWS=block_rows,
            N_GROUPS=group_count,
            BLOCK_D=triton.next_power_of_2(head_dim),
            C=num_seeds,
            C_PAD=triton.next_power_of_2(num_seeds),
            SELECTION_MODE=selection_mode_id,
            FORCE_LOCAL_M1=force_local_minus_one,
            FULL_KEYS=full_keys,
            ALPHA=float(alpha),
            SCALE=float(scale),
            INPUT_PRECISION="ieee",
            num_warps=gs_num_warps,
            num_stages=gs_num_stages,
        )
    else:
        key_tiles = triton.cdiv(key_outer, gs_key_tile)
        _hellinger_gs_keytile_kernel[
            (query_outer * rows * group_count * key_tiles, batch * heads)
        ](
            q,
            k,
            probes,
            candidate_rows,
            anchor_indices,
            table,
            query_outer,
            key_outer,
            rows,
            columns,
            heads,
            head_dim,
            CANDIDATES=candidate_count,
            TOP_PROBES=top_probes,
            UNIFORM_PROBES=uniform_probes,
            BLOCK_J=block_j,
            BLOCK_ROWS=block_rows,
            N_GROUPS=group_count,
            BLOCK_D=triton.next_power_of_2(head_dim),
            C=num_seeds,
            C_PAD=triton.next_power_of_2(num_seeds),
            KEY_TILE=gs_key_tile,
            N_KEY_TILES=key_tiles,
            FORCE_LOCAL_M1=force_local_minus_one,
            ALPHA=float(alpha),
            SCALE=float(scale),
            INPUT_PRECISION="ieee",
            num_warps=gs_num_warps,
            num_stages=gs_num_stages,
        )
    if timing_events is not None:
        timing_events[2].record()
        timing_events[2].synchronize()
        timings["probe_ms"] = timing_events[0].elapsed_time(timing_events[1])
        timings["gs_ms"] = timing_events[1].elapsed_time(timing_events[2])
    return table


def temporal_local_refine_seed_rows_triton(
    q,
    k,
    coarse_table,
    scale=None,
    *,
    radius=2,
    probes=8,
    alpha=0.3,
    block_j=4,
    compact=False,
    compact_block_j=8,
    deduplicate=True,
    fast_logit=False,
    sketch_dim=0,
    num_warps=4,
    num_stages=1,
):
    """Refine a one-frame coarse table for every query frame and head.

    The output keeps the index-matched row and its left neighbor, then adds two
    importance-weighted Hellinger-span rows from radius-two neighborhoods around
    the coarse dynamic seeds.  No dense attention is used.
    """
    if triton is None:
        raise RuntimeError("Triton is required for temporal seed refinement.")
    q, k, coarse_table = q.contiguous(), k.contiguous(), coarse_table.contiguous()
    batch, query_outer, rows, columns, heads, head_dim = q.shape
    key_outer = k.size(1)
    if scale is None:
        scale = head_dim**-0.5
    expected = (batch, 1, key_outer, rows, columns, heads, C4)
    if tuple(coarse_table.shape) != expected:
        raise ValueError(
            f"coarse temporal seed table must be {expected}, got {tuple(coarse_table.shape)}"
        )
    if coarse_table.dtype != torch.uint8:
        raise TypeError("coarse temporal seed table must use uint8 row indices")
    if radius != 2 or probes != 8:
        raise ValueError("the production temporal refinement requires radius=2, probes=8")
    effective_sketch_dim = head_dim if not sketch_dim else int(sketch_dim)
    if effective_sketch_dim not in (32, 64, 128):
        raise ValueError("temporal refinement sketch_dim must be 32, 64, or 128")
    if not fast_logit and effective_sketch_dim != head_dim:
        raise ValueError("reduced sketch_dim is currently limited to fast-logit mode")

    kernel_block_j = compact_block_j if compact else block_j
    if kernel_block_j not in (2, 4, 8, 16):
        raise ValueError("temporal refinement BLOCK_J must be one of 2, 4, 8, 16")
    group_count = triton.cdiv(columns, kernel_block_j)
    output = torch.empty(
        (batch, query_outer, key_outer, rows, columns, heads, C4),
        device=q.device,
        dtype=torch.uint8,
    )
    _temporal_local_refine_kernel[
        (query_outer * rows * group_count, batch * heads)
    ](
        q,
        k,
        coarse_table,
        output,
        query_outer,
        key_outer,
        rows,
        columns,
        heads,
        head_dim,
        RADIUS=radius,
        PROBES=probes,
        BLOCK_J=kernel_block_j,
        BLOCK_ROWS=8 if compact else 16,
        N_GROUPS=group_count,
        BLOCK_PROBES=16,
        BLOCK_D=triton.next_power_of_2(head_dim),
        SKETCH_D=effective_sketch_dim,
        C=C4,
        ALPHA=float(alpha),
        SCALE=float(scale),
        COMPACT=bool(compact),
        DEDUPLICATE=bool(deduplicate),
        FAST_LOGIT=bool(fast_logit),
        INPUT_PRECISION="ieee",
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


__all__ = [
    "SHAPE_CONFIG",
    "adaptive_seed_attention",
    "adaptive_seed_attention_triton",
    "hellinger_span_seed_rows_triton",
    "query_l2_kcenter_seed_rows_triton",
    "span_seed_rows_triton",
    "span_seed_rows_triton_split",
    "span_seed_rows_triton_temporal_probes_per_frame",
    "temporal_local_refine_seed_rows_triton",
    "temporal_query_row_map_triton",
]
