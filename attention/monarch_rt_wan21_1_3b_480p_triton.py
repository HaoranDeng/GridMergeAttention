"""One-step MonarchRT for Wan2.1-T2V-1.3B at 480p.

Validated config: ``configs/token_merging_wan_monarchrt_triton_fewstep.yaml``.

The kernels mirror the upstream MonarchRT forward pass:

1. For each ``30-query x 52-key`` style grid block, use the index-matched
   seed query to softmax-merge the key columns into one compressed K/V token.
2. Run a full softmax over all compressed tokens for each query.

Inputs and output use block layout ``[B, outer, row, column, heads, dim]``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


MONARCH_RT_BACKEND = "monarch_rt"
MONARCH_BLOCK_J = 64
MONARCH_BLOCK_L = 64
MONARCH_BLOCK_I = 32
MONARCH_BLOCK_K = 32
MONARCH_NUM_WARPS = 4
MONARCH_NUM_STAGES = 3
MONARCH_INPUT_PRECISION = "ieee"


if triton is not None:

    @triton.jit
    def _merge_columns_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        merged_k_ptr,
        merged_v_ptr,
        entropy_ptr,
        key_valid_ptr,
        compressed_valid_ptr,
        BATCH: tl.constexpr,
        VALID_BATCH: tl.constexpr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_D: tl.constexpr,
        J_TILES: tl.constexpr,
        FLAT_GRID: tl.constexpr,
        SCALE_SQRT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        HAS_KEY_VALID: tl.constexpr,
    ):
        if FLAT_GRID:
            linear_id = tl.program_id(0)
            j_start = (linear_id % J_TILES) * BLOCK_J
            block_id = linear_id // J_TILES
        else:
            j_start = tl.program_id(0) * BLOCK_J
            block_id = tl.program_id(2)
        batch_head = tl.program_id(1)

        head = batch_head % HEADS
        batch = batch_head // HEADS
        row = block_id % ROWS
        tmp = block_id // ROWS
        key_outer = tmp % K_OUTER
        query_outer = tmp // K_OUTER

        if HAS_KEY_VALID:
            valid_batch = batch if VALID_BATCH > 1 else 0
            row_is_valid = tl.load(
                compressed_valid_ptr
                + ((valid_batch * K_OUTER + key_outer) * ROWS + row)
            )
        else:
            valid_batch = 0
            row_is_valid = True

        js = j_start + tl.arange(0, BLOCK_J)
        ls = tl.arange(0, BLOCK_L)
        dims = tl.arange(0, BLOCK_D)
        j_mask = js < COLS
        d_mask = dims < D_HEAD

        q_base = (((batch * Q_OUTER + query_outer) * ROWS + row) * COLS + js[:, None])
        seed_q = tl.load(
            q_ptr + ((q_base * HEADS + head) * D_HEAD + dims[None, :]),
            mask=j_mask[:, None] & d_mask[None, :],
            other=0.0,
        ) * SCALE_SQRT

        row_max = tl.full((BLOCK_J,), -float("inf"), tl.float32)
        row_max = tl.where(row_is_valid, row_max, 0.0)
        row_sum = tl.zeros((BLOCK_J,), tl.float32)
        key_acc = tl.zeros((BLOCK_J, BLOCK_D), tl.float32)
        value_acc = tl.zeros((BLOCK_J, BLOCK_D), tl.float32)
        entropy_acc = tl.zeros((BLOCK_J,), tl.float32)

        for l_start in tl.static_range(0, COLS, BLOCK_L):
            key_cols = l_start + ls
            key_mask = key_cols < COLS
            if HAS_KEY_VALID:
                valid_base = (
                    ((valid_batch * K_OUTER + key_outer) * ROWS + row)
                    * COLS
                    + key_cols
                )
                key_mask = key_mask & tl.load(
                    key_valid_ptr + valid_base,
                    mask=key_cols < COLS,
                    other=False,
                )
            kv_base = (
                (((batch * K_OUTER + key_outer) * ROWS + row) * COLS + key_cols[:, None])
            )
            key = tl.load(
                k_ptr + ((kv_base * HEADS + head) * D_HEAD + dims[None, :]),
                mask=key_mask[:, None] & d_mask[None, :],
                other=0.0,
            ) * SCALE_SQRT
            value = tl.load(
                v_ptr + ((kv_base * HEADS + head) * D_HEAD + dims[None, :]),
                mask=key_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            logits = tl.dot(seed_q, tl.trans(key), input_precision=INPUT_PRECISION)
            logits = logits * 1.4426950408889634
            logits = tl.where(j_mask[:, None] & key_mask[None, :], logits, -float("inf"))

            new_max = tl.maximum(row_max, tl.max(logits, axis=1))
            probs = tl.exp2(logits - new_max[:, None])
            block_sum = tl.sum(probs, axis=1)
            old_scale = tl.exp2(row_max - new_max)

            key_acc = key_acc * old_scale[:, None] + tl.dot(probs.to(seed_q.dtype), key, input_precision=INPUT_PRECISION)
            value_acc = value_acc * old_scale[:, None] + tl.dot(probs.to(seed_q.dtype), value, input_precision=INPUT_PRECISION)
            logit_probs = tl.where(key_mask[None, :], logits * probs, 0.0)
            entropy_acc = entropy_acc * old_scale + tl.sum(logit_probs, axis=1)
            row_sum = row_sum * old_scale + block_sum
            row_max = new_max

        safe_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
        inv_sum = tl.where(row_sum > 0.0, 1.0 / safe_sum, 0.0)
        key_acc = key_acc * inv_sum[:, None]
        value_acc = value_acc * inv_sum[:, None]
        entropy = entropy_acc * inv_sum - (row_max + tl.log2(safe_sum))
        entropy = tl.where(row_sum > 0.0, entropy, 0.0)

        out_base = (
            ((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer) * COLS + js[:, None]) * ROWS + row)
        )
        tl.store(
            merged_k_ptr + (((out_base * HEADS + head) * D_HEAD) + dims[None, :]),
            key_acc,
            mask=j_mask[:, None] & d_mask[None, :],
        )
        tl.store(
            merged_v_ptr + (((out_base * HEADS + head) * D_HEAD) + dims[None, :]),
            value_acc,
            mask=j_mask[:, None] & d_mask[None, :],
        )
        tl.store(
            entropy_ptr + (((((batch * HEADS + head) * Q_OUTER + query_outer) * K_OUTER + key_outer) * COLS + js) * ROWS + row),
            entropy,
            mask=j_mask,
        )

    @triton.jit
    def _route_rows_kernel(
        q_ptr,
        merged_k_ptr,
        merged_v_ptr,
        entropy_ptr,
        compressed_valid_ptr,
        out_ptr,
        BATCH: tl.constexpr,
        VALID_BATCH: tl.constexpr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE_SQRT: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        HAS_KEY_VALID: tl.constexpr,
    ):
        i_start = tl.program_id(0) * BLOCK_I
        batch_head = tl.program_id(1)
        query_group = tl.program_id(2)

        head = batch_head % HEADS
        batch = batch_head // HEADS
        valid_batch = batch if VALID_BATCH > 1 else 0
        query_outer = query_group // COLS
        query_col = query_group % COLS

        rows = i_start + tl.arange(0, BLOCK_I)
        key_rows = tl.arange(0, BLOCK_K)
        dims = tl.arange(0, BLOCK_D)
        row_mask = rows < ROWS
        d_mask = dims < D_HEAD

        q_base = (((batch * Q_OUTER + query_outer) * ROWS + rows[:, None]) * COLS + query_col)
        query = tl.load(
            q_ptr + ((q_base * HEADS + head) * D_HEAD + dims[None, :]),
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ) * SCALE_SQRT

        row_max = tl.full((BLOCK_I,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_I,), tl.float32)
        acc = tl.zeros((BLOCK_I, BLOCK_D), tl.float32)

        for key_outer in tl.range(0, K_OUTER):
            for key_start in tl.static_range(0, ROWS, BLOCK_K):
                key_row = key_start + key_rows
                key_mask = key_row < ROWS
                if HAS_KEY_VALID:
                    valid_base = (
                        (valid_batch * K_OUTER + key_outer) * ROWS + key_row
                    )
                    key_mask = key_mask & tl.load(
                        compressed_valid_ptr + valid_base,
                        mask=key_row < ROWS,
                        other=False,
                    )
                mk_base = (
                    ((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer) * COLS + query_col) * ROWS + key_row[:, None])
                )
                merged_key = tl.load(
                    merged_k_ptr + (((mk_base * HEADS + head) * D_HEAD) + dims[None, :]),
                    mask=key_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                merged_value = tl.load(
                    merged_v_ptr + (((mk_base * HEADS + head) * D_HEAD) + dims[None, :]),
                    mask=key_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )
                entropy = tl.load(
                    entropy_ptr
                    + (((((batch * HEADS + head) * Q_OUTER + query_outer) * K_OUTER + key_outer) * COLS + query_col) * ROWS + key_row),
                    mask=key_mask,
                    other=0.0,
                )

                logits = tl.dot(query, tl.trans(merged_key), input_precision=INPUT_PRECISION)
                logits = logits * 1.4426950408889634 - entropy[None, :]
                logits = tl.where(row_mask[:, None] & key_mask[None, :], logits, -float("inf"))

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
        out_base = (((batch * Q_OUTER + query_outer) * ROWS + rows[:, None]) * COLS + query_col)
        tl.store(
            out_ptr + ((out_base * HEADS + head) * D_HEAD + dims[None, :]),
            out,
            mask=row_mask[:, None] & d_mask[None, :],
        )


def monarch_rt_attention_triton_blocks(
    q, k, v, scale=None, output=None, key_valid=None
):
    """Run one-step MonarchRT on block-layout Q/K/V.

    Args:
        q, k, v: ``[B, outer, row, column, heads, dim]`` contiguous tensors.
        scale: usual attention scale. Defaults to ``dim ** -0.5``.
        output: optional output tensor with the same shape as ``q``.
        key_valid: optional ``[1|B, key_outer, row, column]`` boolean mask.
    """

    if triton is None:
        raise RuntimeError("Triton is required for monarch_rt_attention_triton_blocks.")
    if scale is None:
        scale = q.size(-1) ** -0.5
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, query_outer, rows, cols, heads, dim = q.shape
    key_outer = k.size(1)
    if key_valid is not None:
        expected_tail = (key_outer, rows, cols)
        if key_valid.ndim != 4 or tuple(key_valid.shape[1:]) != expected_tail:
            raise ValueError(
                "key_valid must have shape [1|B, key_outer, row, column], "
                f"got {tuple(key_valid.shape)} for tail {expected_tail}."
            )
        if key_valid.size(0) not in {1, batch}:
            raise ValueError("key_valid batch must be one or match q batch.")
        key_valid = key_valid.to(device=q.device, dtype=torch.bool).contiguous()
        compressed_valid = key_valid.any(dim=-1).contiguous()
        valid_batch = key_valid.size(0)
        has_key_valid = True
    else:
        key_valid = q
        compressed_valid = q
        valid_batch = 1
        has_key_valid = False
    block_d = triton.next_power_of_2(dim)
    scale_sqrt = float(scale) ** 0.5

    merged_shape = (batch, query_outer, key_outer, cols, rows, heads, dim)
    merged_k = torch.empty(merged_shape, device=q.device, dtype=q.dtype)
    merged_v = torch.empty(merged_shape, device=q.device, dtype=q.dtype)
    entropy = torch.empty(
        (batch, heads, query_outer, key_outer, cols, rows),
        device=q.device,
        dtype=torch.float32,
    )
    if output is None:
        output = torch.empty_like(q)

    narrow_columns = cols <= 32
    merge_block_j = 16 if narrow_columns else MONARCH_BLOCK_J
    merge_block_l = 16 if narrow_columns else MONARCH_BLOCK_L
    merge_j_tiles = triton.cdiv(cols, merge_block_j)
    merge_blocks = query_outer * key_outer * rows
    if narrow_columns:
        merge_grid = (merge_j_tiles * merge_blocks, batch * heads, 1)
    else:
        merge_grid = (merge_j_tiles, batch * heads, merge_blocks)
    # MI210 cannot launch the 4-warp specialization for narrow W4 blocks
    # (20 columns at 720p); the smaller launch also matches its useful tile.
    merge_num_warps = 1 if narrow_columns else MONARCH_NUM_WARPS
    merge_num_stages = 1 if narrow_columns else MONARCH_NUM_STAGES
    _merge_columns_kernel[merge_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        entropy,
        key_valid,
        compressed_valid,
        batch,
        valid_batch,
        query_outer,
        key_outer,
        rows,
        cols,
        heads,
        dim,
        BLOCK_J=merge_block_j,
        BLOCK_L=merge_block_l,
        BLOCK_D=block_d,
        J_TILES=merge_j_tiles,
        FLAT_GRID=narrow_columns,
        SCALE_SQRT=scale_sqrt,
        INPUT_PRECISION=MONARCH_INPUT_PRECISION,
        HAS_KEY_VALID=has_key_valid,
        num_warps=merge_num_warps,
        num_stages=merge_num_stages,
    )

    route_grid = (
        triton.cdiv(rows, MONARCH_BLOCK_I),
        batch * heads,
        query_outer * cols,
    )
    _route_rows_kernel[route_grid](
        q,
        merged_k,
        merged_v,
        entropy,
        compressed_valid,
        output,
        batch,
        valid_batch,
        query_outer,
        key_outer,
        rows,
        cols,
        heads,
        dim,
        BLOCK_I=MONARCH_BLOCK_I,
        BLOCK_K=MONARCH_BLOCK_K,
        BLOCK_D=block_d,
        SCALE_SQRT=scale_sqrt,
        INPUT_PRECISION=MONARCH_INPUT_PRECISION,
        HAS_KEY_VALID=has_key_valid,
        num_warps=MONARCH_NUM_WARPS,
        num_stages=MONARCH_NUM_STAGES,
    )
    return output


__all__ = [
    "MONARCH_RT_BACKEND",
    "MONARCH_BLOCK_I",
    "MONARCH_BLOCK_J",
    "MONARCH_BLOCK_K",
    "MONARCH_BLOCK_L",
    "monarch_rt_attention_triton_blocks",
]
