"""Experimental finer-grid Hard C4 for Wan2.1-14B 720p.

The physical latent grid is 45 x 80. Query rows are padded logically to 48,
so the h-reduced experiment layouts are 24 x 80 and 12 x 80. The 48 x 40
w-reduced layout is retained for comparison. Every valid grid is
compressed into four KV tokens, followed by full attention over all compressed
tokens visible to that query grid.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


ROWS = 45
COLS = 80
C = 4
QUERY_OUTER_GROUP = 21
BLOCK_J = 16
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@triton.jit
def _merge_grid_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    merged_k_ptr,
    merged_v_ptr,
    bias_ptr,
    Q_OUTER: tl.constexpr,
    Q_GROUPS: tl.constexpr,
    K_OUTER: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    ACTIVE_HEADS: tl.constexpr,
    HEAD_START: tl.constexpr,
    D_HEAD: tl.constexpr,
    QUERY_BLOCK_ROWS: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    KEY_BLOCK_COLS: tl.constexpr,
    KEY_BLOCKS: tl.constexpr,
    BLOCK_L: tl.constexpr,
    SCALE: tl.constexpr,
):
    block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % ACTIVE_HEADS
    global_head = HEAD_START + head
    batch = batch_head // ACTIVE_HEADS
    batch_i64 = batch.to(tl.int64)

    query_group = block % Q_GROUPS
    block //= Q_GROUPS
    query_block = block % QUERY_BLOCKS
    block //= QUERY_BLOCKS
    key_col_block = block % KEY_BLOCKS
    block //= KEY_BLOCKS
    key_outer = block % K_OUTER
    key_row = block // K_OUTER

    key_cols = tl.arange(0, BLOCK_L)
    dims = tl.arange(0, 128)
    key_col = key_col_block * KEY_BLOCK_COLS + key_cols
    key_mask = (key_cols < KEY_BLOCK_COLS) & (key_col < 80)
    key_base = (
        ((batch * TOTAL_HEADS + global_head) * K_OUTER + key_outer)
        * 45
        * 80
        * D_HEAD
        + key_row * 80 * D_HEAD
    )
    key_offsets = key_base + key_col[:, None] * D_HEAD + dims[None, :]
    key = tl.load(k_ptr + key_offsets, mask=key_mask[:, None], other=0.0)
    value = tl.load(v_ptr + key_offsets, mask=key_mask[:, None], other=0.0)

    flat = tl.arange(0, 64)
    seed_id = flat % 4
    query_start = query_block * QUERY_BLOCK_ROWS
    valid_query_rows = tl.minimum(QUERY_BLOCK_ROWS, 45 - query_start)
    local_key_row = tl.minimum(
        tl.maximum(key_row - query_start, 0), valid_query_rows - 1
    )
    seed_start = tl.minimum(
        tl.maximum(local_key_row - 1, 0), valid_query_rows - 4
    )
    seed_rows = query_start + seed_start + seed_id
    wide_offset = tl.where(
        seed_id == 0,
        -2,
        tl.where(seed_id == 1, 0, tl.where(seed_id == 2, 1, 3)),
    )
    use_wide = (local_key_row >= 2) & (local_key_row <= valid_query_rows - 4)
    seed_rows = tl.where(
        use_wide, query_start + local_key_row + wide_offset, seed_rows
    )

    for query_offset in tl.range(0, 21, num_stages=2):
        query_outer = query_group * 21 + query_offset
        q_base = (
            (batch * TOTAL_HEADS + global_head)
            * Q_OUTER
            * 45
            * 80
            * D_HEAD
            + query_outer * 45 * 80 * D_HEAD
        )
        for query_col_start in tl.range(0, 80, 16):
            query_cols = query_col_start + flat // 4
            seed_offsets = (
                q_base
                + (seed_rows[:, None] * 80 + query_cols[:, None]) * D_HEAD
                + dims[None, :]
            )
            seed_query = tl.load(q_ptr + seed_offsets)
            logits = tl.dot(seed_query, tl.trans(key), input_precision="ieee")
            logits = logits * (SCALE * 1.4426950408889634)
            logits = tl.where(key_mask[None, :], logits, -float("inf"))

            row_max = tl.max(logits, axis=1)
            row_sum = tl.sum(tl.exp2(logits - row_max[:, None]), axis=1)
            log_prob = tl.reshape(
                logits - (row_max + tl.log2(row_sum))[:, None],
                (16, 4, BLOCK_L),
            )
            winner, route = tl.max(
                log_prob,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            cluster = tl.arange(0, 4)[None, :, None]
            assigned = (route[:, None, :] == cluster) & key_mask[None, None, :]
            probability = tl.where(
                assigned, tl.exp2(winner)[:, None, :], 0.0
            )
            probability_sum = tl.sum(probability, axis=2)
            has_key = probability_sum > 0.0
            safe_sum = tl.where(has_key, probability_sum, 1.0)
            valid_winner = tl.where(key_mask[None, :], winner, 0.0)
            expected_log_prob = tl.sum(
                probability * valid_winner[:, None, :], axis=2
            ) / safe_sum
            weights = probability / safe_sum[:, :, None]
            route_bias = tl.where(
                has_key,
                (expected_log_prob - tl.log2(safe_sum)) * 0.6931471805599453,
                float("inf"),
            )

            weights = tl.reshape(weights, (64, BLOCK_L))
            merged_key = tl.dot(
                weights.to(key.dtype), key, input_precision="ieee"
            )
            merged_value = tl.dot(
                weights.to(value.dtype), value, input_precision="ieee"
            )

            token_base = (
                ((((((((batch_i64 * Q_OUTER + query_outer) * 80
                        + query_cols[:, None])
                       * ACTIVE_HEADS
                       + head)
                      * QUERY_BLOCKS
                      + query_block)
                     * K_OUTER
                     + key_outer)
                    * 45
                    + key_row)
                   * KEY_BLOCKS
                   + key_col_block)
                  * 4)
                + seed_id[:, None]
            )
            output_offsets = token_base * D_HEAD + dims[None, :]
            tl.store(merged_k_ptr + output_offsets, merged_key)
            tl.store(merged_v_ptr + output_offsets, merged_value)

            query_col_ids = query_col_start + tl.arange(0, 16)
            bias_seed = tl.arange(0, 4)
            bias_base = (
                ((((((((batch_i64 * Q_OUTER + query_outer) * 80
                        + query_col_ids[:, None])
                       * ACTIVE_HEADS
                       + head)
                      * QUERY_BLOCKS
                      + query_block)
                     * K_OUTER
                     + key_outer)
                    * 45
                    + key_row)
                   * KEY_BLOCKS
                   + key_col_block)
                  * 4)
                + bias_seed[None, :]
            )
            tl.store(bias_ptr + bias_base, route_bias)


@triton.jit
def _route_grid_kernel(
    q_ptr,
    merged_k_ptr,
    merged_v_ptr,
    bias_ptr,
    out_ptr,
    Q_OUTER: tl.constexpr,
    K_OUTER: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    ACTIVE_HEADS: tl.constexpr,
    HEAD_START: tl.constexpr,
    D_HEAD: tl.constexpr,
    QUERY_BLOCK_ROWS: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    KEY_BLOCKS: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    SCALE: tl.constexpr,
):
    batch_head = tl.program_id(1)
    query_group = tl.program_id(2)
    head = batch_head % ACTIVE_HEADS
    global_head = HEAD_START + head
    batch = batch_head // ACTIVE_HEADS
    batch_i64 = batch.to(tl.int64)

    query_block = query_group % QUERY_BLOCKS
    query_group //= QUERY_BLOCKS
    query_col = query_group % 80
    query_outer = query_group // 80
    local_rows = tl.arange(0, BLOCK_I)
    query_rows = query_block * QUERY_BLOCK_ROWS + local_rows
    query_mask = (local_rows < QUERY_BLOCK_ROWS) & (query_rows < 45)
    dims = tl.arange(0, 128)
    q_token = (query_outer * 45 + query_rows[:, None]) * 80 + query_col
    q_address = (
        ((batch * TOTAL_HEADS + global_head) * Q_OUTER * 45 * 80 + q_token)
        * D_HEAD
        + dims[None, :]
    )
    query = tl.load(q_ptr + q_address, mask=query_mask[:, None], other=0.0)

    total_tokens = K_OUTER * 45 * KEY_BLOCKS * 4
    route_base = (
        (((((batch_i64 * Q_OUTER + query_outer) * 80 + query_col)
            * ACTIVE_HEADS
            + head)
           * QUERY_BLOCKS
           + query_block)
          * total_tokens)
    )
    row_max = tl.full((BLOCK_I,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_I,), tl.float32)
    accumulator = tl.zeros((BLOCK_I, 128), tl.float32)
    token_offsets = tl.arange(0, BLOCK_TOKENS)

    for token_start in tl.range(
        0, total_tokens, BLOCK_TOKENS, num_stages=LOOP_STAGES
    ):
        tokens = token_start + token_offsets
        token_mask = tokens < total_tokens
        token_base = route_base + tokens[:, None]
        merged_key = tl.load(
            merged_k_ptr + token_base * D_HEAD + dims[None, :],
            mask=token_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        merged_value = tl.load(
            merged_v_ptr + token_base * D_HEAD + dims[None, :],
            mask=token_mask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        bias = tl.load(
            bias_ptr + route_base + tokens,
            mask=token_mask,
            other=float("inf"),
            cache_modifier=".cg",
        )
        logits = tl.dot(query, tl.trans(merged_key), input_precision="ieee")
        logits = (logits * SCALE - bias[None, :]) * 1.4426950408889634
        logits = tl.where(token_mask[None, :], logits, -float("inf"))
        next_max = tl.maximum(row_max, tl.max(logits, axis=1))
        probability = tl.exp2(logits - next_max[:, None])
        old_scale = tl.exp2(row_max - next_max)
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probability.to(query.dtype), merged_value, input_precision="ieee"
        )
        row_sum = row_sum * old_scale + tl.sum(probability, axis=1)
        row_max = next_max

    tl.store(
        out_ptr + q_address,
        accumulator / row_sum[:, None],
        mask=query_mask[:, None],
    )


def hard_c4_grid_720p_attention_triton_head_major(
    q,
    k,
    v,
    query_block_rows,
    key_block_cols,
    scale=None,
    output=None,
    head_start=0,
    head_count=None,
):
    """Run a 24x80, 48x40, or 48x80 Hard-C4 grid experiment."""

    scale = q.size(-1) ** -0.5 if scale is None else scale
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, total_heads, query_length, dim = q.shape
    query_outer = query_length // (ROWS * COLS)
    key_outer = k.size(2) // (ROWS * COLS)
    active_heads = int(head_count or total_heads)
    query_groups = query_outer // QUERY_OUTER_GROUP
    query_blocks = triton.cdiv(ROWS, query_block_rows)
    key_blocks = triton.cdiv(COLS, key_block_cols)
    block_l = triton.next_power_of_2(key_block_cols)
    block_i = triton.next_power_of_2(query_block_rows)

    tokens_per_query_block = key_outer * ROWS * key_blocks * C
    compressed_tokens = (
        batch
        * query_outer
        * COLS
        * active_heads
        * query_blocks
        * tokens_per_query_block
    )
    merged_k = torch.empty(
        (compressed_tokens, dim), device=q.device, dtype=q.dtype
    )
    merged_v = torch.empty_like(merged_k)
    bias = torch.empty(compressed_tokens, device=q.device, dtype=torch.float32)
    output = torch.empty_like(q) if output is None else output

    merge_grid = (
        query_groups * query_blocks * key_blocks * key_outer * ROWS,
        batch * active_heads,
    )
    _merge_grid_kernel[merge_grid](
        q,
        k,
        v,
        merged_k,
        merged_v,
        bias,
        query_outer,
        query_groups,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        query_block_rows,
        query_blocks,
        key_block_cols,
        key_blocks,
        block_l,
        SCALE=float(scale),
        num_warps=4,
        num_stages=2,
    )

    route_grid = (
        1,
        batch * active_heads,
        query_outer * COLS * query_blocks,
    )
    _route_grid_kernel[route_grid](
        q,
        merged_k,
        merged_v,
        bias,
        output,
        query_outer,
        key_outer,
        total_heads,
        active_heads,
        int(head_start),
        dim,
        query_block_rows,
        query_blocks,
        key_blocks,
        block_i,
        64,
        1 if torch.version.hip is not None else 3,
        SCALE=float(scale),
        num_warps=4,
        num_stages=1,
    )
    return output


def hard_c4_grid12x80_720p_attention_triton_head_major(q, k, v, **kwargs):
    return hard_c4_grid_720p_attention_triton_head_major(
        q, k, v, query_block_rows=12, key_block_cols=80, **kwargs
    )


def hard_c4_grid24x80_720p_attention_triton_head_major(q, k, v, **kwargs):
    return hard_c4_grid_720p_attention_triton_head_major(
        q, k, v, query_block_rows=24, key_block_cols=80, **kwargs
    )


def hard_c4_grid48x40_720p_attention_triton_head_major(q, k, v, **kwargs):
    return hard_c4_grid_720p_attention_triton_head_major(
        q, k, v, query_block_rows=48, key_block_cols=40, **kwargs
    )


def hard_c4_grid48x80_720p_attention_triton_head_major(q, k, v, **kwargs):
    return hard_c4_grid_720p_attention_triton_head_major(
        q, k, v, query_block_rows=48, key_block_cols=80, **kwargs
    )


__all__ = [
    "hard_c4_grid12x80_720p_attention_triton_head_major",
    "hard_c4_grid24x80_720p_attention_triton_head_major",
    "hard_c4_grid48x40_720p_attention_triton_head_major",
    "hard_c4_grid48x80_720p_attention_triton_head_major",
]
