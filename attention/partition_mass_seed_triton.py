"""Uniform-probe C4 selection by the missing variational partition mass.

The fixed row anchor is followed by three greedy selections. Each selection
recomputes the conditional distributions of the current hard cells and picks
the candidate with the largest globally weighted partition-mass deficit.
The selector uses the same C32/G45 bank and Uniform16 local QK products as
Hellinger-Span. A shared Uniform256 global normalizer supplies block mass.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _partition_mass_seed_kernel(
        q_ptr,
        k_ptr,
        norm_ptr,
        candidate_ptr,
        anchor_ptr,
        seed_row_ptr,
        K_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        N_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
        REUSE_OWNER_LOGP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        group = pid % N_GROUPS
        key_row = pid // N_GROUPS
        log2e: tl.constexpr = 1.4426950408889634

        dims = tl.arange(0, BLOCK_D)
        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        column_slot = flat // BLOCK_ROWS
        candidate_slot = flat % BLOCK_ROWS
        columns = group * BLOCK_J + column_slot
        candidate_rows = tl.load(
            candidate_ptr + key_row * CANDIDATES + candidate_slot,
            mask=candidate_slot < CANDIDATES,
            other=0,
        )
        valid = (columns < COLS) & (candidate_slot < CANDIDATES)
        query_offset = (
            ((batch * ROWS + candidate_rows[:, None]) * COLS + columns[:, None])
            * HEADS * D_HEAD + head * D_HEAD + dims[None, :]
        )
        query = tl.load(
            q_ptr + query_offset,
            mask=valid[:, None] & (dims[None, :] < D_HEAD),
            other=0.0,
        )
        norm_offset = (
            ((batch * ROWS + candidate_rows) * COLS + columns) * HEADS + head
        )
        global_norm = tl.load(norm_ptr + norm_offset, mask=valid, other=0.0)
        global_norm = tl.reshape(global_norm, (BLOCK_J, BLOCK_ROWS))
        candidate_rows_2d = tl.reshape(candidate_rows, (BLOCK_J, BLOCK_ROWS))
        valid_2d = tl.reshape(valid, (BLOCK_J, BLOCK_ROWS))
        candidate_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        anchor_index = tl.load(anchor_ptr + key_row)
        seed_slots = tl.arange(0, 4)
        probe_slots = tl.arange(0, 16)
        probe_columns = probe_slots * (COLS // 16)
        output_columns = group * BLOCK_J + tl.arange(0, BLOCK_J)

        for key_outer in tl.range(0, K_OUTER):
            key_base = (
                ((batch * K_OUTER + key_outer) * ROWS + key_row)
                * COLS * HEADS * D_HEAD
            )
            probe_keys = tl.load(
                k_ptr + key_base
                + (probe_columns[:, None] * HEADS + head) * D_HEAD
                + dims[None, :],
                mask=dims[None, :] < D_HEAD,
                other=0.0,
            )
            logits = tl.dot(query, tl.trans(probe_keys), input_precision="ieee")
            logits = logits * (SCALE * log2e)
            row_max = tl.max(logits, axis=1)
            log_mass = row_max + tl.log2(
                tl.sum(tl.exp2(logits - row_max[:, None]), axis=1)
            )
            log_probability = tl.reshape(
                logits - log_mass[:, None], (BLOCK_J, BLOCK_ROWS, 16)
            )
            mass = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS)) - global_norm
            maximum_mass = tl.max(tl.where(valid_2d, mass, -float("inf")), axis=1)
            maximum_mass = tl.where(output_columns < COLS, maximum_mass, 0.0)
            importance = tl.where(
                valid_2d, tl.exp2(mass - maximum_mass[:, None]), 0.0
            )

            used = candidate_ids == anchor_index
            anchor_row = tl.sum(tl.where(used, candidate_rows_2d, 0), axis=1)
            seeds = anchor_row[:, None] * (seed_slots[None, :] == 0).to(tl.int32)
            anchor_logp = tl.sum(
                tl.where(used[:, :, None], log_probability, 0.0), axis=1
            )
            if not REUSE_OWNER_LOGP:
                selected_logp = tl.where(
                    seed_slots[None, :, None] == 0, anchor_logp[:, None, :], 0.0
                )
            owner = tl.zeros((BLOCK_J, 16), dtype=tl.int32)
            owner_logp = anchor_logp

            for step in tl.static_range(1, 4):
                represented_mass = tl.zeros((BLOCK_J, BLOCK_ROWS), tl.float32)
                for cell_slot in tl.static_range(0, step):
                    if REUSE_OWNER_LOGP:
                        # On this cell, the winning logp is exactly this seed's
                        # logp. Values outside the cell are always masked, so
                        # storing/gathering all four seed profiles is redundant.
                        seed_logp = owner_logp
                    else:
                        seed_logp = tl.sum(
                            tl.where(
                                seed_slots[None, :, None] == cell_slot,
                                selected_logp,
                                0.0,
                            ),
                            axis=1,
                        )
                    cell = owner == cell_slot
                    active = tl.sum(cell.to(tl.int32), axis=1) > 0
                    cell_max = tl.max(
                        tl.where(cell, seed_logp, -float("inf")), axis=1
                    )
                    cell_max = tl.where(active, cell_max, 0.0)
                    exponential = tl.where(
                        cell, tl.exp2(seed_logp - cell_max[:, None]), 0.0
                    )
                    denominator = tl.sum(exponential, axis=1)
                    safe_denominator = tl.where(active, denominator, 1.0)
                    weight = exponential / safe_denominator[:, None]
                    log_weight = tl.where(
                        cell,
                        seed_logp - cell_max[:, None]
                        - tl.log2(safe_denominator)[:, None],
                        0.0,
                    )
                    entropy = -tl.sum(weight * log_weight, axis=1)
                    bound = tl.sum(
                        log_probability * weight[:, None, :], axis=2
                    ) + entropy[:, None]
                    # exp2(bound) is the cell's represented target mass.
                    # Summing it computes exp2(-gap) without log/exp roundtrips.
                    represented_mass += tl.where(active[:, None], tl.exp2(bound), 0.0)

                deficit = tl.maximum(1.0 - represented_mass, 0.0)
                score = tl.where(valid_2d & ~used, importance * deficit, -1.0)
                _, pick = tl.max(
                    score,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit = candidate_ids == pick[:, None]
                used = used | hit
                row = tl.sum(tl.where(hit, candidate_rows_2d, 0), axis=1)
                seeds += row[:, None] * (seed_slots[None, :] == step).to(tl.int32)
                if step < 3:
                    new_logp = tl.sum(
                        tl.where(hit[:, :, None], log_probability, 0.0), axis=1
                    )
                    if not REUSE_OWNER_LOGP:
                        selected_logp = tl.where(
                            seed_slots[None, :, None] == step,
                            new_logp[:, None, :],
                            selected_logp,
                        )
                    # Strict comparison preserves first-seed ownership on ties.
                    new_owner = new_logp > owner_logp
                    owner = tl.where(new_owner, step, owner)
                    owner_logp = tl.maximum(owner_logp, new_logp)

            output_offset = (
                (((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS
                 + output_columns[:, None]) * HEADS + head
            ) * 4 + seed_slots[None, :]
            tl.store(
                seed_row_ptr + output_offset,
                seeds,
                mask=output_columns[:, None] < COLS,
            )


def partition_mass_seed_rows_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float | None = None,
    *,
    candidate_count: int = 32,
    candidate_geometry_count: int = 45,
    block_j: int = 4,
    probe_count: int = 16,
    global_probe_count: int = 256,
    output_dtype: torch.dtype = torch.uint8,
    gs_num_warps: int = 8,
    num_warps: int | None = None,
    num_stages: int = 1,
    global_normalizer: torch.Tensor | None = None,
    reuse_owner_logp: bool = False,
) -> torch.Tensor:
    """Return native C4 rows [B,1,key_frames,rows,columns,heads,4].

    ``q`` is [B,1,rows,columns,heads,dim], ``k`` is
    [B,key_frames,rows,columns,heads,dim]. The optional shared global
    normalizer must be contiguous float32 [B,1,rows,columns,heads] containing
    base2 logsumexp from ``global_probe_log_norm_triton``. Omitting it runs
    that normalizer once. No local probe-index workspace is allocated.

    ``block_j``, ``gs_num_warps``, and ``num_stages`` are execution settings;
    they do not change the mathematical selector. The algorithm has no
    fitted weights or additional importance-temperature hyperparameter.
    ``num_warps`` is a compatibility alias overriding ``gs_num_warps``.
    ``reuse_owner_logp`` opts into an algebraically identical implementation
    that reuses each probe's winning log probability for its owned cell,
    avoiding the separate four-seed profile buffer and masked gathers.
    The default path is unchanged; GPU parity and speed must be measured.
    """
    if triton is None:
        raise RuntimeError("Triton is required for partition mass seed selection")
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q and k must be [batch, frames, rows, columns, heads, dim]")
    batch, query_frames, rows, columns, heads, head_dim = q.shape
    if query_frames != 1:
        raise ValueError("the shared-reference selector requires one query frame")
    if k.shape[0] != batch or k.shape[2:] != q.shape[2:] or k.size(1) < 1:
        raise ValueError("q and k must have matching batch and spatial/head shapes")
    if not q.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("q and k must share float16, bfloat16, or float32 dtype")
    if columns < 16:
        raise ValueError("Uniform16 requires at least 16 key columns")
    if probe_count != 16:
        raise ValueError("the partition mass selector requires probe_count=16")
    if not 4 <= candidate_count <= candidate_geometry_count <= rows:
        raise ValueError("require 4 <= candidate_count <= candidate_geometry_count <= rows")
    if output_dtype not in (torch.uint8, torch.int32):
        raise TypeError("seed output must be uint8 or int32")
    if output_dtype == torch.uint8 and rows > 256:
        raise ValueError("uint8 cannot represent this many row indices")
    if block_j not in (1, 2, 4, 8):
        raise ValueError("block_j must be 1, 2, 4, or 8")
    effective_num_warps = gs_num_warps if num_warps is None else num_warps
    if effective_num_warps not in (4, 8):
        raise ValueError("gs_num_warps must be 4 or 8")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    if not isinstance(reuse_owner_logp, bool):
        raise TypeError("reuse_owner_logp must be bool")
    if scale is None:
        scale = head_dim**-0.5
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")

    from .adaptive_seed_pytorch import multiscale_candidate_table
    from .entropy_mass_seed_triton import global_probe_log_norm_triton

    q, k = q.contiguous(), k.contiguous()
    if global_normalizer is None:
        global_normalizer = global_probe_log_norm_triton(
            q, k, scale=scale, global_probe_count=global_probe_count
        )
    expected_norm_shape = (batch, 1, rows, columns, heads)
    if (
        tuple(global_normalizer.shape) != expected_norm_shape
        or global_normalizer.dtype != torch.float32
        or global_normalizer.device != q.device
        or not global_normalizer.is_contiguous()
    ):
        raise ValueError("global_normalizer must be contiguous FP32 [B,1,R,C,H] on q.device")
    candidates, anchors = multiscale_candidate_table(
        rows, candidate_count, q.device, geometry_count=candidate_geometry_count
    )
    output = torch.empty(
        (batch, 1, k.size(1), rows, columns, heads, 4),
        device=q.device,
        dtype=output_dtype,
    )
    groups = triton.cdiv(columns, block_j)
    _partition_mass_seed_kernel[(rows * groups, batch * heads)](
        q,
        k,
        global_normalizer,
        candidates,
        anchors,
        output,
        K_OUTER=k.size(1),
        ROWS=rows,
        COLS=columns,
        HEADS=heads,
        D_HEAD=head_dim,
        CANDIDATES=candidate_count,
        BLOCK_J=block_j,
        BLOCK_ROWS=max(16, triton.next_power_of_2(candidate_count)),
        N_GROUPS=groups,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        SCALE=float(scale),
        REUSE_OWNER_LOGP=reuse_owner_logp,
        num_warps=effective_num_warps,
        num_stages=num_stages,
    )
    return output


__all__ = ["partition_mass_seed_rows_triton"]
