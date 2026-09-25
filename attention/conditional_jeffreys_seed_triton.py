"""Uniform16/G256 conditional-Jeffreys C4 real-query seed selection.

This isolated candidate matches the CPU jeffreys_p16_g256 criterion, not
no_sq_score. Full16 normalized seed probabilities determine ownership
BEFORE an observed SQ1 column is removed from its bulk cell. Singleton
Jeffreys contribution is zero. The fixed anchor precedes three weighted
farthest pivots; adding seeds does not guarantee lower loss.

Gamma is evaluated in bits inside the kernel; multiplying every score by
ln(2) would recover the reference's nats without changing its exact ranking.
FP32 cancellation at tiny Gamma and near ties can change pivots relative to
the FP64 diagnostic. Native parity and timing require separate GPU checks.
No merge, routing, production dispatch, or default is changed by this file.
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


INTERMEDIATE_AUDIT_PLANES = (
    "qk_dot", "logits_bits", "centered_logits_bits", "log_probability_bits",
    "probability", "importance", "local_log_mass_bits",
    "owner_prefix1", "owner_prefix2", "owner_prefix3",
    "owner_logp_prefix1", "owner_logp_prefix2", "owner_logp_prefix3",
    "gamma_bits_prefix1", "gamma_bits_prefix2", "gamma_bits_prefix3",
    "priority_prefix1", "priority_prefix2", "priority_prefix3",
)


if triton is not None:

    @triton.jit
    def _conditional_jeffreys_seed_kernel(
        q_ptr,
        k_ptr,
        norm_ptr,
        candidate_ptr,
        anchor_ptr,
        seed_row_ptr,
        score_audit_ptr,
        intermediate_audit_ptr,
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
        AUDIT_SCORES: tl.constexpr,
        AUDIT_INTERMEDIATES: tl.constexpr,
        AUDIT_BATCH: tl.constexpr,
        AUDIT_FRAME: tl.constexpr,
        AUDIT_ROW: tl.constexpr,
        AUDIT_COLUMN: tl.constexpr,
        AUDIT_HEAD: tl.constexpr,
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
        if AUDIT_INTERMEDIATES:
            # tl.store broadcasts values/masks TO the pointer shape, not the
            # pointer to their larger J dimension. Only one J lane is enabled.
            intermediate_offset = (
                candidate_ids[:, :, None] * 16 + probe_slots[None, None, :]
                + tl.zeros((BLOCK_J, 1, 1), dtype=tl.int32)
            )
            intermediate_bank = (batch == AUDIT_BATCH) & (head == AUDIT_HEAD) & (key_row == AUDIT_ROW)
            intermediate_column = (output_columns == AUDIT_COLUMN)[:, None, None]

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
            if AUDIT_INTERMEDIATES:
                intermediate_mask = (
                    intermediate_bank & (key_outer == AUDIT_FRAME)
                    & intermediate_column & valid_2d[:, :, None]
                )
                tl.store(
                    intermediate_audit_ptr + intermediate_offset,
                    tl.reshape(logits, (BLOCK_J, BLOCK_ROWS, 16)),
                    mask=intermediate_mask,
                )
            logits = logits * (SCALE * log2e)
            row_max = tl.max(logits, axis=1)
            # Reuse the SAME centered tensor for normalization and logp.
            # Subtracting a rounded O(500)-bit absolute log_mass from logits
            # can lose a small conditional log probability on high-offset QK.
            centered_logits = logits - row_max[:, None]
            centered_log_mass = tl.log2(
                tl.sum(tl.exp2(centered_logits), axis=1)
            )
            log_mass = row_max + centered_log_mass
            log_probability = tl.reshape(
                centered_logits - centered_log_mass[:, None],
                (BLOCK_J, BLOCK_ROWS, 16),
            )
            probability = tl.exp2(log_probability)
            mass = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS)) - global_norm
            maximum_mass = tl.max(tl.where(valid_2d, mass, -float("inf")), axis=1)
            maximum_mass = tl.where(output_columns < COLS, maximum_mass, 0.0)
            importance = tl.where(
                valid_2d, tl.exp2(mass - maximum_mass[:, None]), 0.0
            )
            if AUDIT_INTERMEDIATES:
                tl.store(intermediate_audit_ptr + 1 * (32 * 16) + intermediate_offset,
                         tl.reshape(logits, (BLOCK_J, BLOCK_ROWS, 16)), mask=intermediate_mask)
                tl.store(intermediate_audit_ptr + 2 * (32 * 16) + intermediate_offset,
                         tl.reshape(centered_logits, (BLOCK_J, BLOCK_ROWS, 16)), mask=intermediate_mask)
                tl.store(intermediate_audit_ptr + 3 * (32 * 16) + intermediate_offset,
                         log_probability, mask=intermediate_mask)
                tl.store(intermediate_audit_ptr + 4 * (32 * 16) + intermediate_offset,
                         probability, mask=intermediate_mask)
                tl.store(intermediate_audit_ptr + 5 * (32 * 16) + intermediate_offset,
                         importance[:, :, None], mask=intermediate_mask)
                tl.store(intermediate_audit_ptr + 6 * (32 * 16) + intermediate_offset,
                         tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS))[:, :, None], mask=intermediate_mask)

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
                if AUDIT_INTERMEDIATES:
                    tl.store(intermediate_audit_ptr + (6 + step) * (32 * 16) + intermediate_offset,
                             owner[:, None, :], mask=intermediate_mask)
                    tl.store(intermediate_audit_ptr + (9 + step) * (32 * 16) + intermediate_offset,
                             owner_logp[:, None, :], mask=intermediate_mask)
                gamma = tl.zeros((BLOCK_J, BLOCK_ROWS), tl.float32)
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
                    # Ownership is already computed on normalized full16
                    # profiles. Peel SQ1 only here, never before softmax or
                    # owner updates. An unobserved SQ1 removes no probe.
                    cell = (owner == cell_slot) & (
                        probe_columns[None, :] != output_columns[:, None]
                    )
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
                    target_in_cell = tl.where(
                        cell[:, None, :], probability, 0.0
                    )
                    target_mass = tl.sum(target_in_cell, axis=2)
                    # Sum_c [sum_j p_ij 1_c (log p_ij - log w_cj)
                    #        - P_ic (sum_j w_cj log p_ij + H_c)].
                    # Combine before reduction: sum_j (p_ij 1_c - P_ic w_cj)
                    # * (log p_ij - log w_cj). This is algebraically identical
                    # and avoids subtracting separately reduced O(1) terms.
                    # Empty bulk cells and the peeled singleton contribute 0.
                    probability_residual = (
                        target_in_cell - target_mass[:, :, None] * weight[:, None, :]
                    )
                    log_ratio = log_probability - log_weight[:, None, :]
                    gamma += tl.where(
                        active[:, None],
                        tl.sum(probability_residual * log_ratio, axis=2),
                        0.0,
                    )

                priority = importance * tl.maximum(gamma, 0.0)
                if AUDIT_INTERMEDIATES:
                    tl.store(intermediate_audit_ptr + (12 + step) * (32 * 16) + intermediate_offset,
                             gamma[:, :, None], mask=intermediate_mask)
                    tl.store(intermediate_audit_ptr + (15 + step) * (32 * 16) + intermediate_offset,
                             priority[:, :, None], mask=intermediate_mask)
                if AUDIT_SCORES:
                    audit_offset = (
                        (((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS
                         + output_columns[:, None]) * HEADS + head
                    ) * (3 * CANDIDATES) + (step - 1) * CANDIDATES + candidate_ids
                    tl.store(score_audit_ptr + audit_offset, priority, mask=valid_2d)
                score = tl.where(valid_2d & ~used, priority, -1.0)
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


def conditional_jeffreys_seed_rows_triton(
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
    gs_num_warps: int = 4,
    num_warps: int | None = None,
    num_stages: int = 1,
    global_normalizer: torch.Tensor | None = None,
    reuse_owner_logp: bool = True,
    score_audit: torch.Tensor | None = None,
    intermediate_audit: torch.Tensor | None = None,
    intermediate_audit_bank: tuple[int, int, int, int, int] | None = None,
) -> torch.Tensor:
    """Return uint8 C4 rows [B,1,key_frames,45,80,heads,4] by default.

    Inputs are q=[B,1,45,80,H,D] (the caller's fixed RF0 query slice) and
    k=[B,key_frames,45,80,H,D]. Candidate geometry is exactly C32/g45;
    local probes are columns 0,5,...,75 and the global sketch is Uniform256.
    The optional normalizer is the original contiguous FP32 base2 logsumexp
    [B,1,45,80,H] from global_probe_log_norm_triton. This selector does not
    select or replace the caller's reference frame.

    J4/four warps are defaults; block_j, num_warps/gs_num_warps, num_stages,
    and reuse_owner_logp only select algebraically equivalent execution.
    No fitted parameter, importance temperature, or exact-Z oracle is used.

    An optional preallocated score_audit is contiguous FP32
    [B,1,key_frames,45,80,H,3,32] on q.device. It records importance times
    max(Gamma_bits, 0) BEFORE used-candidate masking, for the three pivots.
    It can expose tiny-score/near-tie discrepancies without altering scoring.
    Omit it for timing: auditing adds stores and is not the production path.

    Optional intermediate_audit is contiguous FP32 [19,32,16], observing just
    intermediate_audit_bank=(batch,key_frame,key_row,query_column,head_slot).
    Plane names are INTERMEDIATE_AUDIT_PLANES: 0:5 are full candidate/probe
    matrices; planes5:7 and13:19 repeat row scalars across probes; planes7:13
    repeat probe ownership/profile vectors across candidates. Ownership is
    recorded BEFORE singleton removal, at each of the three actual prefixes.
    These are stores of this kernel's actual intermediates, not recomputation.
    Observational stores may affect compiler lowering: compare debug-on/off
    rows AND scores before attributing an observed intermediate to a failure.
    Never enable this buffer for timing. Omitting it compiles all stores out.
    """
    if triton is None:
        raise RuntimeError("Triton is required for conditional Jeffreys seed selection")
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
    if rows != 45 or columns != 80 or min(batch, heads, head_dim) < 1:
        raise ValueError("the native candidate requires nonempty [B,1,45,80,H,D]")
    if probe_count != 16 or global_probe_count != 256:
        raise ValueError("the candidate requires Uniform16 local / Uniform256 global probes")
    if candidate_count != 32 or candidate_geometry_count != 45:
        raise ValueError("the candidate requires candidate_count=32, candidate_geometry_count=45")
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
    expected_audit_shape = (batch, 1, k.size(1), rows, columns, heads, 3, candidate_count)
    if score_audit is not None and (
        tuple(score_audit.shape) != expected_audit_shape
        or score_audit.dtype != torch.float32
        or score_audit.device != q.device
        or not score_audit.is_contiguous()
    ):
        raise ValueError("score_audit must be contiguous FP32 [B,1,F,45,80,H,3,32] on q.device")
    if intermediate_audit is None:
        if intermediate_audit_bank is not None:
            raise ValueError("intermediate_audit_bank requires an intermediate_audit buffer")
        audit_bank = (0, 0, 0, 0, 0)
    else:
        if (
            tuple(intermediate_audit.shape) != (19, 32, 16)
            or intermediate_audit.dtype != torch.float32
            or intermediate_audit.device != q.device
            or not intermediate_audit.is_contiguous()
        ):
            raise ValueError("intermediate_audit must be contiguous FP32 [19,32,16] on q.device")
        if not isinstance(intermediate_audit_bank, tuple) or len(intermediate_audit_bank) != 5:
            raise ValueError("intermediate_audit_bank must be (batch,key_frame,key_row,query_column,head_slot)")
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   and 0 <= value < limit for value, limit in zip(
                       intermediate_audit_bank, (batch, k.size(1), rows, columns, heads))):
            raise ValueError("intermediate_audit_bank coordinates are out of range or non-integer")
        audit_bank = intermediate_audit_bank
    candidates, anchors = multiscale_candidate_table(
        rows, candidate_count, q.device, geometry_count=candidate_geometry_count
    )
    output = torch.empty(
        (batch, 1, k.size(1), rows, columns, heads, 4),
        device=q.device,
        dtype=output_dtype,
    )
    groups = triton.cdiv(columns, block_j)
    _conditional_jeffreys_seed_kernel[(rows * groups, batch * heads)](
        q,
        k,
        global_normalizer,
        candidates,
        anchors,
        output,
        output if score_audit is None else score_audit,
        output if intermediate_audit is None else intermediate_audit,
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
        AUDIT_SCORES=score_audit is not None,
        AUDIT_INTERMEDIATES=intermediate_audit is not None,
        AUDIT_BATCH=audit_bank[0],
        AUDIT_FRAME=audit_bank[1],
        AUDIT_ROW=audit_bank[2],
        AUDIT_COLUMN=audit_bank[3],
        AUDIT_HEAD=audit_bank[4],
        num_warps=effective_num_warps,
        num_stages=num_stages,
    )
    return output


__all__ = ["conditional_jeffreys_seed_rows_triton", "INTERMEDIATE_AUDIT_PLANES"]
