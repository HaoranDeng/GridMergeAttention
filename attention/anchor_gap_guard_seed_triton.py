"""Isolated Anchor32 incumbent + last-slot quadrature reverse-gap guard.

This is explicitly TWO stages: the unchanged real Anchor-Fisher J4 selector,
then an independent J1 correction kernel. The second stage repeats U16 QK
and observes SQ, using a 32-column dot tile padded around the 16/17 valid
support bins. It is not an all-J4 implementation or a claim of only 6.25%
additional total QK. No native GPU parity, speed or quality is claimed here.

Only the last of four real rows can change. A single challenger maximizes
the current prefix3 reverse gap, excluding only those first three rows.
Accept it only if the uniform mean gap is strictly lower than incumbent4.
This guarantees only the tested weighted-support objective, not full80 hard
partitions, future frames, output error or video quality. No learning, G256
normalizer, mass importance, dense attention, merge or route modification.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _gap_guard_log_probabilities(z, valid, singleton):
        """Natural weighted-bin log probabilities, all arithmetic FP64.

        Full support includes SQ for normalization/ownership. Valid bins are
        U16, plus SQ only when it is absent there; no duplicate singleton.
        Padding receives finite zero logp and is excluded from all cells.
        """
        probes = tl.arange(0, 32)
        bulk = valid & (probes != singleton)
        one = tl.full((32,), 1.0, tl.float64)
        bulk_count = tl.sum(tl.where(bulk, one, 0.0), axis=0)
        bulk_measure = tl.full((), 79.0, tl.float64) / bulk_count
        measure = tl.where(probes == singleton, 1.0, bulk_measure)
        maximum = tl.max(tl.where(valid[None, :], z, -float("inf")), axis=1)
        centered = z - maximum[:, None]
        weighted = centered + libdevice.log(measure)[None, :]
        weighted_max = tl.max(tl.where(valid[None, :], weighted, -float("inf")), axis=1)
        shifted = tl.where(valid[None, :], weighted - weighted_max[:, None], -float("inf"))
        denominator = tl.sum(libdevice.exp(shifted), axis=1)
        return tl.where(valid[None, :], shifted - libdevice.log(denominator)[:, None], 0.0)

    @triton.jit
    def _gap_guard_reverse_gap(logp, valid, singleton, selected, COUNT: tl.constexpr):
        """Exact restricted reverse-KL gap on the supplied weighted support.

        Compute normalized-profile ownership on FULL support before peeling
        SQ. Lowest seed slot wins ties; empty bulk cells have zero represented
        mass. logp already includes log(mu), so bin entropy is appropriate.
        Preserve tiny roundoff-negative gaps, never clip the ranking or mean.
        """
        rows = tl.arange(0, 32)
        probes = tl.arange(0, 32)
        slots = tl.arange(0, 4)
        components = tl.arange(0, 8)
        owner = tl.full((32,), -1, tl.int32)
        owner_logp = tl.full((32,), -float("inf"), tl.float64)
        for cell_slot in tl.static_range(0, COUNT):
            candidate = tl.sum(tl.where(slots == cell_slot, selected, 0), axis=0)
            seed_logp = tl.sum(tl.where(rows[:, None] == candidate, logp, 0.0), axis=0)
            replace = seed_logp > owner_logp
            owner = tl.where(replace, cell_slot, owner)
            owner_logp = tl.where(replace, seed_logp, owner_logp)

        singleton_logp = tl.sum(tl.where(probes[None, :] == singleton, logp, 0.0), axis=1)
        component_log_mass = tl.where(components[None, :] == COUNT,
                                      singleton_logp[:, None], -float("inf"))
        for cell_slot in tl.static_range(0, COUNT):
            candidate = tl.sum(tl.where(slots == cell_slot, selected, 0), axis=0)
            seed_logp = tl.sum(tl.where(rows[:, None] == candidate, logp, 0.0), axis=0)
            cell = valid & (probes != singleton) & (owner == cell_slot)
            active = tl.sum(tl.where(cell, 1, 0), axis=0) > 0
            cell_max = tl.max(tl.where(cell, seed_logp, -float("inf")), axis=0)
            cell_max = tl.where(active, cell_max, 0.0)
            exponential = libdevice.exp(tl.where(cell, seed_logp - cell_max, -float("inf")))
            denominator = tl.sum(exponential, axis=0)
            denominator = tl.where(active, denominator, 1.0)
            weight = exponential / denominator
            log_weight = tl.where(cell, seed_logp - cell_max - libdevice.log(denominator), 0.0)
            entropy = -tl.sum(weight * log_weight, axis=0)
            bound = tl.sum(logp * weight[None, :], axis=1) + entropy
            bound = tl.where(active, bound, -float("inf"))
            component_log_mass = tl.where(components[None, :] == cell_slot,
                                          bound[:, None], component_log_mass)
        maximum = tl.max(component_log_mass, axis=1)
        log_represented = maximum + libdevice.log(tl.sum(
            libdevice.exp(component_log_mass - maximum[:, None]), axis=1))
        return -log_represented

    @triton.jit
    def _gap_guard_last_slot(z, valid, singleton, incumbent):
        rows = tl.arange(0, 32)
        slots = tl.arange(0, 4)
        logp = _gap_guard_log_probabilities(z, valid, singleton)
        prefix_gap = _gap_guard_reverse_gap(logp, valid, singleton, incumbent, 3)
        used = tl.sum(tl.where((rows[:, None] == incumbent[None, :]) & (slots[None, :] < 3),
                               1, 0), axis=1) > 0
        _, challenger = tl.max(tl.where(~used, prefix_gap, -float("inf")), axis=0,
                               return_indices=True, return_indices_tie_break_left=True)
        trial = tl.where(slots == 3, challenger, incumbent)
        base_gap = _gap_guard_reverse_gap(logp, valid, singleton, incumbent, 4)
        trial_gap = _gap_guard_reverse_gap(logp, valid, singleton, trial, 4)
        old_last = tl.sum(tl.where(slots == 3, incumbent, 0), axis=0)
        # Identical ordered tuples cannot be a strict improvement. Separate
        # floating reductions must not turn this exact identity into a move.
        accept = (challenger != old_last) & (
            tl.sum(trial_gap, axis=0) / 32.0 < tl.sum(base_gap, axis=0) / 32.0)
        corrected = tl.where((slots == 3) & accept, challenger, incumbent)
        return corrected, prefix_gap, base_gap, trial_gap, challenger, accept

    @triton.jit
    def _anchor_gap_guard_correction_kernel(
        q_ptr, k_ptr, candidate_ptr, seed_ptr, gap_ptr, decision_ptr,
        K_OUTER: tl.constexpr, HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        BLOCK_D: tl.constexpr, SCALE: tl.constexpr,
        AUDIT_GAPS: tl.constexpr, AUDIT_DECISION: tl.constexpr,
    ):
        # True correction J1, after the unchanged true Anchor-Fisher J4 stage.
        bank = tl.program_id(0)
        batch_head = tl.program_id(1)
        column, key_row = bank % 80, bank // 80
        head = batch_head % HEADS
        batch = (batch_head // HEADS).to(tl.int64)
        rows = tl.arange(0, 32)
        dims = tl.arange(0, BLOCK_D)
        probes = tl.arange(0, 32)
        seed_slots = tl.arange(0, 4)
        candidates = tl.load(candidate_ptr + key_row * 32 + rows)
        duplicate = column % 5 == 0
        singleton = tl.where(duplicate, column // 5, 16)
        valid = (probes < 16) | ((probes == 16) & ~duplicate)
        support_columns = tl.where(probes < 16, probes * 5, column)
        query_tokens = (batch * 45 + candidates) * 80 + column
        queries = tl.load(q_ptr + (query_tokens[:, None] * HEADS + head) * D_HEAD + dims[None, :],
                          mask=dims[None, :] < D_HEAD, other=0.0)
        for key_frame in tl.range(0, K_OUTER):
            key_base = ((batch * K_OUTER + key_frame) * 45 + key_row) * 80
            keys = tl.load(k_ptr + ((key_base + support_columns[:, None]) * HEADS + head) * D_HEAD
                           + dims[None, :], mask=valid[:, None] & (dims[None, :] < D_HEAD), other=0.0)
            accumulated = tl.dot(queries, tl.trans(keys), input_precision="ieee", out_dtype=tl.float32)
            z = accumulated.to(tl.float64) * SCALE
            output_bank = (((batch * K_OUTER + key_frame) * 45 + key_row) * 80 + column) * HEADS + head
            incumbent_rows = tl.load(seed_ptr + output_bank * 4 + seed_slots)
            incumbent = tl.sum(tl.where(candidates[:, None] == incumbent_rows[None, :],
                                        rows[:, None], 0), axis=0)
            corrected, prefix_gap, base_gap, trial_gap, challenger, accept = _gap_guard_last_slot(
                z, valid, singleton, incumbent)
            last_slot = tl.sum(tl.where(seed_slots == 3, corrected, 0), axis=0)
            last_row = tl.sum(tl.where(rows == last_slot, candidates, 0), axis=0)
            # No writes to slots zero/one/two: preserve the actual incumbent.
            tl.store(seed_ptr + output_bank * 4 + 3, last_row)
            if AUDIT_GAPS:
                tl.store(gap_ptr + (output_bank * 3 + 0) * 32 + rows, prefix_gap)
                tl.store(gap_ptr + (output_bank * 3 + 1) * 32 + rows, base_gap)
                tl.store(gap_ptr + (output_bank * 3 + 2) * 32 + rows, trial_gap)
            if AUDIT_DECISION:
                old_last = tl.sum(tl.where(seed_slots == 3, incumbent, 0), axis=0)
                tl.store(decision_ptr + output_bank * 3 + 0, challenger)
                tl.store(decision_ptr + output_bank * 3 + 1, accept.to(tl.int32))
                tl.store(decision_ptr + output_bank * 3 + 2, old_last)


def anchor_gap_guard_seed_rows_triton(
    q, k, scale=None, *, candidate_count=32, candidate_geometry_count=45,
    block_j=4, probe_count=16, global_probe_count=256,
    output_dtype=torch.uint8, gs_num_warps=4, gap_audit=None, decision_audit=None,
):
    """Original Anchor32 J4 incumbent, then J1 last-slot correction in place.

    q=[B,1,45,80,H,D] must be the original RF0 slice, k=[B,F,45,80,H,D].
    Require original BF16, C32/g45/U16, actual incumbent J4/four warps.
    global_probe_count=256 is compatibility metadata only: no global sketch.
    Return four real rows [B,1,F,45,80,H,4]; merge must reload current-frame Q.

    Optional contiguous on-device gap_audit is FP64 [B,1,F,45,80,H,3,32],
    storing raw per-target prefix3/base4/trial4 gaps in natural units.
    decision_audit is int32 [B,1,F,45,80,H,3], storing challenger candidate
    SLOT, accepted 0/1, incumbent last SLOT. Either may be independently
    omitted. They are caller-allocated diagnostics; default allocates neither.
    Finite non-overflowing QK is a caller precondition. No additional input,
    global estimate, alternate precision, learned parameter or soft seed is
    accepted. This wrapper does not execute any merge, route or dense oracle.
    """
    if (type(candidate_count) is not int or candidate_count != 32
            or type(candidate_geometry_count) is not int or candidate_geometry_count != 45):
        raise ValueError("gap guard requires the actual C32/g45 incumbent")
    if type(block_j) is not int or block_j != 4:
        raise ValueError("gap guard requires true Anchor-Fisher J4 followed by correction J1")
    if type(probe_count) is not int or probe_count != 16:
        raise ValueError("gap guard requires Uniform16 plus observed SQ")
    if type(global_probe_count) is not int or global_probe_count != 256:
        raise ValueError("unexpected dispatch global-probe field; no global sketch is used")
    if type(gs_num_warps) is not int or gs_num_warps != 4:
        raise ValueError("gap guard preserves the four-warp Anchor32 incumbent")
    if q.ndim != 6 or k.ndim != 6 or min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("require nonempty [B,F,45,80,H,D] Q/K")
    batch, query_frames, rows, columns, heads, dim = q.shape
    if query_frames != 1 or (rows, columns) != (45, 80) or dim > 128:
        raise ValueError("q must be a single RF0 slice [B,1,45,80,H,D<=128]")
    if k.shape[0] != batch or k.shape[2:] != q.shape[2:]:
        raise ValueError("Q/K must share batch/grid/head/dimension")
    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("Q/K must share a CUDA device")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("gap guard requires original BF16 Q/K")
    if output_dtype not in (torch.uint8, torch.int32):
        raise TypeError("row output must be uint8 or int32")
    scale = dim ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    bank_shape = (batch, 1, k.shape[1], 45, 80, heads)
    if gap_audit is not None and (
        tuple(gap_audit.shape) != (*bank_shape, 3, 32) or gap_audit.dtype != torch.float64
        or gap_audit.device != q.device or not gap_audit.is_contiguous()
    ):
        raise ValueError("gap_audit must be contiguous on-device FP64 [B,1,F,45,80,H,3,32]")
    if decision_audit is not None and (
        tuple(decision_audit.shape) != (*bank_shape, 3) or decision_audit.dtype != torch.int32
        or decision_audit.device != q.device or not decision_audit.is_contiguous()
    ):
        raise ValueError("decision_audit must be contiguous on-device int32 [B,1,F,45,80,H,3]")
    if triton is None:
        raise RuntimeError("Triton is required for the isolated anchor gap guard")

    from .adaptive_seed_pytorch import multiscale_candidate_table
    from .affine_hellinger_seed_triton import fisher_anchor_seed_rows_triton

    q, k = q.contiguous(), k.contiguous()
    output = fisher_anchor_seed_rows_triton(q, k, scale, candidate_count=32,
        candidate_geometry_count=45, block_j=4, probe_count=16, global_probe_count=256,
        output_dtype=output_dtype, gs_num_warps=4)
    candidates, _ = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    _anchor_gap_guard_correction_kernel[(45 * 80, batch * heads)](
        q, k, candidates, output, output if gap_audit is None else gap_audit,
        output if decision_audit is None else decision_audit,
        K_OUTER=k.shape[1], HEADS=heads, D_HEAD=dim,
        BLOCK_D=max(32, triton.next_power_of_2(dim)), SCALE=scale,
        AUDIT_GAPS=gap_audit is not None, AUDIT_DECISION=decision_audit is not None,
        num_warps=4, num_stages=1,
    )
    return output


__all__ = ["anchor_gap_guard_seed_rows_triton"]
