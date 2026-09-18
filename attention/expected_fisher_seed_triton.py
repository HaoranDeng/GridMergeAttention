"""Isolated expected-Fisher RF0 seed selection; no production dispatch.

Each program owns ONE query column/key row/head (J1), and loops over key
frames. No row/cluster decisions are shared across columns or key blocks.
Original BF16 QK accumulates in FP32; scaling, softmax, curvature and all
subsequent GS arithmetic are FP64. The metric is the average of candidate
categorical Fishers, NOT Fisher at their average distribution. Four actual
row indices are returned, with the matched anchor fixed in slot zero.

Only the seed table is allocated by default. A bounded 32x32 Y tensor and
two residual columns replace the full residual Gram matrix. These are Triton
logical program-local values; register residency/spilling is NOT yet measured.
Optional FP64 score stores are diagnostic only and must be off for timing.
No GPU compilation, native parity or performance result is claimed here.
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
    def _expected_geometry(z, anchor):
        """FP64 delta/rho/Y/diagonal, with Y[i,r] = delta[i] dot p[r]."""
        ids = tl.arange(0, 32)
        probes = tl.arange(0, 16)
        exponential = libdevice.exp(z - tl.max(z, axis=1)[:, None])
        probability = exponential / tl.sum(exponential, axis=1)[:, None]
        rho = tl.sum(probability, axis=0) / 32.0
        anchor_z = tl.sum(tl.where(ids[:, None] == anchor, z, 0.0), axis=0)
        delta = z - anchor_z[None, :]
        first_coordinate = tl.sum(tl.where(probes[None, :] == 0, delta, 0.0), axis=1)
        delta = delta - first_coordinate[:, None]
        y = tl.full((32, 32), 0.0, tl.float64)
        # Sixteen FP64 outer products, no Tensor-Core/FP32 feature product.
        for probe in tl.range(0, 16):
            delta_column = tl.sum(tl.where(probes[None, :] == probe, delta, 0.0), axis=1)
            probability_column = tl.sum(
                tl.where(probes[None, :] == probe, probability, 0.0), axis=1)
            y += delta_column[:, None] * probability_column[None, :]
        diagonal = tl.sum(delta * delta * rho[None, :], axis=1)
        diagonal -= tl.sum(y * y, axis=1) / 32.0
        return delta, rho, y, diagonal

    @triton.jit
    def _expected_covariance_column(delta, rho, y, pivot):
        ids = tl.arange(0, 32)
        pivot_delta = tl.sum(tl.where(ids[:, None] == pivot, delta, 0.0), axis=0)
        pivot_y = tl.sum(tl.where(ids[:, None] == pivot, y, 0.0), axis=0)
        column = tl.sum(delta * pivot_delta[None, :] * rho[None, :], axis=1)
        column -= tl.sum(y * pivot_y[None, :], axis=1) / 32.0
        return column

    @triton.jit
    def _safeguarded_factor(norm):
        # Exact Gram counterpart of x -= <x,u>u with u=x_p/sqrt(d+eps).
        # 1/(d+eps) is a different update whenever eps matters.
        epsilon = tl.full((), 1.0e-12, tl.float64)
        return (norm + 2.0 * epsilon) / ((norm + epsilon) * (norm + epsilon))

    @triton.jit
    def _expected_pivots(z, anchor):
        """Only two rank updates are needed to choose three non-anchor rows."""
        ids = tl.arange(0, 32)
        seed_slots = tl.arange(0, 4)
        delta, rho, y, diagonal = _expected_geometry(z, anchor)
        used = ids == anchor
        score0 = tl.maximum(diagonal, 0.0)
        _, pick1 = tl.max(tl.where(~used, score0, -float("inf")), axis=0,
                         return_indices=True, return_indices_tie_break_left=True)
        used |= ids == pick1
        norm1 = tl.sum(tl.where(ids == pick1, score0, 0.0), axis=0)
        column1 = _expected_covariance_column(delta, rho, y, pick1)
        factor1 = _safeguarded_factor(norm1)
        diagonal1 = diagonal - column1 * column1 * factor1

        score1 = tl.maximum(diagonal1, 0.0)
        _, pick2 = tl.max(tl.where(~used, score1, -float("inf")), axis=0,
                         return_indices=True, return_indices_tie_break_left=True)
        used |= ids == pick2
        norm2 = tl.sum(tl.where(ids == pick2, score1, 0.0), axis=0)
        raw_column2 = _expected_covariance_column(delta, rho, y, pick2)
        first_at_second = tl.sum(tl.where(ids == pick2, column1, 0.0), axis=0)
        # The second pivot column MUST be from the first residual Gram.
        column2 = raw_column2 - column1 * first_at_second * factor1
        factor2 = _safeguarded_factor(norm2)
        diagonal2 = diagonal1 - column2 * column2 * factor2

        score2 = tl.maximum(diagonal2, 0.0)
        _, pick3 = tl.max(tl.where(~used, score2, -float("inf")), axis=0,
                         return_indices=True, return_indices_tie_break_left=True)
        selected = tl.where(seed_slots == 0, anchor,
                   tl.where(seed_slots == 1, pick1, tl.where(seed_slots == 2, pick2, pick3)))
        return selected, score0, score1, score2

    @triton.jit
    def _expected_fisher_seed_kernel(
        q_ptr, k_ptr, candidate_ptr, anchor_ptr, seed_ptr, audit_ptr,
        K_OUTER: tl.constexpr, HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        BLOCK_D: tl.constexpr, SCALE: tl.constexpr, AUDIT_SCORES: tl.constexpr,
    ):
        # Fixed J1 launch: 45*80 programs, not a reinterpretation of J4.
        bank = tl.program_id(0)
        batch_head = tl.program_id(1)
        column = bank % 80
        key_row = bank // 80
        head = batch_head % HEADS
        batch = (batch_head // HEADS).to(tl.int64)
        ids = tl.arange(0, 32)
        dims = tl.arange(0, BLOCK_D)
        probes = tl.arange(0, 16) * 5
        rows = tl.load(candidate_ptr + key_row * 32 + ids)
        anchor = tl.load(anchor_ptr + key_row)
        q_token = (batch * 45 + rows) * 80 + column
        queries = tl.load(q_ptr + (q_token[:, None] * HEADS + head) * D_HEAD + dims[None, :],
                          mask=dims[None, :] < D_HEAD, other=0.0)
        for key_frame in tl.range(0, K_OUTER):
            key_base = ((batch * K_OUTER + key_frame) * 45 + key_row) * 80
            keys = tl.load(k_ptr + ((key_base + probes[:, None]) * HEADS + head) * D_HEAD
                           + dims[None, :], mask=dims[None, :] < D_HEAD, other=0.0)
            dot_logits = tl.dot(queries, tl.trans(keys), input_precision="ieee", out_dtype=tl.float32)
            z = dot_logits.to(tl.float64) * SCALE
            selected, score0, score1, score2 = _expected_pivots(z, anchor)
            real_rows = tl.sum(tl.where(ids[:, None] == selected[None, :], rows[:, None], 0), axis=0)
            output_bank = (((batch * K_OUTER + key_frame) * 45 + key_row) * 80 + column) * HEADS + head
            tl.store(seed_ptr + output_bank * 4 + tl.arange(0, 4), real_rows)
            if AUDIT_SCORES:
                tl.store(audit_ptr + (output_bank * 3 + 0) * 32 + ids, score0)
                tl.store(audit_ptr + (output_bank * 3 + 1) * 32 + ids, score1)
                tl.store(audit_ptr + (output_bank * 3 + 2) * 32 + ids, score2)


def fisher_expected_seed_rows_triton(
    q, k, scale=None, *, candidate_count=32, candidate_geometry_count=45,
    block_j=1, probe_count=16, global_probe_count=256,
    output_dtype=torch.uint8, gs_num_warps=4, score_audit=None,
):
    """Return uint8/int32 [B,1,F,45,80,H,4] real-row indices; J1 ONLY.

    q is the caller's RF0 slice [B,1,45,80,H,D], k=[B,F,45,80,H,D]. The
    caller must supply actual RF0; the outer video audit verifies this view.
    BF16 and 1<=D<=128 are mandatory. Finite tensor values are a caller
    precondition, not a hidden device reduction/synchronization here.

    Unlike the existing J4 wrapper, block_j defaults to1 and non1 is rejected.
    global_probe_count=256 is accepted only as a dispatch-compatibility field;
    no global probes/normalizer or mass weighting are computed or consumed.
    No **kwargs escape hatch accepts an alternate feature/normalizer/precision.

    Optional score_audit is contiguous FP64 [B,1,F,45,80,H,3,32] on-device,
    storing every raw clamped norm before used-row masking at actual prefixes.
    It adds diagnostic HBM stores; omit it for timing/normal operation.
    """
    if (type(candidate_count) is not int or candidate_count != 32
            or type(candidate_geometry_count) is not int or candidate_geometry_count != 45):
        raise ValueError("expected-Fisher requires C32/g45")
    if type(block_j) is not int or block_j != 1:
        raise ValueError("expected-Fisher native prototype supports actual J1 only")
    if type(probe_count) is not int or probe_count != 16:
        raise ValueError("expected-Fisher requires Uniform16")
    if type(global_probe_count) is not int or global_probe_count != 256:
        raise ValueError("unexpected dispatch global-probe field; no global sketch is used")
    if type(gs_num_warps) is not int or gs_num_warps != 4:
        raise ValueError("expected-Fisher J1 prototype requires four warps")
    if q.ndim != 6 or k.ndim != 6 or min(q.shape) < 1 or min(k.shape) < 1:
        raise ValueError("require nonempty [B,F,45,80,H,D] Q/K")
    batch, query_frames, rows, columns, heads, dim = q.shape
    if query_frames != 1 or (rows, columns) != (45, 80) or dim > 128:
        raise ValueError("q must be the single RF0 slice [B,1,45,80,H,D<=128]")
    if k.shape[0] != batch or k.shape[2:] != q.shape[2:]:
        raise ValueError("Q/K must share batch/spatial/head/dimension shapes")
    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("Q/K must share a CUDA device")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("expected-Fisher requires original BF16 Q/K")
    if output_dtype not in (torch.uint8, torch.int32):
        raise TypeError("row-table output must be uint8 or int32")
    scale = dim ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    audit_shape = (batch, 1, k.shape[1], 45, 80, heads, 3, 32)
    if score_audit is not None and (
        tuple(score_audit.shape) != audit_shape or score_audit.dtype != torch.float64
        or score_audit.device != q.device or not score_audit.is_contiguous()
    ):
        raise ValueError("score_audit must be contiguous CUDA FP64 [B,1,F,45,80,H,3,32]")
    if triton is None:
        raise RuntimeError("Triton is required for expected-Fisher seed selection")

    from .adaptive_seed_pytorch import multiscale_candidate_table

    q, k = q.contiguous(), k.contiguous()
    candidates, anchors = multiscale_candidate_table(45, 32, q.device, geometry_count=45)
    output = torch.empty((batch, 1, k.shape[1], 45, 80, heads, 4), device=q.device, dtype=output_dtype)
    _expected_fisher_seed_kernel[(45 * 80, batch * heads)](
        q, k, candidates, anchors, output, output if score_audit is None else score_audit,
        K_OUTER=k.shape[1], HEADS=heads, D_HEAD=dim,
        BLOCK_D=max(32, triton.next_power_of_2(dim)), SCALE=scale,
        AUDIT_SCORES=score_audit is not None, num_warps=4, num_stages=1,
    )
    return output


__all__ = ["fisher_expected_seed_rows_triton"]
