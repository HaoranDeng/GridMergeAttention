"""Isolated, unweighted anchored-affine Hellinger seed-selection candidate.

This module is intentionally not wired into production dispatch. Its features
are sqrt(p_i) - sqrt(p_anchor), not mean-centered root probabilities. The real
matched row remains seed zero; three real rows are chosen by residual norm.
No global mass sketch, fitted weight, or singleton-complement mask is used.
The isolated fisher_zero mode is different: its first slot is a dummy index
for an externally supplied zero query, followed by three real query rows.
It is invalid with the ordinary four-real-query merge.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    if torch.version.hip is not None:
        from triton.language.extra.hip import libdevice
    else:
        from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _affine_hellinger_seed_kernel(
        q_ptr, k_ptr, candidate_ptr, anchor_ptr, seed_ptr, audit_ptr, mass_ptr,
        K_OUTER: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr, CANDIDATES: tl.constexpr,
        BLOCK_J: tl.constexpr, BLOCK_ROWS: tl.constexpr,
        N_GROUPS: tl.constexpr, BLOCK_D: tl.constexpr, SCALE: tl.constexpr,
        AUDIT_SCORES: tl.constexpr,
        FEATURE_MODE: tl.constexpr,
        SEEDS: tl.constexpr = 4,
    ):
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        group = pid % N_GROUPS
        key_row = pid // N_GROUPS
        dims = tl.arange(0, BLOCK_D)
        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        columns = group * BLOCK_J + flat // BLOCK_ROWS
        candidate_slot = flat % BLOCK_ROWS
        candidate_rows = tl.load(
            candidate_ptr + key_row * CANDIDATES + candidate_slot,
            mask=candidate_slot < CANDIDATES, other=0,
        )
        valid = (columns < COLS) & (candidate_slot < CANDIDATES)
        query_token = (batch * ROWS + candidate_rows) * COLS + columns
        query = tl.load(
            q_ptr + (query_token[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < D_HEAD), other=0.0,
        )
        candidate_rows_2d = tl.reshape(candidate_rows, (BLOCK_J, BLOCK_ROWS))
        valid_2d = tl.reshape(valid, (BLOCK_J, BLOCK_ROWS))
        candidate_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        anchor_index = tl.load(anchor_ptr + key_row)
        output_columns = group * BLOCK_J + tl.arange(0, BLOCK_J)
        seed_slots = tl.arange(0, triton.next_power_of_2(SEEDS))[None, :]
        probe_columns = tl.arange(0, 16) * (COLS // 16)
        if FEATURE_MODE == "fisher_mass":
            global_norm = tl.reshape(tl.load(
                mass_ptr + query_token * HEADS + head, mask=valid, other=0.0
            ).to(tl.float64) * 0.6931471805599453, (BLOCK_J, BLOCK_ROWS))
        if FEATURE_MODE == "fisher_mean" or FEATURE_MODE == "fisher_mean_anchor":
            # One original BF16 mean query per column, padded to the minimum
            # 16-row dot tile. Do not replace any of the C32 real candidates.
            mean_slots = tl.arange(0, 16)
            mean_columns = group * BLOCK_J + mean_slots
            mean_query = tl.load(
                mass_ptr + ((batch * COLS + mean_columns[:, None]) * HEADS + head)
                * D_HEAD + dims[None, :],
                mask=(mean_slots[:, None] < BLOCK_J) & (mean_columns[:, None] < COLS)
                & (dims[None, :] < D_HEAD), other=0.0,
            )

        for key_outer in tl.range(0, K_OUTER):
            key_base = ((batch * K_OUTER + key_outer) * ROWS + key_row)
            key_base *= COLS * HEADS * D_HEAD
            keys = tl.load(
                k_ptr + key_base + (probe_columns[:, None] * HEADS + head)
                * D_HEAD + dims[None, :],
                mask=dims[None, :] < D_HEAD, other=0.0,
            )
            # Original BF16/FP16 inputs, FP32 accumulation; no FP8 or TF32.
            dot_logits = tl.dot(query, tl.trans(keys), input_precision="ieee")
            if FEATURE_MODE == "fisher_mean" or FEATURE_MODE == "fisher_mean_anchor":
                mean_dot_logits = tl.dot(mean_query, tl.trans(keys), input_precision="ieee")
            logits = dot_logits * (SCALE * 1.4426950408889634)
            maximum = tl.max(logits, axis=1)
            exponential = tl.exp2(logits - maximum[:, None])
            denominator = tl.sum(exponential, axis=1)
            profile = tl.reshape(
                tl.sqrt(exponential / denominator[:, None]),
                (BLOCK_J, BLOCK_ROWS, 16),
            )
            used = (candidate_ids == anchor_index) & valid_2d
            anchor_profile = tl.sum(
                tl.where(used[:, :, None], profile, 0.0), axis=1
            )
            if FEATURE_MODE == "fisher_zero":
                # Hessian of logsumexp at zero logits: rho=Uniform16.
                # The synthetic zero query is not a real-row anchor.
                z = tl.reshape(dot_logits.to(tl.float64) * SCALE, (BLOCK_J, BLOCK_ROWS, 16))
                centered = z - tl.sum(z, axis=2)[:, :, None] / 16.0
                residual = centered * 0.25
            elif FEATURE_MODE != "hellinger":
                z = tl.reshape(dot_logits.to(tl.float64) * SCALE, (BLOCK_J, BLOCK_ROWS, 16))
                center = tl.sum(z, axis=1) / CANDIDATES
                if FEATURE_MODE == "fisher_barycentric_medoid":
                    # Discrete reverse-KL barycenter medoid: all four slots
                    # remain actual rows. No extra QK products or global pass.
                    center_shifted = center - tl.max(center, axis=1)[:, None]
                    center_logp = center_shifted - libdevice.log(tl.sum(
                        libdevice.exp(center_shifted), axis=1))[:, None]
                    shifted = z - tl.max(z, axis=2)[:, :, None]
                    logp = shifted - libdevice.log(tl.sum(
                        libdevice.exp(shifted), axis=2))[:, :, None]
                    cost = tl.sum(libdevice.exp(logp) * (logp-center_logp[:, None, :]), axis=2)
                    _, first = tl.max(tl.where(valid_2d, -cost, -float("inf")),
                        axis=1, return_indices=True, return_indices_tie_break_left=True)
                    used = (candidate_ids == first[:, None]) & valid_2d
                    center = tl.sum(tl.where(used[:, :, None], z, 0.0), axis=1)
                if FEATURE_MODE == "fisher_anchor":
                    center = tl.sum(tl.where(used[:, :, None], z, 0.0), axis=1)
                if FEATURE_MODE == "fisher_mean" or FEATURE_MODE == "fisher_mean_anchor":
                    mean_index = tl.broadcast_to(tl.arange(0, BLOCK_J)[:, None], (BLOCK_J, 16))
                    center = tl.gather(mean_dot_logits, mean_index, axis=0).to(tl.float64) * SCALE
                rho = libdevice.exp(center - tl.max(center, axis=1)[:, None])
                rho = rho / tl.sum(rho, axis=1)[:, None]
                anchor_z = tl.sum(tl.where(used[:, :, None], z, 0.0), axis=1)
                if FEATURE_MODE == "fisher_mean" or FEATURE_MODE == "fisher_mean_anchor":
                    anchor_z = center
                delta = z - anchor_z[:, None, :]
                centered = delta - tl.sum(delta * rho[:, None, :], axis=2)[:, :, None]
                residual = centered * tl.sqrt(rho[:, None, :])
            else:
                residual = profile - anchor_profile[:, None, :]
            initial_feature = residual
            first_direction = tl.sum(residual * 0.0, axis=1)
            if FEATURE_MODE == "fisher_mass":
                local_max = tl.max(z, axis=2)
                local_norm = local_max + libdevice.log(
                    tl.sum(libdevice.exp(z - local_max[:, :, None]), axis=2))
                difference = local_norm - global_norm
                weight = libdevice.exp(difference - tl.max(
                    tl.where(valid_2d, difference, -float("inf")), axis=1)[:, None])
            seed0 = tl.sum(tl.where(used, candidate_rows_2d, 0), axis=1)
            seeds = seed0[:, None] * (seed_slots == 0).to(tl.int32)
            if FEATURE_MODE == "fisher_zero" or FEATURE_MODE == "fisher_mean" or FEATURE_MODE == "fisher_mean_anchor":
                # Slot zero is only metadata for the external synthetic merge.
                # Every candidate, including this matched row, remains eligible.
                used = used & False
            for step in tl.static_range(1, SEEDS):
                # First pivot is its full squared Hellinger distance to anchor.
                # No anchor basis exists: the anchor difference is zero.
                score = tl.sum(residual * residual, axis=2)
                if FEATURE_MODE == "fisher_joint4":
                    # One row pivot for the concatenated four-column feature.
                    score = tl.broadcast_to(tl.sum(score, axis=0)[None, :],
                                            (BLOCK_J, BLOCK_ROWS))
                if FEATURE_MODE == "fisher_mass":
                    score *= weight
                if AUDIT_SCORES:
                    audit_offset = (
                        (((batch * K_OUTER + key_outer) * ROWS + key_row)
                         * COLS + output_columns[:, None]) * HEADS + head
                    ) * ((SEEDS - 1) * CANDIDATES) + (step - 1) * CANDIDATES + candidate_ids
                    tl.store(audit_ptr + audit_offset, score, mask=valid_2d)
                masked_score = tl.where(valid_2d & ~used, score, -float("inf"))
                _, picked = tl.max(
                    masked_score, axis=1, return_indices=True,
                    return_indices_tie_break_left=True,
                )
                if FEATURE_MODE == "fisher_mean_anchor" and step == 1:
                    # Explicit second variant: synthetic mean, real matched
                    # row, then two residual-norm pivots about the mean.
                    picked = picked * 0 + anchor_index
                if FEATURE_MODE == "fisher_antipodal" and step == 2:
                    opposite = -tl.sum(initial_feature * first_direction[:, None, :], axis=2)
                    opposite = tl.where(valid_2d & ~used, opposite, -float("inf"))
                    opposite_max, opposite_pick = tl.max(
                        opposite, axis=1, return_indices=True,
                        return_indices_tie_break_left=True,
                    )
                    picked = tl.where(opposite_max > 0.0, opposite_pick, picked)
                hit = (candidate_ids == picked[:, None]) & valid_2d
                if FEATURE_MODE == "fisher_antipodal" and step == 1:
                    first_direction = tl.sum(tl.where(hit[:, :, None], initial_feature, 0.0), axis=1)
                used |= hit
                row = tl.sum(tl.where(hit, candidate_rows_2d, 0), axis=1)
                seeds += row[:, None] * (seed_slots == step).to(tl.int32)
                if step < SEEDS - 1:
                    basis_source = tl.sum(
                        tl.where(hit[:, :, None], residual, 0.0), axis=1
                    )
                    # Same numerical safeguard as existing Hellinger-GS.
                    if FEATURE_MODE == "fisher_joint4":
                        basis = basis_source / tl.sqrt(tl.sum(tl.sum(
                            basis_source * basis_source, axis=1), axis=0) + 1.0e-12)
                    elif FEATURE_MODE != "hellinger":
                        basis = basis_source / tl.sqrt(
                            tl.sum(basis_source * basis_source, axis=1) + 1.0e-12
                        )[:, None]
                    else:
                        basis = basis_source * tl.rsqrt(
                            tl.sum(basis_source * basis_source, axis=1) + 1.0e-12
                        )[:, None]
                    projection = tl.sum(residual * basis[:, None, :], axis=2)
                    if FEATURE_MODE == "fisher_joint4":
                        # A common coefficient across columns, not four
                        # independently fitted coefficients for one row set.
                        projection = tl.broadcast_to(tl.sum(projection, axis=0)[None, :],
                                                     (BLOCK_J, BLOCK_ROWS))
                    residual -= projection[:, :, None] * basis[:, None, :]
            output_offset = (
                (((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS
                 + output_columns[:, None]) * HEADS + head
            ) * SEEDS + seed_slots
            tl.store(seed_ptr + output_offset, seeds,
                     mask=(output_columns[:, None] < COLS) & (seed_slots < SEEDS))


def affine_hellinger_seed_rows_triton(
    q: torch.Tensor, k: torch.Tensor, scale: float | None = None, *,
    candidate_count: int = 32, candidate_geometry_count: int = 45,
    block_j: int = 4, probe_count: int = 16,
    output_dtype: torch.dtype = torch.uint8, gs_num_warps: int = 4,
    num_warps: int | None = None, num_stages: int = 1,
    score_audit: torch.Tensor | None = None,
    feature_mode: str = "hellinger",
    num_seeds: int = 4,
) -> torch.Tensor:
    """Return C4 index slots in [B,1,F,45,80,H,4].

    Most ordinary modes preserve the matched real anchor and four distinct rows.
    fisher_barycentric_medoid replaces that anchor with a real reverse-KL
    medoid, retaining four distinct real rows and the ordinary merge.
    fisher_zero instead returns a dummy matched-row index in slot zero and
    three distinct real rows in slots 1:4; a real row may equal the dummy.
    The caller MUST replace slot-zero Q with an analytic zero query at merge.

    q is the caller's original RF0 slice [B,1,45,80,H,D]; k is all key frames
    [B,F,45,80,H,D]. Both are BF16 or FP16, with FP32 dot accumulators.
    Hellinger uses FP32 features; Fisher uses FP64 curvature and GS scores.
    Candidate table is exactly C32/g45 and probes are 0,5,...,75.
    No query-frame selection, singleton mask, merge or route occurs here.
    Only explicit fisher_mean modes average all45 RF0 query rows and require
    the corresponding current-frame synthetic-mean merge override.

    Optional contiguous FP32 score_audit [B,1,F,45,80,H,3,32] records every
    candidate's squared residual before used masking, at the actual selected
    prefix. These are dimensionless squared root-probability distances, NOT
    nats or bits (no ln(2) conversion). Omit audit stores for timing. Native
    selection must be validated against FP64 QK from the original inputs;
    ordinary independently rounded FP32 logits are only supplemental evidence.
    For fisher_zero the same buffer contains squared uniform-Fisher logit
    residuals, not Hellinger distances; all three prefixes begin with no real
    row excluded merely because its index appears in the dummy slot.

    The 1e-12 GS denominator safeguard is numerical, not a fitted weight.
    The method is a local entropy/Fisher surrogate, not an exact C4 optimizer.
    """
    if triton is None:
        raise RuntimeError("Triton is required for affine Hellinger seed selection")
    if type(num_seeds) is not int or num_seeds not in (4, 8, 10) or (num_seeds != 4 and feature_mode != "fisher_geometric"):
        raise ValueError("only geometric Fisher supports the explicit C8/C10 extensions")
    if feature_mode not in ("hellinger", "fisher_geometric", "fisher_anchor", "fisher_antipodal", "fisher_mass", "fisher_joint4", "fisher_zero", "fisher_mean", "fisher_mean_anchor", "fisher_barycentric_medoid"):
        raise ValueError("unknown feature mode")
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q and k must be [B,F,45,80,H,D]")
    batch, query_frames, rows, columns, heads, head_dim = q.shape
    if query_frames != 1 or (rows, columns) != (45, 80):
        raise ValueError("q must contain one 45x80 reference frame")
    if min(batch, heads, head_dim) < 1 or head_dim > 128:
        raise ValueError("require nonempty batch/heads and head dimension 1..128")
    if k.shape[0] != batch or k.shape[2:] != q.shape[2:] or k.shape[1] < 1:
        raise ValueError("q and k must share batch/grid/head/dim sizes")
    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("q and k must share original BF16 or FP16 dtype")
    if feature_mode == "fisher_zero" and q.dtype != torch.bfloat16:
        raise TypeError("fisher_zero requires original BF16 inputs")
    if feature_mode in ("fisher_mean", "fisher_mean_anchor") and q.dtype != torch.bfloat16:
        raise TypeError("mean-query Fisher requires original BF16 inputs")
    if feature_mode == "fisher_barycentric_medoid" and q.dtype != torch.bfloat16:
        raise TypeError("barycentric medoid requires original BF16 inputs")
    allowed_counts = (32, 45) if feature_mode in ("fisher_geometric", "fisher_anchor") else (32,)
    if candidate_count not in allowed_counts or candidate_geometry_count != 45 or probe_count != 16:
        raise ValueError("require supported candidate count, g45 and Uniform16 probes")
    if block_j not in (1, 2, 4, 8):
        raise ValueError("block_j must be 1, 2, 4, or 8")
    if feature_mode == "fisher_joint4" and block_j != 4:
        raise ValueError("joint4 fixes four adjacent columns as an algorithm parameter")
    effective_num_warps = gs_num_warps if num_warps is None else num_warps
    if effective_num_warps not in (4, 8) or num_stages < 1:
        raise ValueError("require 4 or 8 warps and positive num_stages")
    if output_dtype not in (torch.uint8, torch.int32):
        raise TypeError("output_dtype must be uint8 or int32")
    scale = head_dim ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    expected_audit_shape = (batch, 1, k.shape[1], rows, columns, heads, num_seeds - 1, candidate_count)
    if score_audit is not None and (
        tuple(score_audit.shape) != expected_audit_shape
        or score_audit.dtype != torch.float32 or score_audit.device != q.device
        or not score_audit.is_contiguous()
    ):
        raise ValueError("score_audit must be contiguous FP32 [B,1,F,45,80,H,3,32] on q.device")
    if score_audit is not None and feature_mode in ("fisher_antipodal","fisher_mass","fisher_joint4"):
        raise ValueError("variant scores differ from the unweighted residual-norm audit")

    from .adaptive_seed_pytorch import multiscale_candidate_table

    q, k = q.contiguous(), k.contiguous()
    if feature_mode == "fisher_mass":
        from .entropy_mass_seed_triton import global_probe_log_norm_triton
        normalizer=global_probe_log_norm_triton(q,k,scale,global_probe_count=256)
    elif feature_mode in ("fisher_mean", "fisher_mean_anchor"):
        # The supplied q contains RF0 only. This averages all 45 real rows,
        # never candidate logits, other query frames, keys, or values.
        normalizer=mean_query_bf16(q)
    else:
        normalizer=q
    candidates, anchors = multiscale_candidate_table(
        rows, candidate_count, q.device, geometry_count=candidate_geometry_count
    )
    output = torch.empty((batch, 1, k.shape[1], rows, columns, heads, num_seeds),
                         device=q.device, dtype=output_dtype)
    groups = triton.cdiv(columns, block_j)
    _affine_hellinger_seed_kernel[(rows * groups, batch * heads)](
        q, k, candidates, anchors, output, output if score_audit is None else score_audit, normalizer,
        K_OUTER=k.shape[1], ROWS=rows, COLS=columns, HEADS=heads, D_HEAD=head_dim,
        CANDIDATES=candidate_count, BLOCK_J=block_j, BLOCK_ROWS=triton.next_power_of_2(candidate_count),
        N_GROUPS=groups, BLOCK_D=max(32, triton.next_power_of_2(head_dim)),
        SCALE=scale, AUDIT_SCORES=score_audit is not None, FEATURE_MODE=feature_mode, SEEDS=num_seeds,
        num_warps=effective_num_warps, num_stages=num_stages,
    )
    return output


def fisher_geometric_seed_rows_triton(q, k, scale=None, *, candidate_count=32,
    candidate_geometry_count=45, block_j=4, probe_count=16, global_probe_count=256,
    output_dtype=torch.uint8, gs_num_warps=4):
    """Real-key Fisher metric at the mean logit; no global mass weighting.

    Accept the existing isolated Mass dispatch signature, but do not compute
    or use its global sketch. The 256 argument is only a parser compatibility
    field. All seed differences are centered at the retained real anchor.
    """
    if global_probe_count != 256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,candidate_count=candidate_count,
        candidate_geometry_count=candidate_geometry_count,block_j=block_j,probe_count=probe_count,
        output_dtype=output_dtype,gs_num_warps=gs_num_warps,feature_mode="fisher_geometric")


def fisher_geometric_c8_seed_rows_triton(q,k,scale=None,**kwargs):
    """Same geometric RF0 Fisher rule, real anchor and seven GS pivots."""
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_geometric",num_seeds=8,**kwargs)


def fisher_geometric_c10_seed_rows_triton(q,k,scale=None,**kwargs):
    """Same geometric RF0 Fisher rule, real anchor and nine GS pivots."""
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_geometric",num_seeds=10,**kwargs)


def fisher_anchor_seed_rows_triton(q,k,scale=None,**kwargs):
    """Four real seeds, RF0, Fisher curvature at the actual retained anchor."""
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_anchor",**kwargs)


def fisher_antipodal_seed_rows_triton(q,k,scale=None,**kwargs):
    """Original Fisher-GS with one opposite-direction real-query pivot."""
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_antipodal",**kwargs)


def fisher_mass_seed_rows_triton(q,k,scale=None,**kwargs):
    """Geometric Fisher residuals weighted by the unchanged G256 block mass."""
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("require the original G256 normalizer")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_mass",**kwargs)


def fisher_joint4_seed_rows_triton(q,k,scale=None,**kwargs):
    """RF0 four-column concatenated Fisher-GS; share rows, never merged KV.

    Four adjacent columns choose the same real row indices using one common
    Gram-Schmidt projection in their concatenated 64-dimensional feature.
    The existing merge subsequently loads each column/current frame's Q.
    """
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_joint4",**kwargs)


def fisher_zero_seed_rows_triton(q,k,scale=None,**kwargs):
    """Three real RF0 pivots about zero logits, plus a dummy slot-zero row.

    Requires the caller's zero-query merge override: without that override,
    these indices implement a different and invalid four-real-query method.
    Slot zero holds matched row r as metadata; only slots 1:4 are distinct,
    and any of them may equal r. The zero query itself has no row index.
    Uniform16 gives FP64 features (z-mean(z))/4, from BF16 QK with FP32
    accumulation. Candidate32/g45 and the usual three-prefix score audit
    are supported. No synthetic query tensor is generated by this function.
    """
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_zero",**kwargs)


def mean_query_bf16(q):
    """Per-frame/per-column mean of all 45 real rows: FP32 reduce, then BF16.

    No explicit full-Q FP32 intermediate is requested. The returned tensor
    is [B,F,80,H,D], contiguous, and is recomputed from the supplied current
    queries. Passing RF0 alone restricts the selector to RF0 information.
    """
    if (q.ndim != 6 or q.shape[2:4] != (45,80) or min(q.shape) < 1
            or q.shape[-1] > 128 or q.dtype != torch.bfloat16):
        raise ValueError("require original BF16 [B,F,45,80,H,D<=128] queries")
    return q.mean(dim=2,dtype=torch.float32).to(q.dtype).contiguous()


def fisher_mean_seed_rows_triton(q,k,scale=None,**kwargs):
    """RF0 synthetic mean plus three unrestricted real Fisher-GS pivots.

    Slot zero is a dummy matched-row identifier: requires MeanSeedMergeAdapter
    or an equivalent current-frame mean-query merge override. Real slots1:4
    are distinct but may include the dummy row. Curvature is softmax of the
    BF16 mean query's BF16 QK/FP32-accumulated probe logits; features and GS
    use FP64. The mean is over all 45 RF0 rows, not the candidate subset.
    """
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_mean",**kwargs)


def fisher_mean_anchor_seed_rows_triton(q,k,scale=None,**kwargs):
    """RF0 synthetic mean, actual matched row in slot1, then two GS pivots.

    Requires the same current-frame MeanSeedMergeAdapter. The score audit
    records all three raw residual-norm prefixes, but the first real pivot
    is deliberately forced to the matched row, not chosen by maximum score.
    """
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_mean_anchor",**kwargs)


def fisher_barycentric_medoid_seed_rows_triton(q,k,scale=None,**kwargs):
    """Real reverse-KL medoid first, then three anchor-Fisher GS pivots.

    Uses only C32/g45 RF0 queries and Uniform16 probes. All four slots are
    real distinct rows; ordinary current-frame actual-Q merge is required.
    The exact first-seed minimizer on the fixed probe cell does not imply
    an optimal four-cell hard partition or improved video quality.
    """
    if kwargs.pop("global_probe_count",256)!=256:
        raise ValueError("unexpected dispatch global-probe field")
    return affine_hellinger_seed_rows_triton(q,k,scale,feature_mode="fisher_barycentric_medoid",**kwargs)


__all__ = ["affine_hellinger_seed_rows_triton", "fisher_geometric_seed_rows_triton",
           "fisher_anchor_seed_rows_triton", "fisher_antipodal_seed_rows_triton",
           "fisher_mass_seed_rows_triton", "fisher_joint4_seed_rows_triton",
           "fisher_zero_seed_rows_triton", "fisher_mean_seed_rows_triton",
           "fisher_mean_anchor_seed_rows_triton", "fisher_barycentric_medoid_seed_rows_triton",
           "mean_query_bf16"]
