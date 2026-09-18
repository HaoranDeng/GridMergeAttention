"""Experimental, unvalidated SQ1-aware Uniform16/G256 partition-mass seeds.

This isolated selector preserves the C32/G45 real-query bank, matched anchor,
and three current-deficit pivots of ``partition_mass_seed_triton``. It does
not implement insertion lookahead or change merge, routing, or dispatch.

For each query column j0, evaluate its singleton logit explicitly and remove
j0 from the Uniform16 bulk sketch. The m=15 or 16 retained samples each have
multiplicity eta=79/m. Normalize the full pseudo-bin using
``exp(z0) + eta * sum_bulk exp(zP)`` BEFORE assigning bulk ownership. Only
then peel the singleton, whose target probability is represented exactly.
Within each hard bulk cell, the seed's conditional weights contribute
``eta * exp(E_w[log p_target] + H(w))`` represented target mass. The next
unused real row maximizes the nonnegative missing mass times the estimated
full-block/G256 mass. Neither the singleton nor bulk samples are treated as
an unweighted 17-way softmax.

Scores are dimensionless probabilities, even though logarithms use base2.
No ln(2) conversion is appropriate for score parity. Native parity, complete
output quality, and timing have NOT been established for this candidate.
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
    def _sq1_partition_mass_seed_kernel(
        q_ptr,
        k_ptr,
        norm_ptr,
        candidate_ptr,
        anchor_ptr,
        seed_row_ptr,
        score_audit_ptr,
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
        query_3d = tl.reshape(query, (BLOCK_J, BLOCK_ROWS, BLOCK_D))
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

        # This mask removes a duplicate singleton sample, not the singleton
        # mass from the profile normalizer. It is shared by all candidate rows.
        bulk_valid = probe_columns[None, :] != output_columns[:, None]
        bulk_count = tl.sum(bulk_valid.to(tl.float32), axis=1)
        eta = (COLS - 1.0) / bulk_count
        log_eta = tl.log2(eta)

        for key_outer in tl.range(0, K_OUTER):
            key_base = (
                ((batch * K_OUTER + key_outer) * ROWS + key_row)
                * COLS * HEADS * D_HEAD
            )
            singleton_keys = tl.load(
                k_ptr + key_base
                + (output_columns[:, None] * HEADS + head) * D_HEAD
                + dims[None, :],
                mask=(output_columns[:, None] < COLS) & (dims[None, :] < D_HEAD),
                other=0.0,
            )
            # One explicit singleton dot per candidate avoids a 17-column
            # Tensor-Core product padded to 32. The inputs are not quantized;
            # multiplication and reduction for this dot are FP32.
            singleton_logits = tl.sum(
                query_3d.to(tl.float32) * singleton_keys[:, None, :].to(tl.float32),
                axis=2,
            ) * (SCALE * log2e)
            probe_keys = tl.load(
                k_ptr + key_base
                + (probe_columns[:, None] * HEADS + head) * D_HEAD
                + dims[None, :],
                mask=dims[None, :] < D_HEAD,
                other=0.0,
            )
            logits = tl.dot(query, tl.trans(probe_keys), input_precision="ieee")
            logits = tl.reshape(
                logits * (SCALE * log2e), (BLOCK_J, BLOCK_ROWS, 16)
            )

            # Center once, and derive both logp and log mass from that same
            # centered expression. eta is a quadrature multiplicity, not a
            # fitted weight or a softmax temperature.
            row_max = tl.maximum(
                singleton_logits,
                tl.max(tl.where(bulk_valid[:, None, :], logits, -float("inf")), axis=2),
            )
            centered_logits = logits - row_max[:, :, None]
            centered_singleton = singleton_logits - row_max
            centered_log_mass = tl.log2(
                tl.exp2(centered_singleton)
                + eta[:, None] * tl.sum(
                    tl.where(bulk_valid[:, None, :], tl.exp2(centered_logits), 0.0),
                    axis=2,
                )
            )
            log_probability = centered_logits - centered_log_mass[:, :, None]
            singleton_probability = tl.exp2(centered_singleton - centered_log_mass)
            log_mass = row_max + centered_log_mass
            mass = log_mass - global_norm
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
            # Profiles here include singleton mass in their normalizer. Owner
            # comparisons therefore precede SQ1 peeling, including when j0
            # happens to coincide with one of the Uniform16 probe columns.
            owner = tl.zeros((BLOCK_J, 16), dtype=tl.int32)
            owner_logp = anchor_logp

            for step in tl.static_range(1, 4):
                represented_mass = singleton_probability
                for cell_slot in tl.static_range(0, step):
                    if REUSE_OWNER_LOGP:
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
                    # Peel only now: normalize conditional weights on the
                    # remaining bulk portion of this already assigned cell.
                    cell = (owner == cell_slot) & bulk_valid
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
                    ) + entropy[:, None] + log_eta[:, None]
                    # log_eta restores the multiplicity of every sampled
                    # bulk key. The singleton has multiplicity exactly one.
                    represented_mass += tl.where(active[:, None], tl.exp2(bound), 0.0)

                deficit = tl.maximum(1.0 - represented_mass, 0.0)
                priority = importance * deficit
                if AUDIT_SCORES:
                    # Same native bank layout as output, with [step-1, row]
                    # replacing the seed-slot axis. Audit all candidates at
                    # the actual selected prefix, BEFORE excluding used rows.
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
                    # Strict > keeps the earlier selected owner on exact ties.
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


def sq1_partition_mass_seed_rows_triton(
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
    audit_scores: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return isolated SQ1-aware C4 rows [B,1,F,45,80,H,4].

    The ordinary return and arguments are Mass-selector compatible. With
    ``audit_scores=True``, return ``(rows, scores)`` where contiguous FP32
    scores have shape [B,1,F,45,80,H,3,32]. Each step records all weighted,
    clamped deficits at its actual native prefix before used-row masking;
    scores are dimensionless, not bits or nats. This optional audit can be
    large and is never allocated in the ordinary path.

    q is [B,1,45,80,H,D] and k is [B,F,45,80,H,D]. RF0 is supplied by the
    caller: this function accepts exactly one reference query frame and does
    not choose that frame. The optional shared global normalizer is the
    unchanged Uniform256 base2 logsumexp, contiguous FP32 [B,1,45,80,H].
    If omitted it is computed exactly once. Common global sketch scaling
    cancels in the per-bank importance normalization.

    BF16/FP16 inputs remain unquantized: the singleton dot uses FP32 products
    and reduction, and the 16-probe dot accumulates in FP32. Both owner-logp
    implementations express the same criterion. No device-side numerical
    validation, quality evaluation, or speed measurement has yet been done.
    """
    if triton is None:
        raise RuntimeError("Triton is required for SQ1 partition mass seed selection")
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q and k must be [batch, frames, rows, columns, heads, dim]")
    batch, query_frames, rows, columns, heads, head_dim = q.shape
    if query_frames != 1:
        raise ValueError("the shared-reference selector requires one query frame")
    if rows != 45 or columns != 80:
        raise ValueError("the experimental SQ1 selector is fixed to rows=45, columns=80")
    if batch < 1 or heads < 1 or head_dim < 1:
        raise ValueError("batch, heads, and head dimension must be positive")
    if k.shape[0] != batch or k.shape[2:] != q.shape[2:] or k.size(1) < 1:
        raise ValueError("q and k must have matching batch and spatial/head shapes")
    if not q.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("q and k must share float16, bfloat16, or float32 dtype")
    if candidate_count != 32 or candidate_geometry_count != 45:
        raise ValueError("the experimental SQ1 selector requires the C32/G45 bank")
    if probe_count != 16 or global_probe_count != 256:
        raise ValueError("the experimental SQ1 selector requires Uniform16 and global256")
    if output_dtype not in (torch.uint8, torch.int32):
        raise TypeError("seed output must be uint8 or int32")
    if block_j not in (1, 2, 4, 8):
        raise ValueError("block_j must be 1, 2, 4, or 8")
    effective_num_warps = gs_num_warps if num_warps is None else num_warps
    if effective_num_warps not in (4, 8):
        raise ValueError("gs_num_warps must be 4 or 8")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")
    if not isinstance(reuse_owner_logp, bool) or not isinstance(audit_scores, bool):
        raise TypeError("reuse_owner_logp and audit_scores must be bool")
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
    scores = None
    if audit_scores:
        scores = torch.empty(
            (batch, 1, k.size(1), rows, columns, heads, 3, candidate_count),
            device=q.device,
            dtype=torch.float32,
        )
    groups = triton.cdiv(columns, block_j)
    _sq1_partition_mass_seed_kernel[(rows * groups, batch * heads)](
        q,
        k,
        global_normalizer,
        candidates,
        anchors,
        output,
        output if scores is None else scores,
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
        AUDIT_SCORES=audit_scores,
        num_warps=effective_num_warps,
        num_stages=num_stages,
    )
    return (output, scores) if audit_scores else output


__all__ = ["sq1_partition_mass_seed_rows_triton"]
