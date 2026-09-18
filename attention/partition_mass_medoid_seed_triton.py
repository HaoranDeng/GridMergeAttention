"""Experimental RF0 C16 reverse-KL first medoid, then partition deficit C4.

NOT connected to production modes. Callers must supply the actual RF0 query
frame, and must retain the existing SQ1 merge/route path. Only the first seed
changes: argmin_s sum_j p_sj (log2 p_sj - sum_i pi_i log2 p_ij), where pi is
the existing globally normalized block-mass importance normalized over C16.
The matched row remains an eligible real candidate but is not mandatory.

The kernel is a deliberately isolated copy of partition_mass_seed_triton.py,
SHA256 93dbfd0b5c25171fef696af8f7d9afdae7e95064212711be17999997dac95775.
Its local logits, importance, and three deficit steps are unchanged; the
candidate bank and global normalizer are imported, not reimplemented. Copying
avoids modifying or monkey-patching the frozen production kernel. CPU tests
compare the unchanged code regions. GPU parity/quality/runtime are UNTESTED.
Finite scaled logits are required. This one-cell upper-bound initialization
does not imply an optimal or monotonically improving four-cell partition.
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


FROZEN_PARTITION_SOURCE_SHA256 = "93dbfd0b5c25171fef696af8f7d9afdae7e95064212711be17999997dac95775"


if triton is not None:

    @triton.jit
    def _partition_mass_medoid_seed_kernel(
        q_ptr,
        k_ptr,
        norm_ptr,
        candidate_ptr,
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

            # The sole algorithmic change: the best real one-cell medoid.
            # Importance has maximum 1 on each valid column, so its sum is
            # positive; the fallback only protects fully padded columns.
            importance_sum = tl.sum(importance, axis=1)
            pi = importance / tl.where(importance_sum > 0.0, importance_sum, 1.0)[:, None]
            mean_logp = tl.sum(pi[:, :, None] * log_probability, axis=1)
            medoid_cost = tl.sum(
                tl.exp2(log_probability) * (log_probability - mean_logp[:, None, :]), axis=2
            )
            _, first_index = tl.max(
                tl.where(valid_2d, -medoid_cost, -float("inf")),
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            used = candidate_ids == first_index[:, None]
            first_row = tl.sum(tl.where(used, candidate_rows_2d, 0), axis=1)
            seeds = first_row[:, None] * (seed_slots[None, :] == 0).to(tl.int32)
            first_logp = tl.sum(
                tl.where(used[:, :, None], log_probability, 0.0), axis=1
            )
            if not REUSE_OWNER_LOGP:
                selected_logp = tl.where(
                    seed_slots[None, :, None] == 0, first_logp[:, None, :], 0.0
                )
            owner = tl.zeros((BLOCK_J, 16), dtype=tl.int32)
            owner_logp = first_logp

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


def partition_mass_medoid_seed_rows_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float | None = None,
    *,
    candidate_count: int = 16,
    candidate_geometry_count: int = 16,
    block_j: int = 4,
    probe_count: int = 16,
    global_probe_count: int = 256,
    output_dtype: torch.dtype = torch.uint8,
    gs_num_warps: int = 4,
    num_warps: int | None = None,
    num_stages: int = 1,
    global_normalizer: torch.Tensor | None = None,
    reuse_owner_logp: bool = True,
) -> torch.Tensor:
    """Return four distinct real rows [B,1,T,45,80,H,4], not anchored.

    ``q[B,1,45,80,H,D]`` must already be the caller's RF0 slice;
    ``k[B,T,45,80,H,D]`` contains the unchanged key frames. As with the
    existing selector, the returned rows are used at every downstream
    query frame. The wrapper cannot infer which frame the caller sliced.

    API-compatible keyword names are retained, but the experiment is fixed
    to C16/geometry16, P16/G256, J4, four warps, owner-logp reuse, one stage.
    ``global_normalizer`` optionally supplies the unchanged contiguous FP32
    [B,1,45,80,H] base2 G256 LSE, otherwise the original normalizer runs once.
    No anchor fallback, synthetic query, fit, or extra QK pass is introduced.
    """
    if (candidate_count, candidate_geometry_count, probe_count, global_probe_count, block_j) != (16, 16, 16, 256, 4):
        raise ValueError("this isolated experiment requires C16/G16/P16/G256/J4")
    if gs_num_warps != 4 or num_warps not in (None, 4) or num_stages != 1:
        raise ValueError("this isolated experiment requires four warps and one stage")
    if reuse_owner_logp is not True:
        raise ValueError("this isolated experiment requires reuse_owner_logp=True")
    if triton is None:
        raise RuntimeError("Triton is required for partition mass medoid selection")
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q and k must be [batch, frames, rows, columns, heads, dim]")
    batch, query_frames, rows, columns, heads, head_dim = q.shape
    if query_frames != 1 or (rows, columns) != (45, 80):
        raise ValueError("the experiment requires one RF0 query frame on the 45x80 grid")
    if k.shape[0] != batch or k.shape[2:] != q.shape[2:] or k.size(1) < 1:
        raise ValueError("q and k must have matching batch and spatial/head shapes")
    if not q.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.dtype != k.dtype or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("q and k must share float16, bfloat16, or float32 dtype")
    if output_dtype not in (torch.uint8, torch.int32):
        raise TypeError("seed output must be uint8 or int32")
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
    candidates, _ = multiscale_candidate_table(
        rows, candidate_count, q.device, geometry_count=candidate_geometry_count
    )
    output = torch.empty(
        (batch, 1, k.size(1), rows, columns, heads, 4),
        device=q.device,
        dtype=output_dtype,
    )
    groups = triton.cdiv(columns, block_j)
    _partition_mass_medoid_seed_kernel[(rows * groups, batch * heads)](
        q,
        k,
        global_normalizer,
        candidates,
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
        REUSE_OWNER_LOGP=True,
        num_warps=4,
        num_stages=1,
    )
    return output


def partition_mass_medoid_seeds_reference(
    logits: torch.Tensor,
    importance: torch.Tensor,
    *,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Independent diagnostic reference, returning [...,4] candidate positions.

    Input logits use NATURAL units and shape [...,16,16]; ``importance``
    broadcasts to [...,16] and must be finite, nonnegative, positive-total
    existing block-mass weights. The first cost is evaluated by explicit
    pairwise KL, independently of the native mean-logp expression. The
    existing Torch partition reference independently recomputes ownership
    and conditional cells for the remaining three greedy selections.

    This helper is never called by the native wrapper. It materializes a
    [...,16,16,16] diagnostic temporary: callers should pass bounded chunks
    (e.g. 128 blocks), not an entire video's candidate population at once.
    It does not map positions to real rows; use the original candidate bank.
    ``return_diagnostics=True`` returns (positions, diagnostic tensors), with
    natural-unit first-medoid costs and indices from this independent
    reference, NOT native kernel registers or a native optimality claim.
    """
    if logits.ndim < 2 or tuple(logits.shape[-2:]) != (16, 16):
        raise ValueError("reference logits must have [...,16,16] shape")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("reference logits must be floating point and finite")
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    raw = logits.to(dtype)
    rho = torch.broadcast_to(importance.to(device=raw.device, dtype=dtype), raw.shape[:-1])
    total = rho.sum(dim=-1, keepdim=True)
    if not torch.isfinite(rho).all() or (rho < 0).any() or not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("importance must be finite, nonnegative, and positive-total")
    logp = torch.log_softmax(raw, dim=-1)
    pairwise_reverse_kl = (
        logp.exp().unsqueeze(-2) * (logp.unsqueeze(-2) - logp.unsqueeze(-3))
    ).sum(dim=-1)
    cost = (pairwise_reverse_kl * (rho / total).unsqueeze(-2)).sum(dim=-1)
    first = cost.argmin(dim=-1)

    from analysis.partition_variational_selector import select_partition_variational_seeds

    selected = select_partition_variational_seeds(
        raw, first, variant="gap_deficit_weighted", weights=rho, num_seeds=4
    )
    if return_diagnostics:
        return selected, {"first_medoid_cost": cost, "first_medoid_index": first}
    return selected


__all__ = ["partition_mass_medoid_seed_rows_triton", "partition_mass_medoid_seeds_reference"]
