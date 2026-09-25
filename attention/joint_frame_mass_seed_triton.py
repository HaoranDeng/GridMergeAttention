"""Two-frame mass-weighted Hellinger GS with shared C4 row pivots.

Each frame retains its own residual, basis norm, and projection coefficient.
Only weighted residual energies are summed across frames to choose a row.
The caller supplies the two distinct router frames; merge/route are untouched.
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
    def _joint_frame_mass_seed_kernel(
        q_ptr, k_ptr, norm_ptr, candidate_ptr, anchor_ptr, seed_ptr,
        K_OUTER: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
        HEADS: tl.constexpr, D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr, BLOCK_ROWS: tl.constexpr,
        BLOCK_J: tl.constexpr, N_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr, SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch, head = batch_head // HEADS, batch_head % HEADS
        key_row, group = pid // N_GROUPS, pid % N_GROUPS
        dims = tl.arange(0, BLOCK_D)
        # [column, frame, candidate] flattening keeps the local MMA's total
        # row count identical to single-frame C32 when using two-frame C16.
        flat = tl.arange(0, BLOCK_J * 2 * BLOCK_ROWS)
        candidate_slot = flat % BLOCK_ROWS
        frame_slot = (flat // BLOCK_ROWS) % 2
        columns = group * BLOCK_J + flat // (2 * BLOCK_ROWS)
        candidate_rows = tl.load(
            candidate_ptr + key_row * CANDIDATES + candidate_slot,
            mask=candidate_slot < CANDIDATES, other=0,
        )
        valid = (columns < COLS) & (candidate_slot < CANDIDATES)
        query_token = (
            ((batch * 2 + frame_slot) * ROWS + candidate_rows) * COLS + columns
        )
        query = tl.load(
            q_ptr + (query_token[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < D_HEAD), other=0.0,
        )
        global_norm = tl.load(norm_ptr + query_token * HEADS + head, mask=valid, other=0.0)
        global_norm = tl.reshape(global_norm, (BLOCK_J, 2, BLOCK_ROWS))
        candidate_ids = tl.arange(0, BLOCK_ROWS)
        row_ids = tl.load(
            candidate_ptr + key_row * CANDIDATES + candidate_ids,
            mask=candidate_ids < CANDIDATES, other=0,
        )
        output_columns = group * BLOCK_J + tl.arange(0, BLOCK_J)
        valid_2d = (output_columns[:, None] < COLS) & (candidate_ids[None, :] < CANDIDATES)
        anchor = tl.load(anchor_ptr + key_row)
        anchor_row = tl.sum(tl.where(candidate_ids == anchor, row_ids, 0), axis=0)
        seed_slots = tl.arange(0, 4)
        probe_columns = tl.arange(0, 16) * (COLS // 16)

        for key_outer in tl.range(0, K_OUTER):
            key_base = ((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS
            keys = tl.load(
                k_ptr + ((key_base + probe_columns[:, None]) * HEADS + head)
                * D_HEAD + dims[None, :],
                mask=dims[None, :] < D_HEAD, other=0.0,
            )
            logits = tl.dot(query, tl.trans(keys), input_precision="ieee")
            logits *= SCALE * 1.4426950408889634
            maximum = tl.max(logits, axis=1)
            exponential = tl.exp2(logits - maximum[:, None])
            denominator = tl.sum(exponential, axis=1)
            log_mass = maximum + tl.log2(denominator)
            profile = tl.reshape(
                tl.sqrt(exponential / denominator[:, None]),
                (BLOCK_J, 2, BLOCK_ROWS, 16),
            )
            residual = profile - tl.sum(profile, axis=3)[:, :, :, None] * (1.0 / 16.0)
            difference = tl.reshape(log_mass, (BLOCK_J, 2, BLOCK_ROWS)) - global_norm
            # A separate maximum/softmax in each frame would change their
            # relative demand and would not implement the screened objective.
            maximum_mass = tl.max(
                tl.max(tl.where(valid_2d[:, None, :], difference, -float("inf")), axis=2),
                axis=1,
            )
            maximum_mass = tl.where(output_columns < COLS, maximum_mass, 0.0)
            weight = tl.where(
                valid_2d[:, None, :], tl.exp2(difference - maximum_mass[:, None, None]), 0.0
            )
            used = (candidate_ids[None, :] == anchor) & valid_2d
            seeds = tl.full((BLOCK_J, 4), 0, tl.int32)
            seeds += anchor_row * (seed_slots[None, :] == 0).to(tl.int32)
            basis_source = tl.sum(
                tl.where(used[:, None, :, None], residual, 0.0), axis=2
            )
            for step in tl.static_range(1, 4):
                basis = basis_source * tl.rsqrt(
                    tl.sum(basis_source * basis_source, axis=2) + 1.0e-12
                )[:, :, None]
                projection = tl.sum(residual * basis[:, :, None, :], axis=3)
                residual -= projection[:, :, :, None] * basis[:, :, None, :]
                objective = tl.sum(
                    tl.sum(residual * residual, axis=3) * weight, axis=1
                )
                objective = tl.where(valid_2d & ~used, objective, -float("inf"))
                _, picked = tl.max(
                    objective, axis=1, return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit = (candidate_ids[None, :] == picked[:, None]) & valid_2d
                used |= hit
                row = tl.sum(tl.where(hit, row_ids[None, :], 0), axis=1)
                seeds += row[:, None] * (seed_slots[None, :] == step).to(tl.int32)
                if step < 3:
                    basis_source = tl.sum(
                        tl.where(hit[:, None, :, None], residual, 0.0), axis=2
                    )
            offset = (
                (((batch * K_OUTER + key_outer) * ROWS + key_row) * COLS
                 + output_columns[:, None]) * HEADS + head
            ) * 4 + seed_slots[None, :]
            tl.store(seed_ptr + offset, seeds, mask=output_columns[:, None] < COLS)


def _check_joint_inputs(q, k):
    if triton is None:
        raise RuntimeError("Triton is required for joint-frame mass seed selection")
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q/k must be [B,T,45,80,H,D]")
    if q.shape[1:4] != (2, 45, 80):
        raise ValueError("q must contain exactly two 45x80 router frames")
    if k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:] or k.shape[1] < 1:
        raise ValueError("q/k must share batch, spatial, head, and dimension sizes")
    if q.shape[0] < 1 or q.shape[-2] < 1 or not 1 <= q.shape[-1] <= 128:
        raise ValueError("require positive batch/heads and head dimension <= 128")
    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("q/k must share a CUDA device")
    if q.dtype != k.dtype or q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("q/k must share BF16, FP16, or FP32 dtype")


def joint_frame_global_probe_log_norm_triton(
    q, k, scale=None, *, global_probe_count=256, block_q=32,
    num_warps=8, num_stages=1, native_low_precision_dot=False,
):
    """Return contiguous FP32 log2 LSE [B,2,45,80,H] with one launch.

    Reuses the existing normalizer kernel with 7200 query tokens; global key
    probes and FP32/TF32x3 accumulation are unchanged by default. The opt-in
    native dot applies only to BF16/FP16 inputs and requires GPU parity checks.
    """
    _check_joint_inputs(q, k)
    batch, _, rows, columns, heads, dim = q.shape
    key_tokens = k.shape[1] * rows * columns
    if not 1 <= global_probe_count <= min(key_tokens, 1024):
        raise ValueError("global_probe_count must be in [1, min(key_tokens,1024)]")
    if block_q not in (16, 32, 64) or num_warps not in (4, 8) or num_stages < 1:
        raise ValueError("invalid normalizer execution settings")
    scale = dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    from .entropy_mass_seed_triton import _global_probe_log_norm_kernel

    q, k = q.contiguous(), k.contiguous()
    normalizer = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    query_tokens = 2 * rows * columns
    _global_probe_log_norm_kernel[(triton.cdiv(query_tokens, block_q), batch * heads)](
        q, k, normalizer, QUERY_TOKENS=query_tokens, KEY_TOKENS=key_tokens,
        HEADS=heads, D_HEAD=dim, PROBES=global_probe_count, BLOCK_Q=block_q,
        BLOCK_P=max(16, triton.next_power_of_2(global_probe_count)),
        BLOCK_D=max(32, triton.next_power_of_2(dim)), SCALE=scale,
        NATIVE_LOW_PRECISION_DOT=(
            bool(native_low_precision_dot) and q.dtype in (torch.bfloat16, torch.float16)
        ),
        TF32_LOW_PRECISION_DOT=False, num_warps=num_warps, num_stages=num_stages,
    )
    return normalizer


def joint_frame_mass_seed_rows_triton(
    q, k, scale=None, *, candidate_count=16, candidate_geometry_count=16,
    block_j=4, probe_count=16, global_probe_count=256,
    output_dtype=torch.uint8, gs_num_warps=4, gs_num_stages=1,
    norm_block_q=32, norm_num_warps=8, norm_num_stages=1,
    native_low_precision_dot=False,
):
    """Select shared C4 rows from two unaveraged router frames.

    q is [B,2,45,80,H,D], k is [B,T,45,80,H,D]; output is
    [B,1,T,45,80,H,4]. Supported screened banks are C16/G16 and C32/G45.
    Each frame gets its own softmax, centered root profile and GS projection.
    A single maximum across frames AND candidates stabilizes their mass
    weights before their residual energies are summed. Anchor/tie behavior
    matches the existing selector; frame choice is strictly caller-owned.
    """
    _check_joint_inputs(q, k)
    if (candidate_count, candidate_geometry_count) not in ((16, 16), (32, 45)):
        raise ValueError("supported candidate/geometry pairs are C16/G16 and C32/G45")
    if probe_count != 16 or global_probe_count != 256:
        raise ValueError("the joint selector requires Uniform16 and global Uniform256")
    if output_dtype not in (torch.uint8, torch.int32):
        raise TypeError("seed rows must use uint8 or int32")
    if block_j not in (1, 2, 4, 8) or gs_num_warps not in (4, 8) or gs_num_stages < 1:
        raise ValueError("invalid GS execution settings")
    batch, _, rows, columns, heads, dim = q.shape
    scale = dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    from .adaptive_seed_pytorch import multiscale_candidate_table

    q, k = q.contiguous(), k.contiguous()
    normalizer = joint_frame_global_probe_log_norm_triton(
        q, k, scale, global_probe_count=global_probe_count,
        block_q=norm_block_q, num_warps=norm_num_warps, num_stages=norm_num_stages,
        native_low_precision_dot=native_low_precision_dot,
    )
    candidates, anchors = multiscale_candidate_table(
        rows, candidate_count, q.device, geometry_count=candidate_geometry_count,
    )
    output = torch.empty(
        (batch, 1, k.shape[1], rows, columns, heads, 4), device=q.device, dtype=output_dtype,
    )
    groups = triton.cdiv(columns, block_j)
    _joint_frame_mass_seed_kernel[(rows * groups, batch * heads)](
        q, k, normalizer, candidates, anchors, output,
        K_OUTER=k.shape[1], ROWS=rows, COLS=columns, HEADS=heads, D_HEAD=dim,
        CANDIDATES=candidate_count, BLOCK_ROWS=triton.next_power_of_2(candidate_count),
        BLOCK_J=block_j, N_GROUPS=groups, BLOCK_D=max(32, triton.next_power_of_2(dim)),
        SCALE=scale, num_warps=gs_num_warps, num_stages=gs_num_stages,
    )
    return output


__all__ = ["joint_frame_mass_seed_rows_triton", "joint_frame_global_probe_log_norm_triton"]
