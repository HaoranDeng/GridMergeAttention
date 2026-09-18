"""Entropy-mass-weighted Hellinger seed selection with a global probe sketch.

The local probe partition estimates response to one key block.  Dividing by
the same query's global probe partition makes the weight approximate attention
mass on that block, rather than unnormalized response strength.  The mass
version has no importance temperature or learned parameters.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _global_probe_log_norm_kernel(
        q_ptr,
        k_ptr,
        norm_ptr,
        QUERY_TOKENS: tl.constexpr,
        KEY_TOKENS: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        PROBES: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
        NATIVE_LOW_PRECISION_DOT: tl.constexpr,
        TF32_LOW_PRECISION_DOT: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        query_index = query_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
        probe_slot = tl.arange(0, BLOCK_P)
        key_index = probe_slot * (KEY_TOKENS // PROBES)
        dims = tl.arange(0, BLOCK_D)
        query = tl.load(
            q_ptr + ((batch * QUERY_TOKENS + query_index[:, None]) * HEADS + head)
            * D_HEAD + dims[None, :],
            mask=(query_index[:, None] < QUERY_TOKENS) & (dims[None, :] < D_HEAD),
            other=0.0,
        )
        keys = tl.load(
            k_ptr + ((batch * KEY_TOKENS + key_index[:, None]) * HEADS + head)
            * D_HEAD + dims[None, :],
            mask=(probe_slot[:, None] < PROBES) & (dims[None, :] < D_HEAD),
            other=0.0,
        )
        if NATIVE_LOW_PRECISION_DOT:
            # Original BF16/FP16 inputs go directly to Tensor Cores. The
            # accumulator remains FP32; no additional input quantization.
            logits = tl.dot(query, tl.trans(keys), input_precision="ieee")
        elif TF32_LOW_PRECISION_DOT:
            # Normal BF16/FP16 source values are exactly representable in
            # TF32 after the exact FP32 cast. This removes extra residual
            # products, but different MMA lowering/accumulation can still
            # change bits; the benchmark explicitly checks that assumption.
            logits = tl.dot(
                query.to(tl.float32), tl.trans(keys.to(tl.float32)),
                input_precision="tf32",
            )
        else:
            logits = tl.dot(
                query.to(tl.float32), tl.trans(keys.to(tl.float32)),
                input_precision="tf32x3",
            )
        logits *= SCALE * 1.4426950408889634
        logits = tl.where(probe_slot[None, :] < PROBES, logits, -float("inf"))
        maximum = tl.max(logits, axis=1)
        total = tl.sum(tl.exp2(logits - maximum[:, None]), axis=1)
        norm = maximum + tl.log2(total)
        tl.store(
            norm_ptr + (batch * QUERY_TOKENS + query_index) * HEADS + head,
            norm,
            mask=query_index < QUERY_TOKENS,
        )

    @triton.jit
    def _entropy_mass_seed_kernel(
        q_ptr,
        k_ptr,
        norm_ptr,
        candidate_ptr,
        anchor_ptr,
        seed_ptr,
        Q_OUTER: tl.constexpr,
        K_OUTER: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        CANDIDATES: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        PROBES: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_J: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_GROUPS: tl.constexpr,
        SCALE: tl.constexpr,
        MASS_WEIGHT: tl.constexpr,
        ALPHA: tl.constexpr,
        SINGLETON_COMPLEMENT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_head = tl.program_id(1)
        head = batch_head % HEADS
        batch = batch_head // HEADS
        group = pid % N_GROUPS
        key_row = (pid // N_GROUPS) % ROWS
        query_outer = pid // (N_GROUPS * ROWS)
        log2e: tl.constexpr = 1.4426950408889634
        dims = tl.arange(0, BLOCK_D)
        flat = tl.arange(0, BLOCK_J * BLOCK_ROWS)
        columns = group * BLOCK_J + flat // BLOCK_ROWS
        candidate_slot = flat % BLOCK_ROWS
        candidate_rows = tl.load(
            candidate_ptr + key_row * CANDIDATES + candidate_slot,
            mask=candidate_slot < CANDIDATES,
            other=0,
        )
        valid = (columns < COLS) & (candidate_slot < CANDIDATES)
        query_token = (
            ((batch * Q_OUTER + query_outer) * ROWS + candidate_rows) * COLS
            + columns
        )
        query = tl.load(
            q_ptr + (query_token[:, None] * HEADS + head) * D_HEAD + dims[None, :],
            mask=valid[:, None] & (dims[None, :] < D_HEAD),
            other=0.0,
        )
        if MASS_WEIGHT:
            global_norm = tl.load(
                norm_ptr + query_token * HEADS + head,
                mask=valid,
                other=0.0,
            )
            global_norm = tl.reshape(global_norm, (BLOCK_J, BLOCK_ROWS))
        rows_2d = tl.reshape(candidate_rows, (BLOCK_J, BLOCK_ROWS))
        valid_2d = tl.reshape(valid, (BLOCK_J, BLOCK_ROWS))
        candidate_ids = tl.arange(0, BLOCK_ROWS)[None, :]
        anchor = tl.load(anchor_ptr + key_row)
        output_columns = group * BLOCK_J + tl.arange(0, BLOCK_J)
        seed_slots = tl.arange(0, 4)[None, :]
        probe_slot = tl.arange(0, BLOCK_P)
        probe_columns = probe_slot * COLS // PROBES
        probe_valid = probe_slot < PROBES
        if SINGLETON_COMPLEMENT:
            # Uniform probes are distinct, so dropping at most one leaves
            # positive support. This mask is reused for every key frame.
            tl.static_assert(1 < PROBES <= COLS)
            local_probe_valid = probe_valid[None, :] & (
                probe_columns[None, :] != columns[:, None]
            )
            local_probe_count = tl.sum(local_probe_valid.to(tl.float32), axis=1)
            local_probe_valid_3d = tl.reshape(
                local_probe_valid, (BLOCK_J, BLOCK_ROWS, BLOCK_P)
            )
            local_probe_count_2d = tl.reshape(
                local_probe_count, (BLOCK_J, BLOCK_ROWS)
            )

        for key_outer in tl.range(0, K_OUTER):
            key_base = ((batch * K_OUTER + key_outer) * ROWS + key_row)
            key_base *= COLS * HEADS * D_HEAD
            keys = tl.load(
                k_ptr + key_base + (probe_columns[:, None] * HEADS + head)
                * D_HEAD + dims[None, :],
                mask=probe_valid[:, None] & (dims[None, :] < D_HEAD),
                other=0.0,
            )
            logits = tl.dot(query, tl.trans(keys), input_precision="ieee")
            logits *= SCALE * log2e
            if SINGLETON_COMPLEMENT:
                logits = tl.where(local_probe_valid, logits, -float("inf"))
            else:
                logits = tl.where(probe_valid[None, :], logits, -float("inf"))
            maximum = tl.max(logits, axis=1)
            exponential = tl.exp2(logits - maximum[:, None])
            denominator = tl.sum(exponential, axis=1)
            log_mass = maximum + tl.log2(denominator)
            probability = exponential / denominator[:, None]
            profile = tl.sqrt(probability)
            profile = tl.reshape(profile, (BLOCK_J, BLOCK_ROWS, BLOCK_P))
            if SINGLETON_COMPLEMENT:
                residual = tl.where(
                    local_probe_valid_3d,
                    profile - (
                        tl.sum(profile, axis=2) / local_probe_count_2d
                    )[:, :, None],
                    0.0,
                )
            else:
                residual = tl.where(
                    probe_valid[None, None, :],
                    profile - tl.sum(profile, axis=2)[:, :, None] * (1.0 / PROBES),
                    0.0,
                )
            local_norm = tl.reshape(log_mass, (BLOCK_J, BLOCK_ROWS))
            if MASS_WEIGHT:
                difference = local_norm - global_norm
                maximum_difference = tl.max(
                    tl.where(valid_2d, difference, -float("inf")), axis=1
                )
                weight = tl.exp2(difference - maximum_difference[:, None])
            else:
                count = tl.maximum(tl.sum(valid_2d.to(tl.float32), axis=1), 1.0)
                mean = tl.sum(tl.where(valid_2d, local_norm, 0.0), axis=1) / count
                deviation = local_norm - mean[:, None]
                variance = tl.sum(
                    tl.where(valid_2d, deviation * deviation, 0.0), axis=1
                ) / count
                z_score = deviation * tl.rsqrt(variance[:, None] + 1.0e-6)
                weight = tl.exp2((2.0 * ALPHA * log2e) * z_score)

            used = (candidate_ids == anchor) & valid_2d
            seed0 = tl.sum(tl.where(used, rows_2d, 0), axis=1)
            seeds = seed0[:, None] * (seed_slots == 0).to(tl.int32)
            basis_source = tl.sum(
                tl.where(used[:, :, None], residual, 0.0), axis=1
            )
            for step in tl.static_range(1, 4):
                basis = basis_source * tl.rsqrt(
                    tl.sum(basis_source * basis_source, axis=1) + 1.0e-12
                )[:, None]
                projection = tl.sum(residual * basis[:, None, :], axis=2)
                residual -= projection[:, :, None] * basis[:, None, :]
                objective = tl.sum(residual * residual, axis=2) * weight
                objective = tl.where(valid_2d & ~used, objective, -float("inf"))
                _, picked = tl.max(
                    objective,
                    axis=1,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                hit = (candidate_ids == picked[:, None]) & valid_2d
                used |= hit
                row = tl.sum(tl.where(hit, rows_2d, 0), axis=1)
                seeds += row[:, None] * (seed_slots == step).to(tl.int32)
                basis_source = tl.sum(
                    tl.where(hit[:, :, None], residual, 0.0), axis=1
                )
            output_offset = (
                (((((batch * Q_OUTER + query_outer) * K_OUTER + key_outer)
                    * ROWS + key_row) * COLS + output_columns[:, None])
                 * HEADS + head) * 4 + seed_slots
            )
            tl.store(
                seed_ptr + output_offset,
                seeds,
                mask=output_columns[:, None] < COLS,
            )


def _check_inputs(q, k):
    if triton is None:
        raise RuntimeError("Triton is required for entropy-mass seed selection")
    if q.ndim != 6 or k.ndim != 6:
        raise ValueError("q and k must have shape [B,T,row,column,H,D]")
    if not q.is_cuda or not k.is_cuda or q.device != k.device:
        raise ValueError("q and k must be on the same CUDA device")
    if q.dtype not in {torch.bfloat16, torch.float16, torch.float32} or q.dtype != k.dtype:
        raise TypeError("q/k must use the same float16, bfloat16, or float32 dtype")
    if q.shape[1] != 1 or tuple(q.shape[2:4]) != (45, 80):
        raise ValueError("entropy-mass routing requires one 45x80 query frame")
    if k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]:
        raise ValueError("q and k must share batch, grid, head, and dimension sizes")
    if k.shape[1] < 1 or not 1 <= q.shape[-1] <= 128:
        raise ValueError("require at least one key frame and head dimension <= 128")


def global_probe_log_norm_triton(
    q,
    k,
    scale=None,
    *,
    global_probe_count=256,
    block_q=32,
    num_warps=8,
    num_stages=1,
    native_low_precision_dot=False,
    tf32_low_precision_dot=False,
):
    """Return global probe log2-partitions as contiguous ``[B,1,R,C,H]``.

    Probe token indices are exactly ``arange(P) * (N // P)``.  The normalizer
    omits the common quadrature factor N/P, which cancels when block-mass
    weights are normalized across candidates.  FP32 TF32x3 QK accumulation
    avoids applying another low-precision quantization to the input tensors.
    The opt-in native path uses the original BF16/FP16 inputs with FP32
    accumulation. The alternative TF32 path casts BF16/FP16 to FP32 before
    using one TF32 dot. These paths are mutually exclusive and neither
    promises bitwise parity: MMA lowering, accumulation, and exceptional or
    subnormal values require validation. Float32 inputs always retain
    TF32x3, including when either optimization is opted in.
    """
    _check_inputs(q, k)
    if native_low_precision_dot and tf32_low_precision_dot:
        raise ValueError("native_low_precision_dot and tf32_low_precision_dot are mutually exclusive")
    q, k = q.contiguous(), k.contiguous()
    batch, _, rows, columns, heads, head_dim = q.shape
    key_tokens = k.shape[1] * rows * columns
    probes = int(global_probe_count)
    if not 1 <= probes <= min(key_tokens, 1024):
        raise ValueError("global_probe_count must be between 1 and min(N,1024)")
    block_q = int(block_q)
    if block_q not in {16, 32, 64}:
        raise ValueError("block_q must be 16, 32, or 64")
    normalizer = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    _global_probe_log_norm_kernel[(triton.cdiv(rows * columns, block_q), batch * heads)](
        q, k, normalizer,
        QUERY_TOKENS=rows * columns,
        KEY_TOKENS=key_tokens,
        HEADS=heads,
        D_HEAD=head_dim,
        PROBES=probes,
        BLOCK_Q=block_q,
        BLOCK_P=max(16, triton.next_power_of_2(probes)),
        BLOCK_D=max(32, triton.next_power_of_2(head_dim)),
        SCALE=head_dim ** -0.5 if scale is None else float(scale),
        NATIVE_LOW_PRECISION_DOT=(
            bool(native_low_precision_dot)
            and q.dtype in {torch.bfloat16, torch.float16}
        ),
        TF32_LOW_PRECISION_DOT=(
            bool(tf32_low_precision_dot)
            and q.dtype in {torch.bfloat16, torch.float16}
        ),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return normalizer


def entropy_mass_seed_rows_triton(
    q,
    k,
    scale=None,
    *,
    candidate_count=32,
    candidate_geometry_count=45,
    block_j=4,
    probe_count=16,
    global_probe_count=256,
    output_dtype=torch.uint8,
    gs_num_warps=4,
    weight_mode="mass",
    alpha=0.2,
    gs_num_stages=1,
    norm_block_q=32,
    norm_num_warps=8,
    native_low_precision_dot=False,
    tf32_low_precision_dot=False,
    singleton_complement=False,
):
    """Select real C4 query rows with globally normalized block-mass weights.

    Input layouts are ``q[B,1,45,80,H,D]`` and ``k[B,T,45,80,H,D]``;
    output is ``[B,1,T,45,80,H,4]``.  The mass mode does not use ``alpha``.
    ``weight_mode='zscore'`` is only a parity diagnostic for the existing
    Hellinger selector and skips the global-normalizer calculation.
    ``native_low_precision_dot`` and ``tf32_low_precision_dot`` change only
    the global normalizer's dot implementation, are mutually exclusive, and
    default off to preserve existing inference behavior.
    ``singleton_complement`` excludes a sampled key whose column equals the
    query column before local softmax/LSE, then centers the square-root
    profile on its remaining support. The global normalizer, anchor, C4
    budget and merge/route semantics are unchanged. This is a probe proxy
    for the SQ1 bulk complement, not an exact full-key partition objective.
    """
    _check_inputs(q, k)
    if native_low_precision_dot and tf32_low_precision_dot:
        raise ValueError("native_low_precision_dot and tf32_low_precision_dot are mutually exclusive")
    candidate_count = int(candidate_count)
    candidate_geometry_count = int(candidate_geometry_count)
    if not 4 <= candidate_count <= candidate_geometry_count <= q.shape[2]:
        raise ValueError("require 4 <= candidate_count <= geometry_count <= 45")
    block_j = int(block_j)
    if block_j not in {1, 2, 4, 8}:
        raise ValueError("block_j must be 1, 2, 4, or 8")
    probe_count = int(probe_count)
    if singleton_complement and not 2 <= probe_count <= q.shape[3]:
        raise ValueError("singleton complement requires 2 <= probe_count <= columns")
    if probe_count != 16:
        raise ValueError("the current entropy-mass selector uses Uniform16 probes")
    if output_dtype not in {torch.uint8, torch.int32}:
        raise TypeError("output_dtype must be uint8 or int32")
    if weight_mode not in {"mass", "zscore"}:
        raise ValueError("weight_mode must be 'mass' or 'zscore'")

    from .adaptive_seed_pytorch import multiscale_candidate_table

    q, k = q.contiguous(), k.contiguous()
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if weight_mode == "mass":
        normalizer = global_probe_log_norm_triton(
            q, k, scale,
            global_probe_count=global_probe_count,
            block_q=norm_block_q,
            num_warps=norm_num_warps,
            native_low_precision_dot=native_low_precision_dot,
            tf32_low_precision_dot=tf32_low_precision_dot,
        )
    else:
        normalizer = torch.empty((1,), device=q.device, dtype=torch.float32)
    candidate_rows, anchors = multiscale_candidate_table(
        q.shape[2], candidate_count, q.device, geometry_count=candidate_geometry_count
    )
    batch, query_outer, rows, columns, heads, head_dim = q.shape
    key_outer = k.shape[1]
    output = torch.empty(
        (batch, query_outer, key_outer, rows, columns, heads, 4),
        device=q.device, dtype=output_dtype,
    )
    groups = triton.cdiv(columns, block_j)
    _entropy_mass_seed_kernel[(query_outer * rows * groups, batch * heads)](
        q, k, normalizer, candidate_rows, anchors, output,
        Q_OUTER=query_outer,
        K_OUTER=key_outer,
        ROWS=rows,
        COLS=columns,
        HEADS=heads,
        D_HEAD=head_dim,
        CANDIDATES=candidate_count,
        BLOCK_ROWS=max(16, triton.next_power_of_2(candidate_count)),
        PROBES=probe_count,
        BLOCK_P=16,
        BLOCK_J=block_j,
        BLOCK_D=max(32, triton.next_power_of_2(head_dim)),
        N_GROUPS=groups,
        SCALE=scale,
        MASS_WEIGHT=weight_mode == "mass",
        ALPHA=float(alpha),
        SINGLETON_COMPLEMENT=bool(singleton_complement),
        num_warps=int(gs_num_warps),
        num_stages=int(gs_num_stages),
    )
    return output


__all__ = ["entropy_mass_seed_rows_triton", "global_probe_log_norm_triton"]
