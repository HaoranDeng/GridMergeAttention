"""Training-free VMonarch attention for ordinary Wan video self-attention.

This follows the CVPR 2026 VMonarch definition: the Monarch dimensions are
``m=T`` and ``b=H*W``, two alternating updates are used, and only first-frame
queries are recomputed with dense attention at the end.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


VMONARCH_BACKEND = "vmonarch_all"
VMONARCH_BLOCK_M = 32
VMONARCH_BLOCK_N = 64
VMONARCH_NUM_WARPS = 4
VMONARCH_NUM_STAGES = 3
VMONARCH_INPUT_PRECISION = "ieee"
VMONARCH_C_R_MIN = 0.1


if triton is not None:

    @triton.jit
    def _r_update_kernel(
        a_r_ptr,
        k_ptr,
        v_ptr,
        c_r_ptr,
        a_l_ptr,
        c_l_ptr,
        y_ptr,
        BATCH: tl.constexpr,
        HEADS: tl.constexpr,
        FRAMES: tl.constexpr,
        SPATIAL: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        C_R_MIN: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        query_start = tl.program_id(0) * BLOCK_M
        group = tl.program_id(1)
        frame = group % FRAMES
        batch_head = group // FRAMES
        head = batch_head % HEADS
        batch = batch_head // HEADS

        queries = query_start + tl.arange(0, BLOCK_M)
        keys = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        query_mask = queries < SPATIAL
        dim_mask = dims < D_HEAD

        q_base = (
            (((batch * HEADS + head) * FRAMES + frame) * SPATIAL + queries[:, None])
            * D_HEAD
        )
        query = tl.load(
            a_r_ptr + q_base + dims[None, :],
            mask=query_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        c_r = tl.load(
            c_r_ptr
            + (((batch * HEADS + head) * FRAMES + frame) * SPATIAL + queries),
            mask=query_mask,
            other=1.0,
        )
        c_r = tl.maximum(c_r, C_R_MIN)

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        entropy_acc = tl.zeros((BLOCK_M,), tl.float32)
        key_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        value_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in tl.range(0, SPATIAL, BLOCK_N):
            key_indices = key_start + keys
            key_mask = key_indices < SPATIAL
            kv_base = (
                (((batch * HEADS + head) * FRAMES + frame) * SPATIAL + key_indices[:, None])
                * D_HEAD
            )
            key = tl.load(
                k_ptr + kv_base + dims[None, :],
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            value = tl.load(
                v_ptr + kv_base + dims[None, :],
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            logits = tl.dot(query, tl.trans(key), input_precision=INPUT_PRECISION)
            logits = logits / c_r[:, None] * 1.4426950408889634
            valid = query_mask[:, None] & key_mask[None, :]
            logits = tl.where(valid, logits, -float("inf"))

            block_max = tl.max(logits, axis=1)
            new_max = tl.maximum(row_max, block_max)
            new_max = tl.where(query_mask, new_max, 0.0)
            probs = tl.where(valid, tl.exp2(logits - new_max[:, None]), 0.0)
            block_sum = tl.sum(probs, axis=1)
            old_scale = tl.where(
                row_sum > 0.0, tl.exp2(row_max - new_max), 0.0
            )

            key_acc = key_acc * old_scale[:, None] + tl.dot(
                probs.to(query.dtype), key, input_precision=INPUT_PRECISION
            )
            value_acc = value_acc * old_scale[:, None] + tl.dot(
                probs.to(query.dtype), value, input_precision=INPUT_PRECISION
            )
            entropy_acc = entropy_acc * old_scale + tl.sum(
                tl.where(valid, logits * probs, 0.0), axis=1
            )
            row_sum = row_sum * old_scale + block_sum
            row_max = new_max

        safe_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
        inv_sum = tl.where(row_sum > 0.0, 1.0 / safe_sum, 0.0)
        key_acc = key_acc * inv_sum[:, None]
        value_acc = value_acc * inv_sum[:, None]
        # c_L = sum(R * log(R)); convert the online base-2 value to natural log.
        c_l = (
            entropy_acc * inv_sum - (row_max + tl.log2(safe_sum))
        ) * 0.6931471805599453
        c_l = tl.where(query_mask, c_l, 0.0)

        out_base = (
            (((batch * HEADS + head) * SPATIAL + queries[:, None]) * FRAMES + frame)
            * D_HEAD
        )
        out_mask = query_mask[:, None] & dim_mask[None, :]
        tl.store(a_l_ptr + out_base + dims[None, :], key_acc, mask=out_mask)
        tl.store(y_ptr + out_base + dims[None, :], value_acc, mask=out_mask)
        tl.store(
            c_l_ptr
            + (((batch * HEADS + head) * SPATIAL + queries) * FRAMES + frame),
            c_l,
            mask=query_mask,
        )


def _r_update_triton(a_r, key, value, c_r):
    if triton is None:
        raise RuntimeError("Triton is required for VMonarch on GPU.")
    a_r = a_r.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    c_r = c_r.contiguous()
    batch, heads, frames, spatial, dim = a_r.shape
    output_shape = (batch, heads, spatial, frames, dim)
    a_l = torch.empty(output_shape, device=a_r.device, dtype=a_r.dtype)
    y = torch.empty(output_shape, device=a_r.device, dtype=value.dtype)
    c_l = torch.empty(
        (batch, heads, spatial, frames), device=a_r.device, dtype=torch.float32
    )
    grid = (triton.cdiv(spatial, VMONARCH_BLOCK_M), batch * heads * frames)
    _r_update_kernel[grid](
        a_r,
        key,
        value,
        c_r,
        a_l,
        c_l,
        y,
        batch,
        heads,
        frames,
        spatial,
        dim,
        BLOCK_M=VMONARCH_BLOCK_M,
        BLOCK_N=VMONARCH_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(dim),
        C_R_MIN=VMONARCH_C_R_MIN,
        INPUT_PRECISION=VMONARCH_INPUT_PRECISION,
        num_warps=VMONARCH_NUM_WARPS,
        num_stages=VMONARCH_NUM_STAGES,
    )
    return a_l, c_l, y


def _l_update(q_b, a_l, c_l):
    logits = torch.matmul(q_b, a_l.transpose(-1, -2)).float()
    logits.sub_(c_l.unsqueeze(-2))
    left = torch.softmax(logits, dim=-1).to(q_b.dtype)
    c_r = left.float().sum(dim=-2).permute(0, 1, 3, 2).contiguous()
    a_r = torch.matmul(left.transpose(-1, -2), q_b)
    a_r = a_r.permute(0, 1, 3, 2, 4).contiguous()
    return left, a_r, c_r


def vmonarch_attention_reference(
    query,
    key,
    value,
    grid_size,
    *,
    iterations=2,
    first_frame_full_attention=True,
):
    """Readable VMonarch reference used by CPU correctness tests."""

    frames, height, width = (int(x) for x in grid_size)
    batch, sequence, heads, dim = query.shape
    spatial = height * width
    if sequence != frames * spatial:
        raise ValueError("grid_size does not match the QKV sequence length.")
    scale_sqrt = dim**-0.25
    q = (
        query.reshape(batch, frames, spatial, heads, dim)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        * scale_sqrt
    )
    k = (
        key.reshape(batch, frames, spatial, heads, dim)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        * scale_sqrt
    )
    v = (
        value.reshape(batch, frames, spatial, heads, dim)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    q_b = q.permute(0, 1, 3, 2, 4).contiguous()
    a_r = q
    c_r = torch.ones(
        (batch, heads, frames, spatial), device=query.device, dtype=torch.float32
    )
    left = None
    y = None
    for _ in range(int(iterations)):
        scores = torch.matmul(a_r.float(), k.float().transpose(-1, -2))
        scores.div_(c_r.clamp_min(VMONARCH_C_R_MIN).unsqueeze(-1))
        right = torch.softmax(scores, dim=-1).to(query.dtype)
        c_l = (right.float() * right.float().clamp_min(1e-30).log()).sum(dim=-1)
        c_l = c_l.permute(0, 1, 3, 2).contiguous()
        a_l = torch.matmul(right, k).permute(0, 1, 3, 2, 4).contiguous()
        y = torch.matmul(right, v).permute(0, 1, 3, 2, 4).contiguous()
        left, a_r, c_r = _l_update(q_b, a_l, c_l)
    output = torch.matmul(left, y).permute(0, 3, 2, 1, 4).contiguous()
    output = output.reshape(batch, sequence, heads, dim)
    if first_frame_full_attention:
        q0 = query[:, :spatial].transpose(1, 2)
        full = F.scaled_dot_product_attention(
            q0, key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0
        )
        output[:, :spatial] = full.transpose(1, 2)
    return output


def vmonarch_attention(
    query,
    key,
    value,
    grid_size,
    *,
    iterations=2,
    head_chunk=1,
    first_frame_full_attention=True,
):
    """Run memory-bounded training-free VMonarch on ``[B,S,H,D]`` QKV."""

    if triton is None:
        return vmonarch_attention_reference(
            query,
            key,
            value,
            grid_size,
            iterations=iterations,
            first_frame_full_attention=first_frame_full_attention,
        )
    frames, height, width = (int(x) for x in grid_size)
    batch, sequence, heads, dim = query.shape
    spatial = height * width
    if sequence != frames * spatial:
        raise ValueError("grid_size does not match the QKV sequence length.")
    if int(iterations) < 1:
        raise ValueError("VMonarch requires at least one alternating iteration.")

    output = torch.empty_like(query)
    scale_sqrt = math.sqrt(dim**-0.5)
    head_chunk = max(1, int(head_chunk))
    for head_start in range(0, heads, head_chunk):
        head_end = min(head_start + head_chunk, heads)
        q = (
            query[:, :, head_start:head_end]
            .reshape(batch, frames, spatial, head_end - head_start, dim)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        k = (
            key[:, :, head_start:head_end]
            .reshape(batch, frames, spatial, head_end - head_start, dim)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        v = (
            value[:, :, head_start:head_end]
            .reshape(batch, frames, spatial, head_end - head_start, dim)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        # Do not scale in-place: singleton batch/head dimensions can make the
        # contiguous reshape above alias the caller's Q/K tensors.
        q = q * scale_sqrt
        k = k * scale_sqrt
        q_b = q.permute(0, 1, 3, 2, 4).contiguous()
        a_r = q
        c_r = torch.ones(
            (batch, head_end - head_start, frames, spatial),
            device=query.device,
            dtype=torch.float32,
        )
        left = None
        y = None
        for _ in range(int(iterations)):
            a_l, c_l, y = _r_update_triton(a_r, k, v, c_r)
            left, a_r, c_r = _l_update(q_b, a_l, c_l)
        chunk_output = torch.matmul(left, y)
        chunk_output = chunk_output.permute(0, 3, 2, 1, 4).contiguous()
        chunk_output = chunk_output.reshape(
            batch, sequence, head_end - head_start, dim
        )
        if first_frame_full_attention:
            dense_first = F.scaled_dot_product_attention(
                query[:, :spatial, head_start:head_end].transpose(1, 2),
                key[:, :, head_start:head_end].transpose(1, 2),
                value[:, :, head_start:head_end].transpose(1, 2),
                dropout_p=0.0,
            )
            chunk_output[:, :spatial] = dense_first.transpose(1, 2)
        output[:, :, head_start:head_end] = chunk_output
    return output


__all__ = [
    "VMONARCH_BACKEND",
    "vmonarch_attention",
    "vmonarch_attention_reference",
]
