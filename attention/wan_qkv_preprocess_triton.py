"""Fused Wan Q/K normalization, RoPE, and attention-layout conversion.

The kernels preserve the existing operation order used by the SVG fast path:
RMSNorm is rounded to the input dtype before RoPE uses FP64 arithmetic and
rounds back to the input dtype. Intermediate normalized Q/K tensors are not
written to HBM.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional for CPU-only tests.
    triton = None
    tl = None


WAN_QKV_ROPE_BLOCK_TOKENS = 4
WAN_QKV_ROPE_NUM_WARPS = 1


if triton is not None:

    @triton.jit
    def _rmsnorm_rstd_kernel(
        x_ptr,
        rstd_ptr,
        ROW_STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        PADDED_WIDTH: tl.constexpr,
        eps,
    ):
        token = tl.program_id(0)
        rows = token + tl.arange(0, 1)
        cols = tl.arange(0, PADDED_WIDTH)
        valid = cols[None, :] < WIDTH
        x = tl.load(
            x_ptr + rows[:, None] * ROW_STRIDE + cols[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(x * x, axis=1) / WIDTH
        rstd = 1 / tl.sqrt(variance + eps)
        tl.store(rstd_ptr + rows, rstd)

    @triton.jit
    def _qk_rmsnorm_rstd_kernel(
        q_ptr,
        k_ptr,
        q_rstd_ptr,
        k_rstd_ptr,
        Q_ROW_STRIDE: tl.constexpr,
        K_ROW_STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        PADDED_WIDTH: tl.constexpr,
        q_eps,
        k_eps,
    ):
        token = tl.program_id(0)
        rows = token + tl.arange(0, 1)
        cols = tl.arange(0, PADDED_WIDTH)
        valid = cols[None, :] < WIDTH
        q = tl.load(
            q_ptr + rows[:, None] * Q_ROW_STRIDE + cols[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + rows[:, None] * K_ROW_STRIDE + cols[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        q_variance = tl.sum(q * q, axis=1) / WIDTH
        k_variance = tl.sum(k * k, axis=1) / WIDTH
        tl.store(q_rstd_ptr + rows, 1 / tl.sqrt(q_variance + q_eps))
        tl.store(k_rstd_ptr + rows, 1 / tl.sqrt(k_variance + k_eps))

    @triton.jit
    def _normalize_rope_permute_qkv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_weight_ptr,
        k_weight_ptr,
        q_rstd_ptr,
        k_rstd_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        Q_ROW_STRIDE: tl.constexpr,
        K_ROW_STRIDE: tl.constexpr,
        V_ROW_STRIDE: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        WIDTH: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
    ):
        token = tl.program_id(0)
        dims = tl.arange(0, D_HEAD)
        q_rstd = tl.load(q_rstd_ptr + token)
        k_rstd = tl.load(k_rstd_ptr + token)
        sequence_token = token % SEQ_LEN
        batch = token // SEQ_LEN

        for head in tl.range(0, HEADS):
            flat_cols = head * D_HEAD + dims
            q = tl.load(q_ptr + token * Q_ROW_STRIDE + flat_cols).to(tl.float32)
            k = tl.load(k_ptr + token * K_ROW_STRIDE + flat_cols).to(tl.float32)
            v = tl.load(v_ptr + token * V_ROW_STRIDE + flat_cols)
            q_weight = tl.load(q_weight_ptr + flat_cols)
            k_weight = tl.load(k_weight_ptr + flat_cols)
            q_hat = q * q_rstd
            k_hat = k * k_rstd
            q = (q_hat * q_weight).to(q_ptr.type.element_ty)
            k = (k_hat * k_weight).to(k_ptr.type.element_ty)

            output_offsets = (
                (batch * HEADS + head) * SEQ_LEN + sequence_token
            ) * D_HEAD + dims
            tl.store(q_out_ptr + output_offsets, q)
            tl.store(k_out_ptr + output_offsets, k)
            tl.store(v_out_ptr + output_offsets, v)

    @triton.jit
    def _normalize_permute_rope_qkv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_weight_ptr,
        k_weight_ptr,
        q_rstd_ptr,
        k_rstd_ptr,
        rope_real_ptr,
        rope_imag_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        Q_ROW_STRIDE: tl.constexpr,
        K_ROW_STRIDE: tl.constexpr,
        V_ROW_STRIDE: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        TOTAL_TOKENS: tl.constexpr,
        WIDTH: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        tokens = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        token_mask = tokens < TOTAL_TOKENS
        pairs = tl.arange(0, D_HEAD // 2)
        dims = tl.arange(0, D_HEAD)
        even_dims = 2 * pairs
        odd_dims = even_dims + 1
        q_rstd = tl.load(q_rstd_ptr + tokens, mask=token_mask, other=0.0)
        k_rstd = tl.load(k_rstd_ptr + tokens, mask=token_mask, other=0.0)
        sequence_token = tokens % SEQ_LEN
        batch = tokens // SEQ_LEN
        rope_offsets = sequence_token[:, None] * (D_HEAD // 2) + pairs[None, :]
        rope_real = tl.load(
            rope_real_ptr + rope_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float64)
        rope_imag = tl.load(
            rope_imag_ptr + rope_offsets,
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float64)

        for head in tl.range(0, HEADS):
            head_offset = head * D_HEAD
            even_cols = head_offset + even_dims
            odd_cols = head_offset + odd_dims
            q_even = (
                tl.load(
                    q_ptr + tokens[:, None] * Q_ROW_STRIDE + even_cols[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                * q_rstd[:, None]
                * tl.load(q_weight_ptr + even_cols)[None, :]
            ).to(q_ptr.type.element_ty)
            q_odd = (
                tl.load(
                    q_ptr + tokens[:, None] * Q_ROW_STRIDE + odd_cols[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                * q_rstd[:, None]
                * tl.load(q_weight_ptr + odd_cols)[None, :]
            ).to(q_ptr.type.element_ty)
            k_even = (
                tl.load(
                    k_ptr + tokens[:, None] * K_ROW_STRIDE + even_cols[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                * k_rstd[:, None]
                * tl.load(k_weight_ptr + even_cols)[None, :]
            ).to(k_ptr.type.element_ty)
            k_odd = (
                tl.load(
                    k_ptr + tokens[:, None] * K_ROW_STRIDE + odd_cols[None, :],
                    mask=token_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                * k_rstd[:, None]
                * tl.load(k_weight_ptr + odd_cols)[None, :]
            ).to(k_ptr.type.element_ty)

            q_even_f64 = q_even.to(tl.float64)
            q_odd_f64 = q_odd.to(tl.float64)
            k_even_f64 = k_even.to(tl.float64)
            k_odd_f64 = k_odd.to(tl.float64)
            q_rot_even = q_even_f64 * rope_real - q_odd_f64 * rope_imag
            q_rot_odd = q_odd_f64 * rope_real + q_even_f64 * rope_imag
            k_rot_even = k_even_f64 * rope_real - k_odd_f64 * rope_imag
            k_rot_odd = k_odd_f64 * rope_real + k_even_f64 * rope_imag

            output_base = (
                (batch * HEADS + head) * SEQ_LEN + sequence_token
            ) * D_HEAD
            tl.store(
                q_out_ptr + output_base[:, None] + even_dims[None, :],
                q_rot_even,
                mask=token_mask[:, None],
            )
            tl.store(
                q_out_ptr + output_base[:, None] + odd_dims[None, :],
                q_rot_odd,
                mask=token_mask[:, None],
            )
            tl.store(
                k_out_ptr + output_base[:, None] + even_dims[None, :],
                k_rot_even,
                mask=token_mask[:, None],
            )
            tl.store(
                k_out_ptr + output_base[:, None] + odd_dims[None, :],
                k_rot_odd,
                mask=token_mask[:, None],
            )
            value = tl.load(
                v_ptr
                + tokens[:, None] * V_ROW_STRIDE
                + head_offset
                + dims[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            tl.store(
                v_out_ptr + output_base[:, None] + dims[None, :],
                value,
                mask=token_mask[:, None],
            )


def preprocess_qkv_head_major(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    query_eps: float,
    key_eps: float,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize Q/K and convert all QKV to contiguous head-major layout."""

    if triton is None:
        raise RuntimeError("Triton is required for fused Wan QKV preprocessing.")
    batch, sequence_length, width = query.shape
    head_dim = width // heads
    query_out = torch.empty(
        (batch, heads, sequence_length, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    key_out = torch.empty_like(query_out)
    value_out = torch.empty_like(query_out)
    query_rstd = torch.empty(
        batch * sequence_length, dtype=torch.float32, device=query.device
    )
    key_rstd = torch.empty_like(query_rstd)
    padded_width = triton.next_power_of_2(width)
    stats_grid = (batch * sequence_length,)
    _rmsnorm_rstd_kernel[stats_grid](
        query,
        query_rstd,
        query.stride(-2),
        width,
        padded_width,
        float(query_eps),
        num_warps=8,
        num_stages=1,
    )
    _rmsnorm_rstd_kernel[stats_grid](
        key,
        key_rstd,
        key.stride(-2),
        width,
        padded_width,
        float(key_eps),
        num_warps=8,
        num_stages=1,
    )
    _normalize_rope_permute_qkv_kernel[(batch * sequence_length,)](
        query,
        key,
        value,
        query_weight,
        key_weight,
        query_rstd,
        key_rstd,
        query_out,
        key_out,
        value_out,
        query.stride(-2),
        key.stride(-2),
        value.stride(-2),
        sequence_length,
        width,
        heads,
        head_dim,
        num_warps=4,
        num_stages=1,
    )
    return query_out, key_out, value_out


def preprocess_qkv_head_major_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    query_eps: float,
    key_eps: float,
    heads: int,
    rope_real: torch.Tensor,
    rope_imag: torch.Tensor,
    rope_block_tokens: int = WAN_QKV_ROPE_BLOCK_TOKENS,
    rope_num_warps: int = WAN_QKV_ROPE_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize, rotate, and convert QKV to contiguous head-major layout."""

    if triton is None:
        raise RuntimeError("Triton is required for fused Wan QKV preprocessing.")
    batch, sequence_length, width = query.shape
    head_dim = width // heads
    query_out = torch.empty(
        (batch, heads, sequence_length, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    key_out = torch.empty_like(query_out)
    value_out = torch.empty_like(query_out)
    query_rstd = torch.empty(
        batch * sequence_length, dtype=torch.float32, device=query.device
    )
    key_rstd = torch.empty_like(query_rstd)
    padded_width = triton.next_power_of_2(width)
    stats_grid = (batch * sequence_length,)
    _qk_rmsnorm_rstd_kernel[stats_grid](
        query,
        key,
        query_rstd,
        key_rstd,
        query.stride(-2),
        key.stride(-2),
        width,
        padded_width,
        float(query_eps),
        float(key_eps),
        num_warps=8,
        num_stages=1,
    )
    total_tokens = batch * sequence_length
    _normalize_permute_rope_qkv_kernel[(triton.cdiv(total_tokens, rope_block_tokens),)](
        query,
        key,
        value,
        query_weight,
        key_weight,
        query_rstd,
        key_rstd,
        rope_real,
        rope_imag,
        query_out,
        key_out,
        value_out,
        query.stride(-2),
        key.stride(-2),
        value.stride(-2),
        sequence_length,
        total_tokens,
        width,
        heads,
        head_dim,
        rope_block_tokens,
        num_warps=rope_num_warps,
        num_stages=1,
    )
    return query_out, key_out, value_out


__all__ = [
    "WAN_QKV_ROPE_BLOCK_TOKENS",
    "WAN_QKV_ROPE_NUM_WARPS",
    "preprocess_qkv_head_major",
    "preprocess_qkv_head_major_rope",
]
