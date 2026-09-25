"""Dense Triton attention maintained for Wan2.1-T2V-1.3B at 480p.

Validated config: ``configs/dense_wan_flash_attn_triton_fewstep.yaml``.
The kernel accepts dynamic sequence lengths, but its launch parameters are
maintained against this Wan configuration on H100/H200-class GPUs.
"""

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is missing.
    triton = None
    tl = None


FLASH_BLOCK_M = 128
FLASH_BLOCK_N = 64
FLASH_NUM_WARPS = 8
FLASH_NUM_STAGES = 3
FLASH_INPUT_PRECISION = "ieee"


if triton is not None:

    @triton.jit
    def _flash_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        BATCH: tl.constexpr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        HEADS: tl.constexpr,
        D_HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        group = tl.program_id(0)
        q_block = tl.program_id(1)

        head = group % HEADS
        batch = group // HEADS
        q_offsets = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offsets = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr
            + (((batch * Q_LEN + q_offsets[:, None]) * HEADS + head) * D_HEAD + dims[None, :]),
            mask=(q_offsets[:, None] < Q_LEN) & (dims[None, :] < D_HEAD),
            other=0.0,
        )

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, K_LEN, BLOCK_N):
            keys = key_start + k_offsets
            k = tl.load(
                k_ptr
                + (((batch * K_LEN + keys[:, None]) * HEADS + head) * D_HEAD + dims[None, :]),
                mask=(keys[:, None] < K_LEN) & (dims[None, :] < D_HEAD),
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * SCALE
            scores = tl.where(
                (q_offsets[:, None] < Q_LEN) & (keys[None, :] < K_LEN),
                scores,
                -float("inf"),
            )

            next_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
            probs = tl.exp2((scores - next_row_max[:, None]) * 1.4426950408889634)
            old_scale = tl.exp2((row_max - next_row_max) * 1.4426950408889634)

            v = tl.load(
                v_ptr
                + (((batch * K_LEN + keys[:, None]) * HEADS + head) * D_HEAD + dims[None, :]),
                mask=(keys[:, None] < K_LEN) & (dims[None, :] < D_HEAD),
                other=0.0,
            )
            acc = acc * old_scale[:, None] + tl.dot(probs.to(v.dtype), v)
            row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
            row_max = next_row_max

        out = acc / row_sum[:, None]
        tl.store(
            out_ptr
            + (((batch * Q_LEN + q_offsets[:, None]) * HEADS + head) * D_HEAD + dims[None, :]),
            out,
            mask=(q_offsets[:, None] < Q_LEN) & (dims[None, :] < D_HEAD),
        )


def flash_attention_triton(q, k, v, scale=None, output=None):
    """Forward-only dense FlashAttention-style Triton kernel.

    q: [B, Lq, H, D]
    k: [B, Lk, H, D]
    v: [B, Lk, H, D]

    This test kernel intentionally supports only dense non-causal attention
    without masks or dropout.
    """
    if triton is None:
        raise RuntimeError("Triton is required for flash_attention_triton.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, q_len, heads, dim = q.shape
    _, k_len, _, _ = k.shape
    if scale is None:
        scale = dim ** -0.5

    output = output if output is not None and output.is_contiguous() else torch.empty_like(q)
    grid = (batch * heads, triton.cdiv(q_len, FLASH_BLOCK_M))
    _flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        BATCH=batch,
        Q_LEN=q_len,
        K_LEN=k_len,
        HEADS=heads,
        D_HEAD=dim,
        BLOCK_M=FLASH_BLOCK_M,
        BLOCK_N=FLASH_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(dim),
        SCALE=float(scale),
        INPUT_PRECISION=FLASH_INPUT_PRECISION,
        num_warps=FLASH_NUM_WARPS,
        num_stages=FLASH_NUM_STAGES,
    )
    return output


__all__ = [
    "FLASH_BLOCK_M",
    "FLASH_BLOCK_N",
    "FLASH_INPUT_PRECISION",
    "FLASH_NUM_STAGES",
    "FLASH_NUM_WARPS",
    "flash_attention_triton",
]
