"""Validate combined first-frame and fixed-head rescue at the native 720p shape."""

from __future__ import annotations

import json
import time

import torch

from attention.core import flash_attention
from attention.token_merging import token_merging_attn


def _run(query, key, value, *, dense_heads=(), first_frame=False):
    return token_merging_attn(
        query,
        key,
        value,
        f_tied=1,
        h_reduce=1,
        w_reduce=1,
        h=45,
        w=80,
        query_outer_chunk=21,
        head_chunk=1,
        num_merge_tokens=4,
        hard_route_backend="hard_c4",
        dense_heads=dense_heads,
        first_frame_full_attention=first_frame,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native-shape runner.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    frame_tokens = 45 * 80
    sequence = 21 * frame_tokens
    shape = (1, sequence, 2, 128)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    key = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    value = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1

    torch.cuda.synchronize()
    start = time.perf_counter()
    baseline = _run(query, key, value)
    combined = _run(
        query,
        key,
        value,
        dense_heads=(1,),
        first_frame=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    dense_first = flash_attention(
        query[:, :frame_tokens],
        key,
        value,
        softmax_scale=query.size(-1) ** -0.5,
    )
    dense_head = flash_attention(
        query[:, :, 1:2],
        key[:, :, 1:2],
        value[:, :, 1:2],
        softmax_scale=query.size(-1) ** -0.5,
    )
    torch.cuda.synchronize()

    report = {
        "shape": list(shape),
        "elapsed_compile_and_two_calls_sec": elapsed,
        "first_frame_max_abs_error": float(
            (combined[:, :frame_tokens] - dense_first).abs().max().item()
        ),
        "dense_head_max_abs_error": float(
            (combined[:, :, 1:2] - dense_head).abs().max().item()
        ),
        "c4_remainder_max_abs_error": float(
            (
                combined[:, frame_tokens:, 0]
                - baseline[:, frame_tokens:, 0]
            )
            .abs()
            .max()
            .item()
        ),
        "finite": bool(torch.isfinite(combined).all().item()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
