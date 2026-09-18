"""Check Wan2.1-14B 720p hard C5/C6 Triton against PyTorch."""

from __future__ import annotations

import argparse
import json
import time

import torch
from einops import rearrange

from attention.hard_c5_c6_wan21_14b_720p_triton import (
    hard_c5_720p_attention_triton_head_major,
    hard_c6_720p_attention_triton_head_major,
)
from attention.hard_route_pytorch import (
    biased_sdpa,
    gather_seed_queries,
    hard_route_merge_kv_pytorch,
)


ROWS = 45
COLS = 80
QUERY_OUTER = 21


def pytorch_reference(q, k, v, c, scale):
    batch, heads, _, dim = q.shape
    key_outer = k.size(2) // (ROWS * COLS)
    q_blocks = rearrange(
        q, "b h (a i j) d -> b a i j h d", a=QUERY_OUTER, i=ROWS, j=COLS
    )
    k_blocks = rearrange(
        k, "b h (f i j) d -> b f i j h d", f=key_outer, i=ROWS, j=COLS
    )
    v_blocks = rearrange(
        v, "b h (f i j) d -> b f i j h d", f=key_outer, i=ROWS, j=COLS
    )

    key_row = torch.arange(ROWS, device=q.device).view(1, 1, ROWS, 1)
    seed_start = (key_row - c // 2).clamp(0, ROWS - c)
    seed_offset = torch.arange(c, device=q.device).view(1, 1, 1, 1, c)
    seed_rows = (seed_start.unsqueeze(-1) + seed_offset).expand(
        QUERY_OUTER, key_outer, ROWS, COLS, c
    )
    seed_queries = gather_seed_queries(q_blocks, seed_rows)
    merged_k, merged_v, bias = hard_route_merge_kv_pytorch(
        seed_queries, k_blocks, v_blocks, scale
    )

    query = rearrange(q_blocks, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_k, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_v, "b a f j k c h d -> (b h a j) (f k c) d")
    route_bias = rearrange(bias, "b h a f k j c -> (b h a j) (f k c)")
    output = biased_sdpa(query, key, value, route_bias, scale=scale)
    return rearrange(
        output,
        "(b h a j) i d -> b h (a i j) d",
        b=batch,
        h=heads,
        a=QUERY_OUTER,
        j=COLS,
    )


def timed(function, repeats):
    for _ in range(2):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, choices=(5, 6), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    dim = 128
    q = torch.randn(1, 1, QUERY_OUTER * ROWS * COLS, dim, device=device, dtype=dtype)
    k = torch.randn(1, 1, ROWS * COLS, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    scale = dim**-0.5
    kernel = (
        hard_c5_720p_attention_triton_head_major
        if args.c == 5
        else hard_c6_720p_attention_triton_head_major
    )

    triton_output = kernel(q, k, v, scale)
    reference = pytorch_reference(q, k, v, args.c, scale)
    difference = triton_output.float() - reference.float()
    result = {
        "c": args.c,
        "mean_ms": timed(lambda: kernel(q, k, v, scale), args.repeats),
        "rel_l2_vs_pytorch": float(
            difference.square().sum().sqrt() / reference.float().square().sum().sqrt()
        ),
        "mse_vs_pytorch": float(difference.square().mean()),
        "max_abs_vs_pytorch": float(difference.abs().max()),
        "finite": bool(torch.isfinite(triton_output).all()),
        "shape": list(triton_output.shape),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
