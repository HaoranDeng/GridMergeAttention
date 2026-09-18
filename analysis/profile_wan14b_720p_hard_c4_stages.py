"""Profile the maintained Wan2.1-14B 720p Hard C4 Triton stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attention.hard_c4_wan21_14b_720p_triton import (
    HARD_720_C,
    HARD_720_COLS,
    HARD_720_INPUT_PRECISION,
    HARD_720_MERGE_BLOCK_J,
    HARD_720_MERGE_DISABLE_LICM,
    HARD_720_MERGE_LOOP_STAGES,
    HARD_720_MERGE_NUM_STAGES,
    HARD_720_MERGE_NUM_WARPS,
    HARD_720_QUERY_OUTER_GROUP,
    HARD_720_ROUTE_BLOCK_I,
    HARD_720_ROUTE_BLOCK_TOKENS,
    HARD_720_ROUTE_NUM_STAGES,
    HARD_720_ROUTE_NUM_WARPS,
    HARD_720_ROWS,
    _hard_c4_720p_merge_kernel,
    _hard_c4_720p_route_kernel,
    hard_c4_720p_attention_triton_head_major,
)


def timed(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query_outer", type=int, default=21)
    parser.add_argument("--key_outer", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--stress_stage",
        choices=("none", "merge", "route", "full"),
        default="none",
        help="Optionally repeat one stage long enough for external GPU counters.",
    )
    parser.add_argument("--stress_repeats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    batch, dim = 1, 128
    query_length = args.query_outer * HARD_720_ROWS * HARD_720_COLS
    key_length = args.key_outer * HARD_720_ROWS * HARD_720_COLS
    q = torch.randn((batch, args.heads, query_length, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, args.heads, key_length, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output = torch.empty_like(q)
    scale = dim**-0.5

    compressed_tokens = (
        batch
        * args.query_outer
        * HARD_720_COLS
        * args.heads
        * args.key_outer
        * HARD_720_ROWS
        * HARD_720_C
    )
    merged_k = torch.empty((compressed_tokens, dim), device="cuda", dtype=q.dtype)
    merged_v = torch.empty_like(merged_k)
    bias = torch.empty(compressed_tokens, device="cuda", dtype=torch.float32)
    unused_u8 = torch.empty(1, device="cuda", dtype=torch.uint8)
    unused_f32 = torch.empty(1, device="cuda", dtype=torch.float32)

    query_groups = args.query_outer // HARD_720_QUERY_OUTER_GROUP
    merge_grid = (
        query_groups * args.key_outer * HARD_720_ROWS,
        batch * args.heads,
    )
    route_grid = (1, batch * args.heads, args.query_outer * HARD_720_COLS)
    route_loop_stages = 1 if torch.version.hip is not None else 3

    def merge():
        _hard_c4_720p_merge_kernel[merge_grid](
            q,
            q,
            unused_u8,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            unused_u8,
            args.query_outer,
            query_groups,
            HARD_720_QUERY_OUTER_GROUP,
            args.key_outer,
            args.heads,
            args.heads,
            0,
            dim,
            True,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_MERGE_BLOCK_J,
            HARD_720_MERGE_LOOP_STAGES,
            False,
            HARD_720_MERGE_DISABLE_LICM,
            HARD_720_INPUT_PRECISION,
            False,
            False,
            True,
            False,
            0,
            0,
            ROUTE_BIAS_SCALE=1.0,
            SCALE=scale,
            num_warps=HARD_720_MERGE_NUM_WARPS,
            num_stages=HARD_720_MERGE_NUM_STAGES,
        )

    def route():
        _hard_c4_720p_route_kernel[route_grid](
            q,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            unused_f32,
            unused_f32,
            output,
            args.query_outer,
            args.key_outer,
            args.heads,
            args.heads,
            0,
            dim,
            True,
            True,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_ROUTE_BLOCK_I,
            HARD_720_ROUTE_BLOCK_TOKENS,
            route_loop_stages,
            HARD_720_INPUT_PRECISION,
            False,
            0,
            False,
            0,
            SCALE=scale,
            num_warps=HARD_720_ROUTE_NUM_WARPS,
            num_stages=HARD_720_ROUTE_NUM_STAGES,
        )

    merge()
    torch.cuda.synchronize()
    timings = {
        "merge_ms": timed(merge, args.warmup, args.repeats),
        "route_ms": timed(route, args.warmup, args.repeats),
        "full_ms": timed(
            lambda: hard_c4_720p_attention_triton_head_major(q, k, v, output=output),
            args.warmup,
            args.repeats,
        ),
        "sdpa_ms": timed(
            lambda: F.scaled_dot_product_attention(q, k, v),
            args.warmup,
            args.repeats,
        ),
    }
    timings["merge_percent"] = 100 * timings["merge_ms"] / timings["full_ms"]
    timings["route_percent"] = 100 * timings["route_ms"] / timings["full_ms"]
    timings["speedup_vs_sdpa"] = timings["sdpa_ms"] / timings["full_ms"]

    stress = None
    if args.stress_stage != "none" and args.stress_repeats > 0:
        stress_fn = {
            "merge": merge,
            "route": route,
            "full": lambda: hard_c4_720p_attention_triton_head_major(
                q, k, v, output=output
            ),
        }[args.stress_stage]
        stress = {
            "stage": args.stress_stage,
            "repeats": args.stress_repeats,
            "mean_ms": timed(stress_fn, 1, args.stress_repeats),
        }

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": {"query": list(q.shape), "key": list(k.shape)},
        "compressed_tokens": compressed_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "route_loop_stages": route_loop_stages,
        "timings": timings,
        "stress": stress,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
