import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from attention.hard_c4_wan21_1_3b_480p_triton import (  # noqa: E402
    HARD_RT_BLOCK_I,
    HARD_RT_BLOCK_J,
    HARD_RT_BLOCK_K,
    HARD_RT_BLOCK_L,
    HARD_RT_C,
    HARD_RT_INPUT_PRECISION,
    HARD_RT_MERGE_NUM_STAGES,
    HARD_RT_MERGE_NUM_WARPS,
    HARD_RT_ROUTE_NUM_STAGES,
    HARD_RT_ROUTE_NUM_WARPS,
    _hard_c4_route_kernel,
    hard_c4_rt_attention_triton_blocks,
    launch_hard_c4_merge,
)


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(repeats, 1))


def profile(args):
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q_shape = (args.batch, args.query_outer, args.rows, args.columns, args.heads, args.dim)
    kv_shape = (args.batch, args.key_outer, args.rows, args.columns, args.heads, args.dim)
    q = torch.randn(q_shape, device="cuda", dtype=dtype)
    k = torch.randn(kv_shape, device="cuda", dtype=dtype)
    v = torch.randn(kv_shape, device="cuda", dtype=dtype)
    scale = args.dim**-0.5
    block_d = 1 << (args.dim - 1).bit_length()

    merged_shape = (
        args.batch,
        args.query_outer,
        args.key_outer,
        args.rows,
        args.columns,
        HARD_RT_C,
        args.heads,
        args.dim,
    )
    merged_k = torch.empty(merged_shape, device="cuda", dtype=dtype)
    merged_v = torch.empty(merged_shape, device="cuda", dtype=dtype)
    bias = torch.empty(
        (
            args.batch,
            args.heads,
            args.query_outer,
            args.key_outer,
            args.rows,
            args.columns,
            HARD_RT_C,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    out = torch.empty_like(q)

    route_grid = (
        triton_cdiv(args.rows, HARD_RT_BLOCK_I),
        args.batch * args.heads,
        args.query_outer * args.columns,
    )

    def run_merge():
        launch_hard_c4_merge(
            q,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            scale,
            block_d,
        )

    def run_route():
        _hard_c4_route_kernel[route_grid](
            q,
            merged_k,
            merged_v,
            bias,
            out,
            args.batch,
            args.query_outer,
            args.key_outer,
            args.rows,
            args.columns,
            args.heads,
            args.heads,
            0,
            args.dim,
            BLOCK_I=HARD_RT_BLOCK_I,
            BLOCK_K=HARD_RT_BLOCK_K,
            BLOCK_D=block_d,
            C=HARD_RT_C,
            SCALE=float(scale),
            INPUT_PRECISION=HARD_RT_INPUT_PRECISION,
            num_warps=HARD_RT_ROUTE_NUM_WARPS,
            num_stages=HARD_RT_ROUTE_NUM_STAGES,
        )

    run_merge()
    torch.cuda.synchronize()

    result = {
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "shape_q": list(q_shape),
        "shape_kv": list(kv_shape),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "config": {
            "block_i": HARD_RT_BLOCK_I,
            "block_j": HARD_RT_BLOCK_J,
            "block_k": HARD_RT_BLOCK_K,
            "block_l": HARD_RT_BLOCK_L,
            "c": HARD_RT_C,
            "merge_num_warps": HARD_RT_MERGE_NUM_WARPS,
            "route_num_warps": HARD_RT_ROUTE_NUM_WARPS,
            "merge_num_stages": HARD_RT_MERGE_NUM_STAGES,
            "route_num_stages": HARD_RT_ROUTE_NUM_STAGES,
            "input_precision": HARD_RT_INPUT_PRECISION,
        },
        "merge_ms": timed(run_merge, args.warmup, args.repeats),
        "route_ms": timed(run_route, args.warmup, args.repeats),
        "full_ms": timed(
            lambda: hard_c4_rt_attention_triton_blocks(q, k, v, scale),
            args.warmup,
            args.repeats,
        ),
    }
    result["merge_pct_of_full"] = result["merge_ms"] / result["full_ms"]
    result["route_pct_of_full"] = result["route_ms"] / result["full_ms"]
    return result


def triton_cdiv(x, y):
    return (x + y - 1) // y


def parse_args():
    parser = argparse.ArgumentParser(description="Profile hard C4 RT Triton merge/route/full stages.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query_outer", type=int, default=10)
    parser.add_argument("--key_outer", type=int, default=21)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--columns", type=int, default=52)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    result = profile(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
