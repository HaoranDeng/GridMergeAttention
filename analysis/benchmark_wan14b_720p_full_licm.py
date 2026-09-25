"""Confirm the 720p C4 merge LICM hint in complete 40-head attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attention.hard_c4_wan21_14b_720p_triton import (
    HARD_720_COLS,
    HARD_720_ROWS,
    _run_hard_c4_720p,
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


def exact_equal(left, right, rows=4096):
    flat_left = left.reshape(-1, left.size(-1))
    flat_right = right.reshape(-1, right.size(-1))
    for start in range(0, flat_left.size(0), rows):
        if not torch.equal(
            flat_left[start : start + rows], flat_right[start : start + rows]
        ):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head_chunk", type=int, default=10)
    parser.add_argument("--query_outer", type=int, default=21)
    parser.add_argument("--key_outer", type=int, default=21)
    parser.add_argument("--configs", default="0,1,1,0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dim = 128
    query_length = args.query_outer * HARD_720_ROWS * HARD_720_COLS
    key_length = args.key_outer * HARD_720_ROWS * HARD_720_COLS
    q = torch.randn((1, args.heads, query_length, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, args.heads, key_length, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    outputs = {
        False: torch.empty((1, query_length, args.heads * dim), device="cuda", dtype=q.dtype),
        True: torch.empty((1, query_length, args.heads * dim), device="cuda", dtype=q.dtype),
    }

    def run(disable_licm):
        output = outputs[disable_licm]
        for head_start in range(0, args.heads, args.head_chunk):
            _run_hard_c4_720p(
                q,
                k,
                v,
                output=output,
                head_start=head_start,
                head_count=min(args.head_chunk, args.heads - head_start),
                head_major=True,
                output_head_major=False,
                merge_disable_licm=disable_licm,
            )
        return output

    run(False)
    run(True)
    torch.cuda.synchronize()
    bitwise_equal = exact_equal(outputs[False], outputs[True])
    variants = []
    for value in args.configs.split(","):
        disable_licm = bool(int(value))
        variants.append(
            {
                "disable_licm": disable_licm,
                "mean_ms": timed(
                    lambda flag=disable_licm: run(flag), args.warmup, args.repeats
                ),
            }
        )

    sdpa_ms = timed(
        lambda: F.scaled_dot_product_attention(q, k, v),
        args.warmup,
        args.repeats,
    )
    for variant in variants:
        variant["speedup_vs_sdpa"] = sdpa_ms / variant["mean_ms"]

    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": {"query": list(q.shape), "key": list(k.shape)},
        "head_chunk": args.head_chunk,
        "bitwise_equal": bitwise_equal,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sdpa_ms": sdpa_ms,
        "variants": variants,
        "best": min(variants, key=lambda item: item["mean_ms"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
