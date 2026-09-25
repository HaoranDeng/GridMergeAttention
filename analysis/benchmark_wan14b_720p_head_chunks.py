"""Benchmark exact 720p Hard C4 head-chunk dispatch on one H200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attention.hard_c4_wan21_14b_720p_triton import (
    HARD_720_COLS,
    HARD_720_ROWS,
    hard_c4_720p_attention_triton_head_major_to_token,
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


def exact_equal(left, right, chunk_rows=4096):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    flat_left = left.reshape(-1, left.size(-1))
    flat_right = right.reshape(-1, right.size(-1))
    for start in range(0, flat_left.size(0), chunk_rows):
        if not torch.equal(
            flat_left[start : start + chunk_rows],
            flat_right[start : start + chunk_rows],
        ):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--query_outer", type=int, default=21)
    parser.add_argument("--key_outer", type=int, default=21)
    parser.add_argument("--head_chunks", default="4,5,8,10,20")
    parser.add_argument("--reference_head_chunk", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    batch, dim = 1, 128
    query_length = args.query_outer * HARD_720_ROWS * HARD_720_COLS
    key_length = args.key_outer * HARD_720_ROWS * HARD_720_COLS
    q = torch.randn(
        (batch, args.heads, query_length, dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch, args.heads, key_length, dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    output = torch.empty(
        (batch, query_length, args.heads * dim),
        device="cuda",
        dtype=q.dtype,
    )

    def run_chunked(head_chunk):
        for head_start in range(0, args.heads, head_chunk):
            hard_c4_720p_attention_triton_head_major_to_token(
                q,
                k,
                v,
                output=output,
                head_start=head_start,
                head_count=min(head_chunk, args.heads - head_start),
            )
        return output

    run_chunked(args.reference_head_chunk)
    torch.cuda.synchronize()
    reference = output.clone()
    variants = []
    for head_chunk in [int(value) for value in args.head_chunks.split(",")]:
        torch.cuda.empty_cache()
        try:
            run_chunked(head_chunk)
            torch.cuda.synchronize()
            variants.append(
                {
                    "head_chunk": head_chunk,
                    "launches": (args.heads + head_chunk - 1) // head_chunk,
                    "bitwise_equal": exact_equal(output, reference),
                    "mean_ms": timed(
                        lambda: run_chunked(head_chunk),
                        args.warmup,
                        args.repeats,
                    ),
                }
            )
        except Exception as exception:
            variants.append({"head_chunk": head_chunk, "error": str(exception)})
        finally:
            torch.cuda.empty_cache()

    sdpa_ms = timed(
        lambda: F.scaled_dot_product_attention(q, k, v),
        args.warmup,
        args.repeats,
    )
    successful = [item for item in variants if "mean_ms" in item]
    for item in successful:
        item["speedup_vs_sdpa"] = sdpa_ms / item["mean_ms"]

    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": {
            "query": list(q.shape),
            "key": list(k.shape),
            "output": list(output.shape),
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sdpa_ms": sdpa_ms,
        "variants": variants,
        "best": min(successful, key=lambda item: item["mean_ms"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
