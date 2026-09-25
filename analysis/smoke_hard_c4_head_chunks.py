"""Check hard-C4 head chunks against the one-launch result on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention.hard_c4_wan21_1_3b_480p_triton import hard_c4_rt_attention_triton_blocks


def parse_chunks(text: str, heads: int) -> list[int]:
    chunks = []
    for item in text.replace(",", " ").split():
        chunk = int(item)
        if chunk < 1 or chunk > heads:
            raise ValueError(f"head chunk must be in [1, {heads}], got {chunk}")
        if chunk not in chunks:
            chunks.append(chunk)
    if heads not in chunks:
        chunks.append(heads)
    return chunks


def run_chunked(q, k, v, scale: float, head_chunk: int):
    heads = q.size(4)
    output = torch.empty_like(q)
    for head_start in range(0, heads, head_chunk):
        hard_c4_rt_attention_triton_blocks(
            q,
            k,
            v,
            scale,
            output=output,
            head_start=head_start,
            head_count=min(head_chunk, heads - head_start),
        )
    return output


def timed(fn, warmup: int, repeats: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query_outer", type=int, default=2)
    parser.add_argument("--key_outer", type=int, default=2)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--columns", type=int, default=52)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--chunks", default="1,2")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is required")
    torch.manual_seed(args.seed)
    q_shape = (
        args.batch,
        args.query_outer,
        args.rows,
        args.columns,
        args.heads,
        args.dim,
    )
    kv_shape = (
        args.batch,
        args.key_outer,
        args.rows,
        args.columns,
        args.heads,
        args.dim,
    )
    q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    scale = args.dim**-0.5
    chunks = parse_chunks(args.chunks, args.heads)

    reference = run_chunked(q, k, v, scale, args.heads)
    torch.cuda.synchronize()
    results = {}
    for chunk in chunks:
        output = run_chunked(q, k, v, scale, chunk)
        torch.cuda.synchronize()
        diff = output.float() - reference.float()
        ref_norm = reference.float().norm().clamp_min(1.0e-20)
        results[str(chunk)] = {
            "max_abs_vs_full_heads": float(diff.abs().max().item()),
            "rel_l2_vs_full_heads": float(diff.norm().item() / ref_norm.item()),
            "mean_ms": timed(
                lambda chunk=chunk: run_chunked(q, k, v, scale, chunk),
                args.warmup,
                args.repeats,
            ),
        }

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape_q": list(q_shape),
        "shape_kv": list(kv_shape),
        "chunks": chunks,
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

    for result in results.values():
        if result["max_abs_vs_full_heads"] != 0.0:
            raise AssertionError("head-chunk output differs from full-head output")


if __name__ == "__main__":
    main()
