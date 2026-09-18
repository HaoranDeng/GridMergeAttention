"""Benchmark exact 720p C4 merge loop-scheduling hints on Hopper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention.hard_c4_wan21_14b_720p_triton import (
    HARD_720_C,
    HARD_720_COLS,
    HARD_720_INPUT_PRECISION,
    HARD_720_MERGE_BLOCK_J,
    HARD_720_MERGE_LOOP_STAGES,
    HARD_720_MERGE_NUM_STAGES,
    HARD_720_MERGE_NUM_WARPS,
    HARD_720_QUERY_OUTER_GROUP,
    HARD_720_ROWS,
    _hard_c4_720p_merge_kernel,
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


def metadata(compiled):
    result = {}
    compiled_metadata = getattr(compiled, "metadata", None)
    for name in ("shared", "num_warps", "num_stages", "tmem_size"):
        value = getattr(compiled_metadata, name, None)
        if value is not None:
            result[name] = int(value)
    for name in ("n_regs", "n_spills"):
        value = getattr(compiled, name, None)
        if value is not None:
            result[name] = int(value)
    return result


def exact_equal(left, right, rows=1 << 20):
    for start in range(0, left.size(0), rows):
        if not torch.equal(left[start : start + rows], right[start : start + rows]):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--query_outer", type=int, default=21)
    parser.add_argument("--key_outer", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--configs",
        default="00,01,10,11",
        help="Comma-separated flatten/disable-LICM boolean pairs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dim = 128
    query_length = args.query_outer * HARD_720_ROWS * HARD_720_COLS
    key_length = args.key_outer * HARD_720_ROWS * HARD_720_COLS
    q = torch.randn((1, args.heads, query_length, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, args.heads, key_length, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    compressed_tokens = (
        args.query_outer
        * HARD_720_COLS
        * args.heads
        * args.key_outer
        * HARD_720_ROWS
        * HARD_720_C
    )

    def allocate_outputs():
        merged_k = torch.empty((compressed_tokens, dim), device="cuda", dtype=q.dtype)
        merged_v = torch.empty_like(merged_k)
        bias = torch.empty(compressed_tokens, device="cuda", dtype=torch.float32)
        return merged_k, merged_v, bias

    outputs = allocate_outputs()
    reference = allocate_outputs()
    query_groups = args.query_outer // HARD_720_QUERY_OUTER_GROUP
    grid = (query_groups * args.key_outer * HARD_720_ROWS, args.heads)

    def launch(target, flatten, disable_licm):
        return _hard_c4_720p_merge_kernel[grid](
            q,
            k,
            v,
            *target,
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
            flatten,
            disable_licm,
            HARD_720_INPUT_PRECISION,
            True,
            SCALE=dim**-0.5,
            num_warps=HARD_720_MERGE_NUM_WARPS,
            num_stages=HARD_720_MERGE_NUM_STAGES,
        )

    launch(reference, False, False)
    torch.cuda.synchronize()
    variants = []
    configs = [
        (bool(int(value[0])), bool(int(value[1])))
        for value in args.configs.split(",")
    ]
    for flatten, disable_licm in configs:
        config = {"flatten": flatten, "disable_licm": disable_licm}
        try:
            compiled = launch(outputs, flatten, disable_licm)
            torch.cuda.synchronize()
            variants.append(
                {
                    **config,
                    "bitwise_equal": all(
                        exact_equal(candidate, baseline)
                        for candidate, baseline in zip(outputs, reference)
                    ),
                    "mean_ms": timed(
                        lambda f=flatten, d=disable_licm: launch(outputs, f, d),
                        args.warmup,
                        args.repeats,
                    ),
                    "compiled": metadata(compiled),
                }
            )
        except Exception as exception:
            variants.append({**config, "error": str(exception)})

    successful = [variant for variant in variants if "mean_ms" in variant]
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "shape": {"query": list(q.shape), "key": list(k.shape)},
        "warmup": args.warmup,
        "repeats": args.repeats,
        "variants": variants,
        "best": min(successful, key=lambda item: item["mean_ms"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
