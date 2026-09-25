"""Sweep exact Triton launch parameters for Wan2.1-14B 720p Hard C4."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from attention.hard_c4_wan21_14b_720p_triton import (
    HARD_720_C,
    HARD_720_COLS,
    HARD_720_INPUT_PRECISION,
    HARD_720_MERGE_BLOCK_J,
    HARD_720_MERGE_DISABLE_LICM,
    HARD_720_QUERY_OUTER_GROUP,
    HARD_720_ROUTE_BLOCK_I,
    HARD_720_ROWS,
    _hard_c4_720p_merge_kernel,
    _hard_c4_720p_route_kernel,
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


def error(output, reference):
    difference = output.float() - reference.float()
    return {
        "rel_l2_vs_reference": float(
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference.float())
        ),
        "max_abs_vs_reference": float(difference.abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    batch, heads, query_outer, key_outer, dim = 1, 8, 21, 21, 128
    query_length = query_outer * HARD_720_ROWS * HARD_720_COLS
    key_length = key_outer * HARD_720_ROWS * HARD_720_COLS
    q = torch.randn((batch, heads, query_length, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, heads, key_length, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output = torch.empty_like(q)
    scale = dim**-0.5

    compressed_tokens = (
        batch
        * query_outer
        * HARD_720_COLS
        * heads
        * key_outer
        * HARD_720_ROWS
        * HARD_720_C
    )
    merged_k = torch.empty((compressed_tokens, dim), device="cuda", dtype=q.dtype)
    merged_v = torch.empty_like(merged_k)
    bias = torch.empty(compressed_tokens, device="cuda", dtype=torch.float32)
    query_groups = query_outer // HARD_720_QUERY_OUTER_GROUP
    merge_grid = (query_groups * key_outer * HARD_720_ROWS, batch * heads)
    route_grid = (1, batch * heads, query_outer * HARD_720_COLS)

    def merge(loop_stages, warps, stages):
        _hard_c4_720p_merge_kernel[merge_grid](
            q,
            k,
            v,
            merged_k,
            merged_v,
            bias,
            query_outer,
            query_groups,
            HARD_720_QUERY_OUTER_GROUP,
            key_outer,
            heads,
            heads,
            0,
            dim,
            True,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_MERGE_BLOCK_J,
            loop_stages,
            False,
            HARD_720_MERGE_DISABLE_LICM,
            HARD_720_INPUT_PRECISION,
            True,
            SCALE=scale,
            num_warps=warps,
            num_stages=stages,
        )

    def route(block_tokens, loop_stages, warps, stages):
        _hard_c4_720p_route_kernel[route_grid](
            q,
            merged_k,
            merged_v,
            bias,
            output,
            query_outer,
            key_outer,
            heads,
            heads,
            0,
            dim,
            True,
            True,
            HARD_720_ROWS,
            HARD_720_COLS,
            HARD_720_C,
            HARD_720_ROUTE_BLOCK_I,
            block_tokens,
            loop_stages,
            HARD_720_INPUT_PRECISION,
            SCALE=scale,
            num_warps=warps,
            num_stages=stages,
        )

    baseline_merge = (2, 4, 2)
    baseline_route = (64, 1, 4, 1)
    merge(*baseline_merge)
    route(*baseline_route)
    torch.cuda.synchronize()
    reference = output.clone()

    merge_results = []
    for loop_stages, warps, stages in itertools.product((1, 2), (2, 4, 8), (1, 2)):
        config = {"loop_stages": loop_stages, "warps": warps, "stages": stages}
        try:
            launch = lambda: merge(loop_stages, warps, stages)
            launch()
            route(*baseline_route)
            torch.cuda.synchronize()
            result = {
                **config,
                **error(output, reference),
                "mean_ms": timed(launch, args.warmup, args.repeats),
            }
        except Exception as exception:
            result = {**config, "error": str(exception)}
        merge_results.append(result)

    merge(*baseline_merge)
    torch.cuda.synchronize()
    route_results = []
    for block_tokens, warps, stages in itertools.product((32, 64, 128), (2, 4, 8), (1, 2)):
        config = {
            "block_tokens": block_tokens,
            "loop_stages": 1,
            "warps": warps,
            "stages": stages,
        }
        try:
            launch = lambda: route(block_tokens, 1, warps, stages)
            launch()
            torch.cuda.synchronize()
            result = {
                **config,
                **error(output, reference),
                "mean_ms": timed(launch, args.warmup, args.repeats),
            }
        except Exception as exception:
            result = {**config, "error": str(exception)}
        route_results.append(result)

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": {"query": list(q.shape), "key": list(k.shape)},
        "baseline": {
            "merge": {"loop_stages": 2, "warps": 4, "stages": 2},
            "route": {"block_tokens": 64, "loop_stages": 1, "warps": 4, "stages": 1},
        },
        "merge": merge_results,
        "route": route_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
