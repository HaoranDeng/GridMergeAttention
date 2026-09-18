"""Benchmark exact key-frame tiling for the per-frame Weighted-Span selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import span_seed_rows_triton_split
from sweep_simple_weighted_span_qkv import COLS, FRAMES, ROWS, _parse_indices


def _load_qkv(path: str, heads: list[int]) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        grids = []
        for name in ("q", "k", "v"):
            grid = torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            grids.append(
                grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
            )
        return tuple(grids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--timing_heads", type=int, default=40)
    parser.add_argument("--tiles", default="21,7,3,1")
    parser.add_argument("--group_tiles", default="8,16")
    parser.add_argument("--probe_warps", default="4,8")
    parser.add_argument("--gs_warps", default="4,8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    tiles = [int(value) for value in args.tiles.split(",") if value]
    group_tiles = [int(value) for value in args.group_tiles.split(",") if value]
    probe_warps = [int(value) for value in args.probe_warps.split(",") if value]
    gs_warps = [int(value) for value in args.gs_warps.split(",") if value]
    variants = [
        (tile, group_tile, probe_warp, gs_warp)
        for tile in tiles
        for group_tile in group_tiles
        for probe_warp in probe_warps
        for gs_warp in gs_warps
    ]
    q, k, v = (tensor.to(args.device) for tensor in _load_qkv(args.qkv, heads))

    def select(
        inputs: tuple[torch.Tensor, ...],
        tile: int,
        group_tile: int,
        probe_warp: int,
        gs_warp: int,
    ) -> torch.Tensor:
        return span_seed_rows_triton_split(
            *inputs,
            alpha=0.3,
            candidate_count=45,
            top_probes=8,
            uniform_probes=8,
            value_dims=8,
            block_j=4,
            key_outer_tile=tile,
            group_tile=group_tile,
            output_dtype=torch.uint8,
            probe_num_warps=probe_warp,
            gs_num_warps=gs_warp,
        )

    baseline = select((q, k, v), 21, 16, 4, 4)
    correctness = {}
    for tile, group_tile, probe_warp, gs_warp in variants:
        name = f"kt{tile}_gt{group_tile}_pw{probe_warp}_gw{gs_warp}"
        table = select((q, k, v), tile, group_tile, probe_warp, gs_warp)
        mismatches = int(torch.count_nonzero(table != baseline).item())
        correctness[name] = {
            "exact": mismatches == 0,
            "mismatches": mismatches,
            "numel": table.numel(),
        }
        del table

    if args.timing_heads % len(heads):
        raise ValueError("timing_heads must be a multiple of the loaded heads")
    head_repeats = args.timing_heads // len(heads)
    timing_inputs = tuple(
        tensor.repeat(1, 1, 1, 1, head_repeats, 1) for tensor in (q, k, v)
    )
    timings = {}
    for tile, group_tile, probe_warp, gs_warp in variants:
        name = f"kt{tile}_gt{group_tile}_pw{probe_warp}_gw{gs_warp}"
        for _ in range(args.warmup):
            output = select(
                timing_inputs, tile, group_tile, probe_warp, gs_warp
            )
            del output
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output = select(
                timing_inputs, tile, group_tile, probe_warp, gs_warp
            )
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end))
            del output
        timings[name] = {
            "mean_ms": sum(samples) / len(samples),
            "samples_ms": samples,
        }

    result = {
        "settings": vars(args),
        "correctness": correctness,
        "timings": timings,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
