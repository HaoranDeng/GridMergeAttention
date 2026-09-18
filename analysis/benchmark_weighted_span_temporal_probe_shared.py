"""Benchmark temporal-probe per-frame Weighted-Span on captured Wan QKV."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import (
    span_seed_rows_triton_temporal_probes_per_frame,
)
from sweep_simple_weighted_span_qkv import (
    COLS,
    FRAMES,
    ROWS,
    _dense_frame,
    _parse_indices,
)
from sweep_simple_weighted_span_selectors import evaluate_rows


def _load_grid(path: str, heads: list[int]) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,5,10,15,20")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--candidate_count", type=int, default=45)
    parser.add_argument("--probe_count", type=int, default=16)
    parser.add_argument("--value_dims", type=int, default=8)
    parser.add_argument("--block_j", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timing_heads", type=int, default=40)
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    target_frames = _parse_indices(args.frames)
    total_squared_error = 0.0
    total_reference_squared = 0.0
    total_numel = 0
    points = 0
    selector_timings = []

    for capture in args.qkv:
        q_grid, k_grid, v_grid = _load_grid(capture, heads)
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        v_grid = v_grid.to(args.device)
        production_q = q_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        production_k = k_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        production_v = v_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        key_blocks = k_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)
        value_blocks = v_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)

        def select(q_input, k_input, v_input) -> torch.Tensor:
            return span_seed_rows_triton_temporal_probes_per_frame(
                q_input,
                k_input,
                v_input,
                alpha=args.alpha,
                candidate_count=args.candidate_count,
                top_probes=args.probe_count // 2,
                uniform_probes=args.probe_count // 2,
                value_dims=args.value_dims,
                block_j=args.block_j,
            )

        timing_q = production_q
        timing_k = production_k
        timing_v = production_v
        if args.timing_heads:
            current_heads = production_q.size(4)
            if args.timing_heads % current_heads:
                raise ValueError("timing_heads must be a multiple of loaded heads")
            repeats = args.timing_heads // current_heads
            timing_q = production_q.repeat(1, 1, 1, 1, repeats, 1)
            timing_k = production_k.repeat(1, 1, 1, 1, repeats, 1)
            timing_v = production_v.repeat(1, 1, 1, 1, repeats, 1)
        for _ in range(args.warmup):
            select(timing_q, timing_k, timing_v)
        torch.cuda.synchronize()
        for _ in range(args.repeats):
            started = time.perf_counter()
            select(timing_q, timing_k, timing_v)
            torch.cuda.synchronize()
            selector_timings.append(1000.0 * (time.perf_counter() - started))

        table = select(production_q, production_k, production_v)
        for head_slot in range(len(heads)):
            for frame in target_frames:
                dense = _dense_frame(
                    q_grid[head_slot], k_grid[head_slot], v_grid[head_slot], frame
                )
                rows = (
                    table[0, frame, :, :, :, head_slot, :]
                    .reshape(FRAMES * ROWS, COLS, 4)
                    .permute(1, 0, 2)
                    .contiguous()
                    .to(torch.long)
                )
                metrics = evaluate_rows(
                    q_grid[head_slot, frame].permute(1, 0, 2).contiguous(),
                    key_blocks[head_slot],
                    value_blocks[head_slot],
                    dense,
                    rows,
                )
                total_squared_error += metrics["squared_error"]
                total_reference_squared += metrics["reference_squared"]
                total_numel += metrics["numel"]
                points += 1

    result = {
        "settings": vars(args),
        "selector_ms": sum(selector_timings) / len(selector_timings),
        "rel_l2_vs_dense": math.sqrt(
            total_squared_error / max(total_reference_squared, 1e-20)
        ),
        "mse_vs_dense": total_squared_error / total_numel,
        "points": points,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
