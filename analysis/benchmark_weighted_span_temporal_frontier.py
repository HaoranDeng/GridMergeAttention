"""Benchmark temporal weighted-span selectors on captured Wan QKV."""

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
    span_seed_rows_triton_split,
    span_seed_rows_triton_temporal_consensus2,
)
from sweep_simple_weighted_span_qkv import COLS, FRAMES, ROWS, _dense_frame, _parse_indices
from sweep_simple_weighted_span_selectors import evaluate_rows


def _load_grid(path: str, heads: list[int]) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def _router_frames(count: int) -> list[int]:
    if count == 1:
        return [10]
    if count == 4:
        return [0, 7, 13, 20]
    return [
        round((slot + 0.5) * FRAMES / count - 0.5)
        for slot in range(count)
    ]


def _parse_ints(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,5,10,15,20")
    parser.add_argument("--temporal_counts", default="1,2,4")
    parser.add_argument("--probe_counts", default="4,8,16")
    parser.add_argument("--value_dims", default="8,16,32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--candidate_count", type=int, default=45)
    parser.add_argument("--block_j", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timing_heads", type=int, default=0)
    parser.add_argument("--consensus", action="store_true")
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    target_frames = _parse_indices(args.frames)
    temporal_counts = _parse_ints(args.temporal_counts)
    if args.consensus and temporal_counts != [2]:
        raise ValueError("consensus benchmark requires --temporal_counts 2")
    probe_counts = _parse_ints(args.probe_counts)
    value_dims = _parse_ints(args.value_dims)
    variants = [
        (temporal_count, probe_count, dims)
        for temporal_count in temporal_counts
        for probe_count in probe_counts
        for dims in value_dims
    ]
    totals = {
        f"t{t}_p{p}_v{d}": {
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "numel": 0,
            "points": 0,
            "selector_ms": [],
        }
        for t, p, d in variants
    }

    for capture in args.qkv:
        q_grid, k_grid, v_grid = _load_grid(capture, heads)
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        v_grid = v_grid.to(args.device)
        production_k = k_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        production_v = v_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        key_blocks = k_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)
        value_blocks = v_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)
        dense = {
            (head_slot, frame): _dense_frame(
                q_grid[head_slot], k_grid[head_slot], v_grid[head_slot], frame
            )
            for head_slot in range(len(heads))
            for frame in target_frames
        }

        for temporal_count, probe_count, dims in variants:
            name = f"t{temporal_count}_p{probe_count}_v{dims}"
            routers = _router_frames(temporal_count)
            router_index = {
                frame: min(
                    range(len(routers)),
                    key=lambda index: abs(frame - routers[index]),
                )
                for frame in target_frames
            }
            production_q = (
                q_grid[:, routers]
                .permute(1, 2, 3, 0, 4)
                .unsqueeze(0)
                .contiguous()
            )

            def select(q_input, k_input, v_input) -> torch.Tensor:
                selector = (
                    span_seed_rows_triton_temporal_consensus2
                    if args.consensus
                    else span_seed_rows_triton_split
                )
                return selector(
                    q_input,
                    k_input,
                    v_input,
                    alpha=args.alpha,
                    candidate_count=args.candidate_count,
                    top_probes=probe_count // 2,
                    uniform_probes=probe_count // 2,
                    value_dims=dims,
                    block_j=args.block_j,
                )

            timing_q = production_q
            timing_k = production_k
            timing_v = production_v
            if args.timing_heads:
                current_heads = production_q.size(4)
                if args.timing_heads % current_heads:
                    raise ValueError("timing_heads must be a multiple of loaded heads")
                repeat_heads = args.timing_heads // current_heads
                timing_q = production_q.repeat(1, 1, 1, 1, repeat_heads, 1)
                timing_k = production_k.repeat(1, 1, 1, 1, repeat_heads, 1)
                timing_v = production_v.repeat(1, 1, 1, 1, repeat_heads, 1)
            for _ in range(args.warmup):
                select(timing_q, timing_k, timing_v)
            torch.cuda.synchronize()
            for _ in range(args.repeats):
                started = time.perf_counter()
                select(timing_q, timing_k, timing_v)
                torch.cuda.synchronize()
                totals[name]["selector_ms"].append(
                    1000.0 * (time.perf_counter() - started)
                )
            table = select(production_q, production_k, production_v)

            for head_slot in range(len(heads)):
                for frame in target_frames:
                    table_slot = 0 if args.consensus else router_index[frame]
                    rows = (
                        table[0, table_slot, ..., head_slot, :]
                        .reshape(FRAMES * ROWS, COLS, 4)
                        .permute(1, 0, 2)
                        .contiguous()
                    )
                    metrics = evaluate_rows(
                        q_grid[head_slot, frame].permute(1, 0, 2).contiguous(),
                        key_blocks[head_slot],
                        value_blocks[head_slot],
                        dense[(head_slot, frame)],
                        rows,
                    )
                    totals[name]["squared_error"] += metrics["squared_error"]
                    totals[name]["reference_squared"] += metrics["reference_squared"]
                    totals[name]["numel"] += metrics["numel"]
                    totals[name]["points"] += 1

    result = {"settings": vars(args), "variants": {}}
    for name, total in totals.items():
        timings = total.pop("selector_ms")
        result["variants"][name] = {
            "selector_ms": sum(timings) / len(timings),
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "points": total["points"],
        }
    result["ranking"] = sorted(
        result["variants"],
        key=lambda name: (
            result["variants"][name]["rel_l2_vs_dense"],
            result["variants"][name]["selector_ms"],
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["variants"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
