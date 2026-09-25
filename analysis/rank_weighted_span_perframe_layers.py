"""Rank Wan layers by the benefit of per-frame Weighted-Span routing."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import span_seed_rows_triton_split
from benchmark_weighted_span_temporal_frontier import _load_grid
from sweep_simple_weighted_span_qkv import COLS, FRAMES, ROWS, _dense_frame
from sweep_simple_weighted_span_selectors import evaluate_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--frames", default="0,5,10,15,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--candidate_count", type=int, default=45)
    parser.add_argument("--probe_count", type=int, default=16)
    parser.add_argument("--value_dims", type=int, default=8)
    parser.add_argument("--block_j", type=int, default=4)
    return parser.parse_args()


def evaluate_variant(
    q_grid, k_grid, v_grid, frames, router_frames, dense, key_blocks,
    value_blocks, args,
):
    production_q = (
        q_grid[:, router_frames]
        .permute(1, 2, 3, 0, 4)
        .unsqueeze(0)
        .contiguous()
    )
    production_k = k_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
    production_v = v_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
    table = span_seed_rows_triton_split(
        production_q,
        production_k,
        production_v,
        alpha=args.alpha,
        candidate_count=args.candidate_count,
        top_probes=args.probe_count // 2,
        uniform_probes=args.probe_count // 2,
        value_dims=args.value_dims,
        block_j=args.block_j,
        key_outer_tile=7,
        output_dtype=torch.uint8,
        probe_num_warps=4,
        gs_num_warps=8,
    )
    totals = {"squared_error": 0.0, "reference_squared": 0.0, "numel": 0}
    for frame in frames:
        router_slot = min(
            range(len(router_frames)),
            key=lambda index: abs(frame - router_frames[index]),
        )
        rows = (
            table[0, router_slot, ..., 0, :]
            .reshape(FRAMES * ROWS, COLS, 4)
            .permute(1, 0, 2)
            .contiguous()
            .long()
        )
        metrics = evaluate_rows(
            q_grid[0, frame].permute(1, 0, 2).contiguous(),
            key_blocks,
            value_blocks,
            dense[frame],
            rows,
        )
        for key in totals:
            totals[key] += metrics[key]
    return {
        "relative_l2": math.sqrt(
            totals["squared_error"] / max(totals["reference_squared"], 1e-20)
        ),
        "mse": totals["squared_error"] / totals["numel"],
    }


def main():
    args = parse_args()
    frames = [int(value) for value in args.frames.split(",") if value]
    captures = sorted(Path(args.qkv_dir).glob("call*_qkv.npz"))
    records = []
    for layer, capture in enumerate(captures):
        q_grid, k_grid, v_grid = _load_grid(str(capture), [args.head])
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        v_grid = v_grid.to(args.device)
        dense = {
            frame: _dense_frame(q_grid[0], k_grid[0], v_grid[0], frame)
            for frame in frames
        }
        key_blocks = k_grid[0].reshape(FRAMES * ROWS, COLS, -1)
        value_blocks = v_grid[0].reshape(FRAMES * ROWS, COLS, -1)
        shared = evaluate_variant(
            q_grid, k_grid, v_grid, frames, [10], dense, key_blocks,
            value_blocks, args,
        )
        per_frame = evaluate_variant(
            q_grid, k_grid, v_grid, frames, list(range(FRAMES)), dense,
            key_blocks, value_blocks, args,
        )
        record = {
            "layer": layer,
            "capture": capture.name,
            "shared": shared,
            "per_frame": per_frame,
            "relative_l2_gain": shared["relative_l2"] - per_frame["relative_l2"],
            "mse_ratio": per_frame["mse"] / max(shared["mse"], 1e-20),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    ranking = sorted(
        records,
        key=lambda record: record["relative_l2_gain"],
        reverse=True,
    )
    result = {"settings": vars(args), "records": records, "ranking": ranking}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
