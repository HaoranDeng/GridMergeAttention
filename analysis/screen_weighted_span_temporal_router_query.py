"""Screen one-table temporal router queries on captured Wan QKV."""

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


def temporal_router_query(q_grid, mode):
    """Return [head, row, column, dim] without consulting K or V."""

    if mode.startswith("fixed"):
        return q_grid[:, int(mode.removeprefix("fixed"))]

    q32 = q_grid.float()
    if mode == "mean":
        return q32.mean(dim=1).to(q_grid.dtype)

    if mode.startswith("head_"):
        if mode == "head_max_norm":
            score = q32.square().sum(dim=(2, 3, 4))
            frame = score.argmax(dim=1)
        else:
            normalized = (
                torch.nn.functional.normalize(q32, dim=-1)
                if mode == "head_medoid_cos"
                else q32
            )
            center = normalized.mean(dim=1, keepdim=True)
            if mode == "head_medoid_cos":
                center = torch.nn.functional.normalize(center, dim=-1)
                score = (normalized * center).sum(dim=(2, 3, 4))
                frame = score.argmax(dim=1)
            elif mode == "head_medoid_l2":
                score = (normalized - center).square().sum(dim=(2, 3, 4))
                frame = score.argmin(dim=1)
            else:
                raise ValueError(f"unknown router mode: {mode}")
        return q_grid[torch.arange(q_grid.size(0), device=q_grid.device), frame]

    if mode == "max_norm":
        score = q32.square().sum(dim=-1)
        frame = score.argmax(dim=1)
    elif mode == "medoid_l2":
        center = q32.mean(dim=1, keepdim=True)
        score = (q32 - center).square().sum(dim=-1)
        frame = score.argmin(dim=1)
    elif mode == "medoid_cos":
        normalized = torch.nn.functional.normalize(q32, dim=-1)
        center = torch.nn.functional.normalize(
            normalized.mean(dim=1, keepdim=True), dim=-1
        )
        score = (normalized * center).sum(dim=-1)
        frame = score.argmax(dim=1)
    else:
        raise ValueError(f"unknown router mode: {mode}")

    gather_index = frame[:, None, :, :, None].expand(
        -1, 1, -1, -1, q_grid.size(-1)
    )
    return q_grid.gather(1, gather_index).squeeze(1)


def evaluate_mode(q_grid, k_grid, v_grid, frames, dense, mode, args):
    router = temporal_router_query(q_grid, mode)
    production_q = router.permute(1, 2, 0, 3).unsqueeze(0).unsqueeze(1).contiguous()
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
    rows = (
        table[0, 0, ..., 0, :]
        .reshape(FRAMES * ROWS, COLS, 4)
        .permute(1, 0, 2)
        .contiguous()
        .long()
    )
    key_blocks = k_grid[0].reshape(FRAMES * ROWS, COLS, -1)
    value_blocks = v_grid[0].reshape(FRAMES * ROWS, COLS, -1)
    totals = {"squared_error": 0.0, "reference_squared": 0.0, "numel": 0}
    for frame in frames:
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
    modes = (
        "fixed10",
        "mean",
        "head_medoid_l2",
        "head_medoid_cos",
        "head_max_norm",
        "medoid_l2",
        "medoid_cos",
        "max_norm",
    )
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
        results = {
            mode: evaluate_mode(
                q_grid, k_grid, v_grid, frames, dense, mode, args
            )
            for mode in modes
        }
        best = min(results, key=lambda mode: results[mode]["relative_l2"])
        record = {
            "layer": layer,
            "capture": capture.name,
            "best": best,
            "results": results,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    aggregate = {}
    for mode in modes:
        aggregate[mode] = {
            "mean_relative_l2": sum(
                record["results"][mode]["relative_l2"] for record in records
            )
            / len(records),
            "wins": sum(record["best"] == mode for record in records),
        }
    result = {"settings": vars(args), "aggregate": aggregate, "records": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
