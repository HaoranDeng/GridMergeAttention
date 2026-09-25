"""Measure per-frame Weighted-Span seed sharing across attention heads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import span_seed_rows_triton_split
from sweep_simple_weighted_span_qkv import COLS, FRAMES, ROWS, _dense_frame, _parse_indices
from sweep_simple_weighted_span_selectors import evaluate_rows


def _load_grid(path: str) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0].copy()).reshape(
                payload[name].shape[1], FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def _production(grid: torch.Tensor) -> torch.Tensor:
    return grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()


def _select(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return span_seed_rows_triton_split(
        q,
        k,
        v,
        alpha=0.3,
        candidate_count=45,
        top_probes=8,
        uniform_probes=8,
        value_dims=8,
        block_j=4,
        group_tile=16,
        key_outer_tile=7,
        output_dtype=torch.uint8,
        probe_num_warps=4,
        gs_num_warps=8,
    )


def _rows(table: torch.Tensor, frame: int, head: int) -> torch.Tensor:
    return (
        table[0, frame, ..., head, :]
        .reshape(FRAMES * ROWS, COLS, 4)
        .permute(1, 0, 2)
        .contiguous()
        .long()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", default="2,4,5,8,10")
    parser.add_argument("--representative_offsets", default="")
    parser.add_argument("--frames", default="0,5,10,15,20")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    groups = [int(value) for value in args.groups.split(",") if value]
    requested_offsets = [
        int(value) for value in args.representative_offsets.split(",") if value
    ]
    frames = _parse_indices(args.frames)
    q_grid, k_grid, v_grid = (
        tensor.to(args.device) for tensor in _load_grid(args.qkv)
    )
    heads = q_grid.size(0)
    q, k, v = (_production(tensor) for tensor in (q_grid, k_grid, v_grid))
    own_table = _select(q, k, v)

    variants = {"own": own_table}
    head_maps = {"own": list(range(heads))}
    for group in groups:
        if heads % group:
            raise ValueError(f"head count {heads} is not divisible by group {group}")
        group_count = heads // group
        offsets = requested_offsets or [group // 2]
        for offset in offsets:
            if not 0 <= offset < group:
                continue
            representative = torch.arange(
                offset, heads, group, device=q.device, dtype=torch.long
            )
            name = f"representative_g{group}_o{offset}"
            variants[name] = _select(
                q.index_select(4, representative),
                k.index_select(4, representative),
                v.index_select(4, representative),
            )
            head_maps[name] = [head // group for head in range(heads)]

        averaged = []
        for tensor in (q, k, v):
            averaged.append(
                tensor.reshape(
                    *tensor.shape[:4], group_count, group, tensor.shape[-1]
                ).mean(dim=5)
            )
        variants[f"mean_g{group}"] = _select(*averaged)
        head_maps[f"mean_g{group}"] = [head // group for head in range(heads)]

    totals = {
        name: {
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "numel": 0,
            "points": 0,
            "seed_matches": 0,
            "seed_numel": 0,
        }
        for name in variants
    }
    key_blocks = k_grid.reshape(heads, FRAMES * ROWS, COLS, -1)
    value_blocks = v_grid.reshape(heads, FRAMES * ROWS, COLS, -1)
    for head in range(heads):
        for frame in frames:
            dense = _dense_frame(q_grid[head], k_grid[head], v_grid[head], frame)
            own_rows = _rows(own_table, frame, head)
            for name, table in variants.items():
                rows = _rows(table, frame, head_maps[name][head])
                metrics = evaluate_rows(
                    q_grid[head, frame].permute(1, 0, 2).contiguous(),
                    key_blocks[head],
                    value_blocks[head],
                    dense,
                    rows,
                )
                total = totals[name]
                total["squared_error"] += metrics["squared_error"]
                total["reference_squared"] += metrics["reference_squared"]
                total["numel"] += metrics["numel"]
                total["points"] += 1
                total["seed_matches"] += int(torch.count_nonzero(rows == own_rows).item())
                total["seed_numel"] += rows.numel()

    result = {"settings": vars(args), "variants": {}}
    for name, total in totals.items():
        result["variants"][name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "seed_slot_agreement_vs_own": (
                total["seed_matches"] / total["seed_numel"]
            ),
            "points": total["points"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
