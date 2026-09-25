"""Evaluate a single seed table selected from multiple temporal profiles."""

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

from sweep_simple_weighted_span_qkv import COLS, FRAMES, ROWS, _dense_frame, _parse_indices
from sweep_simple_weighted_span_selectors import (
    _greedy_span,
    _selector_features,
    evaluate_rows,
)


def _load_grid(path: str, head: int) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, head].copy()).reshape(
                FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


@torch.inference_mode()
def _select_consensus(
    q_grid: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    routers: list[int],
    weight_mode: str,
    block_chunk: int,
) -> torch.Tensor:
    chunks = []
    for begin in range(0, key.size(0), block_chunk):
        end = min(begin + block_chunk, key.size(0))
        key_rows = torch.arange(begin, end, device=key.device) % ROWS
        profiles = []
        weights = []
        candidate_rows = None
        anchor = None
        for router in routers:
            features = _selector_features(
                q_grid[router].permute(1, 0, 2).contiguous(),
                key[begin:end],
                value[begin:end],
                key_rows,
                alpha=0.3,
                candidate_count=45,
                block_j=4,
                top_probes=8,
                uniform_probes=8,
                probe_anchor_count=4,
                value_dims=32,
                attention_dim=q_grid.size(-1),
                candidate_layout="local",
            )
            profiles.append(features[0])
            weights.append(features[5])
            candidate_rows = features[6]
            anchor = features[7]
        profile = torch.cat(profiles, dim=-1)
        stacked_weight = torch.stack(weights)
        if weight_mode == "geometric":
            weight = torch.exp(torch.log(stacked_weight).mean(0))
        elif weight_mode == "mean":
            weight = stacked_weight.mean(0)
        elif weight_mode == "max":
            weight = stacked_weight.max(0).values
        else:
            raise ValueError(f"unknown weight mode {weight_mode}")
        chosen = _greedy_span(profile, weight, anchor)
        rows = candidate_rows.unsqueeze(0).expand(COLS, -1, -1).gather(-1, chosen)
        chunks.append(rows)
    return torch.cat(chunks, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--frames", default="0,2,5,8,10,12,15,18,20")
    parser.add_argument("--routers", default="5,15")
    parser.add_argument("--block_chunk", type=int, default=21)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frames = _parse_indices(args.frames)
    routers = _parse_indices(args.routers)
    names = ("geometric", "mean", "max")
    totals = {
        name: {"squared_error": 0.0, "reference_squared": 0.0, "numel": 0, "points": 0}
        for name in names
    }

    for capture in args.qkv:
        q_grid, k_grid, v_grid = _load_grid(capture, args.head)
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        v_grid = v_grid.to(args.device)
        key = k_grid.reshape(FRAMES * ROWS, COLS, -1)
        value = v_grid.reshape(FRAMES * ROWS, COLS, -1)
        tables = {
            name: _select_consensus(
                q_grid, key, value, routers, name, args.block_chunk
            )
            for name in names
        }
        for frame in frames:
            query = q_grid[frame].permute(1, 0, 2).contiguous()
            dense = _dense_frame(q_grid, k_grid, v_grid, frame)
            for name, rows in tables.items():
                metrics = evaluate_rows(query, key, value, dense, rows)
                totals[name]["squared_error"] += metrics["squared_error"]
                totals[name]["reference_squared"] += metrics["reference_squared"]
                totals[name]["numel"] += metrics["numel"]
                totals[name]["points"] += 1

    result = {"settings": vars(args), "variants": {}}
    for name, total in totals.items():
        result["variants"][name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "points": total["points"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["variants"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
