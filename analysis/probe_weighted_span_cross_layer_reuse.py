"""Measure adjacent-layer reuse of per-frame Weighted-Span seed rows."""

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
from sweep_simple_weighted_span_qkv import (
    COLS,
    FRAMES,
    ROWS,
    _dense_frame,
    _parse_indices,
)
from sweep_simple_weighted_span_selectors import evaluate_rows


def _load_grid(path: Path, heads: list[int]) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def _production(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()


def _table_rows(
    table: torch.Tensor, frame: int, head_slot: int
) -> torch.Tensor:
    return (
        table[0, frame, :, :, :, head_slot, :]
        .reshape(FRAMES * ROWS, COLS, 4)
        .permute(1, 0, 2)
        .contiguous()
        .to(torch.long)
    )


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for item in text.split(","):
        source, target = item.split("-")
        pairs.append((int(source), int(target)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", default="1-2,11-12,21-22,31-32,39-40")
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,5,10,15,20")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    pairs = _parse_pairs(args.pairs)
    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    totals = {
        name: {
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "numel": 0,
            "points": 0,
        }
        for name in ("own_layer", "previous_layer")
    }
    shared_seed_count = 0.0
    shared_seed_entries = 0
    exact_seed_sets = 0.0

    for source_call, target_call in pairs:
        source_path = Path(args.qkv_dir) / f"call{source_call:03d}_qkv.npz"
        target_path = Path(args.qkv_dir) / f"call{target_call:03d}_qkv.npz"
        source = tuple(x.to(args.device) for x in _load_grid(source_path, heads))
        target = tuple(x.to(args.device) for x in _load_grid(target_path, heads))
        source_table = span_seed_rows_triton_split(
            *(_production(x) for x in source),
            alpha=0.3,
            candidate_count=45,
            top_probes=8,
            uniform_probes=8,
            value_dims=8,
            block_j=4,
        )
        target_table = span_seed_rows_triton_split(
            *(_production(x) for x in target),
            alpha=0.3,
            candidate_count=45,
            top_probes=8,
            uniform_probes=8,
            value_dims=8,
            block_j=4,
        )
        target_q, target_k, target_v = target
        key_blocks = target_k.reshape(len(heads), FRAMES * ROWS, COLS, -1)
        value_blocks = target_v.reshape(len(heads), FRAMES * ROWS, COLS, -1)

        for head_slot in range(len(heads)):
            for frame in frames:
                own_rows = _table_rows(target_table, frame, head_slot)
                previous_rows = _table_rows(source_table, frame, head_slot)
                own_sorted = own_rows.sort(dim=-1).values
                previous_sorted = previous_rows.sort(dim=-1).values
                exact_seed_sets += float(
                    (own_sorted == previous_sorted).all(dim=-1).float().sum()
                )
                shared_seed_count += float(
                    (
                        own_rows.unsqueeze(-1)
                        == previous_rows.unsqueeze(-2)
                    )
                    .any(dim=-1)
                    .float()
                    .sum()
                )
                shared_seed_entries += own_rows.shape[0] * own_rows.shape[1]

                query = target_q[head_slot, frame].permute(1, 0, 2).contiguous()
                dense = _dense_frame(
                    target_q[head_slot],
                    target_k[head_slot],
                    target_v[head_slot],
                    frame,
                )
                for name, rows in (
                    ("own_layer", own_rows),
                    ("previous_layer", previous_rows),
                ):
                    metrics = evaluate_rows(
                        query,
                        key_blocks[head_slot],
                        value_blocks[head_slot],
                        dense,
                        rows,
                    )
                    totals[name]["squared_error"] += metrics["squared_error"]
                    totals[name]["reference_squared"] += metrics[
                        "reference_squared"
                    ]
                    totals[name]["numel"] += metrics["numel"]
                    totals[name]["points"] += 1

    variants = {}
    for name, total in totals.items():
        variants[name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"]
                / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "points": total["points"],
        }
    result = {
        "settings": vars(args),
        "variants": variants,
        "seed_agreement": {
            "exact_set_rate": exact_seed_sets / shared_seed_entries,
            "shared_of_4": shared_seed_count / shared_seed_entries,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
