"""Benchmark production Hellinger selectors on one real Wan QKV capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton


ROWS = 45
COLS = 80
FRAMES = 21


def _indices(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def _load_qk(path: str, heads: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as payload:
        tensors = []
        for name in ("q", "k"):
            grid = torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            tensors.append(grid)
    return tensors[0], tensors[1]


def _set_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    matches = left[..., :, None] == right[..., None, :]
    return matches.any(dim=-1).float().sum(dim=-1).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modes", default="gs,pivoted_cholesky,pair_first_m4")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    heads = _indices(args.heads)
    modes = [value for value in args.modes.split(",") if value]
    q_grid, k_grid = _load_qk(args.qkv, heads)
    q = (
        q_grid[:, :1]
        .permute(1, 2, 3, 0, 4)
        .unsqueeze(0)
        .contiguous()
        .to(args.device)
    )
    k = (
        k_grid.permute(1, 2, 3, 0, 4)
        .unsqueeze(0)
        .contiguous()
        .to(args.device)
    )

    def select(mode: str, timings: dict[str, float] | None = None) -> torch.Tensor:
        return hellinger_span_seed_rows_triton(
            q,
            k,
            alpha=0.2,
            candidate_count=32,
            candidate_geometry_count=45,
            top_probes=8,
            uniform_probes=8,
            probe_anchor_count=2,
            block_j=4,
            group_tile=32,
            gs_num_warps=8,
            selection_mode=mode,
            timings=timings,
        )

    tables: dict[str, torch.Tensor] = {}
    result: dict[str, object] = {
        "qkv": args.qkv,
        "heads": heads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "modes": {},
    }
    for mode in modes:
        for _ in range(args.warmup):
            tables[mode] = select(mode)
        probe_ms = []
        selection_ms = []
        for _ in range(args.repeats):
            timings: dict[str, float] = {}
            tables[mode] = select(mode, timings)
            probe_ms.append(timings["probe_ms"])
            selection_ms.append(timings["gs_ms"])
        mode_result = {
            "probe_ms_mean": float(np.mean(probe_ms)),
            "probe_ms_std": float(np.std(probe_ms)),
            "selection_ms_mean": float(np.mean(selection_ms)),
            "selection_ms_std": float(np.std(selection_ms)),
            "kernel_ms_mean": float(np.mean(probe_ms) + np.mean(selection_ms)),
        }
        result["modes"][mode] = mode_result
        print(mode, json.dumps(mode_result), flush=True)

    reference = tables["gs"]
    for mode in modes:
        mode_result = result["modes"][mode]
        table = tables[mode]
        mode_result["ordered_seed_match_fraction_vs_gs"] = (
            (table == reference).float().mean().item()
        )
        mode_result["set_overlap_vs_gs"] = _set_overlap(table, reference)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
