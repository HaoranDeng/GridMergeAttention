"""Evaluate fixed random P16 probes for the production Hellinger selector.

This is an offline real-QKV ablation.  It changes only the 16 key-column
probes used to choose the four seed-query rows; merge and compressed
attention remain identical to the production Hellinger C4 path.
"""

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
if str(REPO_ROOT / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "analysis"))

import sweep_simple_weighted_span_selectors as selectors
from sweep_simple_weighted_span_qkv import (
    COLS,
    FRAMES,
    ROWS,
    _dense_frame,
    _parse_indices,
)


def _load_grid(path: str, heads: list[int]) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def _fixed_probe_function(
    probe_table: torch.Tensor,
):
    def fixed_probes(
        query: torch.Tensor,
        key: torch.Tensor,
        key_rows: torch.Tensor,
        scale: float,
        block_j: int,
        top_probes: int,
        uniform_probes: int,
        probe_anchor_count: int,
    ) -> torch.Tensor:
        del key, scale, probe_anchor_count
        probe_count = top_probes + uniform_probes
        groups = math.ceil(query.size(0) / block_j)
        if probe_table.shape != (groups, ROWS, probe_count):
            raise ValueError(
                f"probe table {tuple(probe_table.shape)} does not match "
                f"{(groups, ROWS, probe_count)}"
            )
        return probe_table[:, key_rows]

    return fixed_probes


def _uniform_table(groups: int, probe_count: int, device: torch.device) -> torch.Tensor:
    columns = torch.arange(probe_count, device=device) * (COLS // probe_count)
    return columns.view(1, 1, -1).expand(groups, ROWS, -1).contiguous()


def _random_table(
    groups: int,
    probe_count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    table = np.empty((groups, ROWS, probe_count), dtype=np.int64)
    for group in range(groups):
        for key_row in range(ROWS):
            table[group, key_row] = rng.choice(
                COLS, size=probe_count, replace=False
            )
    return torch.from_numpy(table).to(device)


def _add(total: dict[str, float], metrics: dict[str, float]) -> None:
    total["squared_error"] += metrics["squared_error"]
    total["reference_squared"] += metrics["reference_squared"]
    total["numel"] += metrics["numel"]
    total["overlap"] += metrics["seed_overlap_vs_hybrid"]
    total["points"] += 1


def _select(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    rows, _ = selectors.select_rows(
        query,
        key,
        value,
        alpha=args.alpha,
        candidate_count=args.candidate_count,
        block_j=args.block_j,
        block_chunk=args.block_chunk,
        top_probes=args.probe_count // 2,
        uniform_probes=args.probe_count // 2,
        probe_anchor_count=args.probe_anchor_count,
        candidate_layout=args.candidate_layout,
        variant_names={"hellinger_gs"},
    )
    return rows["hellinger_gs"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,5,10,15,20")
    parser.add_argument("--router_frame", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--candidate_count", type=int, default=32)
    parser.add_argument("--candidate_layout", default="multiscale")
    parser.add_argument("--block_j", type=int, default=4)
    parser.add_argument("--block_chunk", type=int, default=21)
    parser.add_argument("--probe_count", type=int, default=16)
    parser.add_argument("--probe_anchor_count", type=int, default=2)
    parser.add_argument("--random_seeds", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    random_seeds = _parse_indices(args.random_seeds)
    if args.probe_count % 2:
        raise ValueError("probe_count must be even for the production comparison")
    if not 0 <= args.router_frame < FRAMES:
        raise ValueError(f"router_frame must be in [0, {FRAMES})")

    method_names = ["hybrid_p16", "uniform_p16"] + [
        f"random_p16_seed{seed}" for seed in random_seeds
    ]
    totals = {
        name: {
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "numel": 0.0,
            "overlap": 0.0,
            "points": 0,
        }
        for name in method_names
    }
    records: dict[str, dict[str, dict[str, float]]] = {}
    original_probe_function = selectors._probe_indices

    try:
        for capture_index, capture in enumerate(args.qkv):
            q_grid, k_grid, v_grid = _load_grid(capture, heads)
            q_grid = q_grid.to(args.device)
            k_grid = k_grid.to(args.device)
            v_grid = v_grid.to(args.device)

            for head_slot, head in enumerate(heads):
                query_router = q_grid[head_slot, args.router_frame].permute(
                    1, 0, 2
                ).contiguous()
                key = k_grid[head_slot].reshape(FRAMES * ROWS, COLS, -1)
                value = v_grid[head_slot].reshape(FRAMES * ROWS, COLS, -1)
                groups = math.ceil(COLS / args.block_j)

                selectors._probe_indices = original_probe_function
                hybrid_rows = _select(query_router, key, value, args)
                rows_by_method = {"hybrid_p16": hybrid_rows}

                uniform = _uniform_table(groups, args.probe_count, query_router.device)
                selectors._probe_indices = _fixed_probe_function(uniform)
                rows_by_method["uniform_p16"] = _select(
                    query_router, key, value, args
                )

                for seed in random_seeds:
                    # Include capture/head in the fixed table while keeping runs
                    # reproducible from the user-visible random seed.
                    table_seed = seed * 100_000 + capture_index * 1_000 + head
                    random_table = _random_table(
                        groups,
                        args.probe_count,
                        table_seed,
                        query_router.device,
                    )
                    selectors._probe_indices = _fixed_probe_function(random_table)
                    rows_by_method[f"random_p16_seed{seed}"] = _select(
                        query_router, key, value, args
                    )

                for frame in frames:
                    query = q_grid[head_slot, frame].permute(1, 0, 2).contiguous()
                    dense = _dense_frame(
                        q_grid[head_slot], k_grid[head_slot], v_grid[head_slot], frame
                    )
                    point_name = f"{Path(capture).stem}:h{head}:f{frame}"
                    point = {}
                    for name, rows in rows_by_method.items():
                        metrics = selectors.evaluate_rows(
                            query, key, value, dense, rows
                        )
                        metrics["seed_overlap_vs_hybrid"] = selectors._shared_seed_count(
                            rows, hybrid_rows
                        )
                        point[name] = metrics
                        _add(totals[name], metrics)
                    records[point_name] = point
                    print(
                        point_name,
                        " ".join(
                            f"{name}={point[name]['rel_l2_vs_dense']:.6f}"
                            for name in method_names
                        ),
                        flush=True,
                    )
                    del dense

                del query_router, key, value, rows_by_method
                torch.cuda.empty_cache()
            del q_grid, k_grid, v_grid
            torch.cuda.empty_cache()
    finally:
        selectors._probe_indices = original_probe_function

    aggregate = {}
    for name, total in totals.items():
        aggregate[name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "seed_overlap_vs_hybrid": total["overlap"] / total["points"],
            "points": total["points"],
        }

    random_values = [aggregate[f"random_p16_seed{seed}"] for seed in random_seeds]
    random_summary = {
        metric: {
            "mean": float(np.mean([value[metric] for value in random_values])),
            "std": float(np.std([value[metric] for value in random_values])),
            "min": float(np.min([value[metric] for value in random_values])),
            "max": float(np.max([value[metric] for value in random_values])),
        }
        for metric in (
            "rel_l2_vs_dense",
            "mse_vs_dense",
            "seed_overlap_vs_hybrid",
        )
    }
    result = {
        "settings": vars(args),
        "aggregate": aggregate,
        "random_summary": random_summary,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"aggregate": aggregate, "random_summary": random_summary}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
