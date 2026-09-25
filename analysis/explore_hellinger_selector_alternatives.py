"""Compare stable and parallel Hellinger seed selectors on real Wan QKV.

The routing table is selected from one fixed router frame (RF0 by default),
then reused to evaluate multiple target query frames.  This matches the
production first-frame Hellinger setup instead of silently selecting a fresh
table for every evaluated frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from sweep_simple_weighted_span_qkv import (
    COLS,
    FRAMES,
    ROWS,
    _dense_frame,
    _load_heads,
    _parse_indices,
)
from sweep_simple_weighted_span_selectors import (
    _shared_seed_count,
    evaluate_rows,
    select_rows,
)


DEFAULT_VARIANTS = (
    "hellinger_gs",
    "hellinger_pivoted_cholesky",
    "hellinger_two_stage",
    "hellinger_anchor_top3",
    "hellinger_pivot_pair_m8",
    "hellinger_pair_first",
    "hellinger_joint_volume_m8",
    "hellinger_joint_volume_m12",
    "hellinger_joint_volume_m16",
    "hellinger_joint_volume_full",
)


def _add(total: dict[str, float], metrics: dict[str, float]) -> None:
    total["squared_error"] += metrics["squared_error"]
    total["reference_squared"] += metrics["reference_squared"]
    total["numel"] += metrics["numel"]
    total["shared"] += metrics["shared_seeds_vs_hellinger_gs"]
    total["points"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--router_frame", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--candidate_count", type=int, default=32)
    parser.add_argument("--block_j", type=int, default=4)
    parser.add_argument("--block_chunk", type=int, default=21)
    parser.add_argument("--top_probes", type=int, default=8)
    parser.add_argument("--uniform_probes", type=int, default=8)
    parser.add_argument("--probe_anchor_count", type=int, default=2)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    args = parser.parse_args()

    if not 0 <= args.router_frame < FRAMES:
        raise ValueError(f"router_frame must be in [0, {FRAMES})")
    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    variants = {name.strip() for name in args.variants.split(",") if name.strip()}
    if "hellinger_gs" not in variants:
        raise ValueError("hellinger_gs is required as the seed-overlap reference")

    records: dict[str, dict[str, dict[str, float]]] = {}
    totals: dict[str, dict[str, float]] = {
        name: {
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "numel": 0.0,
            "shared": 0.0,
            "points": 0.0,
        }
        for name in variants
    }
    selection_seconds = {name: 0.0 for name in variants}
    routing_tables = 0

    for capture in args.qkv:
        q_heads, k_heads, v_heads = _load_heads(capture, heads)
        for head_slot, head in enumerate(heads):
            q_grid = q_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(args.device)
            k_grid = k_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(args.device)
            v_grid = v_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(args.device)
            key_blocks = k_grid.reshape(FRAMES * ROWS, COLS, -1)
            value_blocks = v_grid.reshape(FRAMES * ROWS, COLS, -1)
            router_query = q_grid[args.router_frame].permute(1, 0, 2).contiguous()
            rows, elapsed = select_rows(
                router_query,
                key_blocks,
                value_blocks,
                alpha=args.alpha,
                candidate_count=args.candidate_count,
                block_j=args.block_j,
                block_chunk=args.block_chunk,
                top_probes=args.top_probes,
                uniform_probes=args.uniform_probes,
                probe_anchor_count=args.probe_anchor_count,
                candidate_layout="multiscale",
                variant_names=variants,
            )
            routing_tables += 1
            for name, seconds in elapsed.items():
                selection_seconds[name] += seconds
            reference = rows["hellinger_gs"]

            for frame in frames:
                query = q_grid[frame].permute(1, 0, 2).contiguous()
                dense = _dense_frame(q_grid, k_grid, v_grid, frame)
                point_name = f"{Path(capture).stem}:h{head}:f{frame}"
                point = {}
                for name, seed_rows in rows.items():
                    metrics = evaluate_rows(
                        query,
                        key_blocks,
                        value_blocks,
                        dense,
                        seed_rows,
                    )
                    metrics["shared_seeds_vs_hellinger_gs"] = _shared_seed_count(
                        seed_rows,
                        reference,
                    )
                    point[name] = metrics
                    _add(totals[name], metrics)
                records[point_name] = point
                print(
                    f"{point_name} "
                    + " ".join(
                        f"{name}={point[name]['rel_l2_vs_dense']:.6f}"
                        for name in sorted(point)
                    ),
                    flush=True,
                )
                del dense
            del rows, q_grid, k_grid, v_grid, key_blocks, value_blocks
            torch.cuda.empty_cache()
        del q_heads, k_heads, v_heads

    aggregate = {}
    for name, total in totals.items():
        aggregate[name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "shared_seeds_vs_hellinger_gs": total["shared"] / total["points"],
            "selection_ms_per_routing_table_python": (
                1000.0 * selection_seconds[name] / routing_tables
            ),
            "points": int(total["points"]),
        }
    ranking = sorted(aggregate, key=lambda name: aggregate[name]["rel_l2_vs_dense"])
    print("\naggregate")
    for name in ranking:
        value = aggregate[name]
        print(
            f"{name:>34s} rel_l2={value['rel_l2_vs_dense']:.6f} "
            f"shared={value['shared_seeds_vs_hellinger_gs']:.3f}/4 "
            f"python_ms={value['selection_ms_per_routing_table_python']:.3f}"
        )

    result = {
        "settings": vars(args),
        "aggregate": aggregate,
        "ranking": ranking,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
