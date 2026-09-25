"""Evaluate the production Hellinger-span Triton selector on real Wan QKV."""

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

from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
from sweep_simple_weighted_span_qkv import (
    FRAMES,
    ROWS,
    COLS,
    _dense_frame,
    _hellinger_flop_budget,
    _parse_indices,
)
from sweep_simple_weighted_span_selectors import (
    _fixed_rows,
    evaluate_rows,
)


def _load_grid(path: str, heads: list[int]) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def _add(total: dict[str, float], metrics: dict[str, float]) -> None:
    total["squared_error"] += metrics["squared_error"]
    total["reference_squared"] += metrics["reference_squared"]
    total["numel"] += metrics["numel"]
    total["points"] += 1


def _router_frames(count: int) -> list[int]:
    if count <= 0 or count >= FRAMES:
        return list(range(FRAMES))
    if count == 4:
        return [0, 7, 13, 20]
    return [
        round((slot + 0.5) * FRAMES / count - 0.5)
        for slot in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--candidate_count", type=int, default=22)
    parser.add_argument("--candidate_geometry_count", type=int)
    parser.add_argument("--top_probes", type=int, default=8)
    parser.add_argument("--uniform_probes", type=int, default=8)
    parser.add_argument("--probe_anchor_count", type=int, default=2)
    parser.add_argument("--block_j", type=int, default=4)
    parser.add_argument("--group_tile", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--gs_num_warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--gs_key_tile", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--selector_warmup", type=int, default=0)
    parser.add_argument("--selector_repeats", type=int, default=1)
    parser.add_argument(
        "--selection_mode",
        choices=(
            "gs",
            "two_stage",
            "anchor_top3",
            "pivoted_cholesky",
            "pair_first_m4",
            "pair_first_m8",
        ),
        default="gs",
    )
    parser.add_argument("--routing_head_group", type=int, default=1)
    parser.add_argument("--routing_head_offset", type=int, default=0)
    parser.add_argument("--total_heads", type=int, default=40)
    parser.add_argument(
        "--temporal_count",
        type=int,
        default=0,
        help="Number of selector frames; 0 evaluates a distinct table per frame.",
    )
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    if args.routing_head_group <= 0:
        raise ValueError("routing_head_group must be positive")
    if not 0 <= args.routing_head_offset < args.routing_head_group:
        raise ValueError("routing_head_offset must be inside the routing group")
    if not 0 <= args.temporal_count <= FRAMES:
        raise ValueError(f"temporal_count must be in [0, {FRAMES}]")
    if args.selector_warmup < 0 or args.selector_repeats <= 0:
        raise ValueError("selector warmup/repeats must be nonnegative/positive")
    router_frames = _router_frames(args.temporal_count)
    router_for_head = {
        head: min(
            (head // args.routing_head_group) * args.routing_head_group
            + args.routing_head_offset,
            args.total_heads - 1,
        )
        for head in heads
    }
    router_heads = sorted(set(router_for_head.values()))
    router_slot = {head: slot for slot, head in enumerate(router_heads)}
    records = {}
    selector_label = f"hellinger_{args.selection_mode}_triton"
    totals = {
        name: {
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "numel": 0,
            "points": 0,
        }
        for name in (selector_label, "fixed_c4")
    }
    selector_seconds = 0.0
    selector_calls = 0

    for capture in args.qkv:
        q_grid, k_grid, v_grid = _load_grid(capture, heads)
        router_q_grid, router_k_grid, _ = _load_grid(capture, router_heads)
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        v_grid = v_grid.to(args.device)
        router_q_grid = router_q_grid.to(args.device)
        router_k_grid = router_k_grid.to(args.device)
        key_blocks = k_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)
        value_blocks = v_grid.reshape(len(heads), FRAMES * ROWS, COLS, -1)
        production_k = (
            router_k_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        )

        production_q = (
            router_q_grid[:, router_frames]
            .permute(1, 2, 3, 0, 4)
            .unsqueeze(0)
            .contiguous()
        )
        def select_seed_table():
            return hellinger_span_seed_rows_triton(
                production_q,
                production_k,
                alpha=args.alpha,
                candidate_count=args.candidate_count,
                candidate_geometry_count=args.candidate_geometry_count,
                top_probes=args.top_probes,
                uniform_probes=args.uniform_probes,
                probe_anchor_count=args.probe_anchor_count,
                block_j=args.block_j,
                group_tile=args.group_tile,
                gs_num_warps=args.gs_num_warps,
                gs_key_tile=args.gs_key_tile,
                selection_mode=args.selection_mode,
            )

        for _ in range(args.selector_warmup):
            seed_table = select_seed_table()
        torch.cuda.synchronize()
        for _ in range(args.selector_repeats):
            started = time.perf_counter()
            seed_table = select_seed_table()
            torch.cuda.synchronize()
            selector_seconds += time.perf_counter() - started
            selector_calls += 1

        for frame in frames:
            table_slot = min(
                range(len(router_frames)),
                key=lambda index: abs(frame - router_frames[index]),
            )
            for head_slot, head in enumerate(heads):
                query = q_grid[head_slot, frame].permute(1, 0, 2).contiguous()
                key = key_blocks[head_slot]
                value = value_blocks[head_slot]
                dense = _dense_frame(
                    q_grid[head_slot], k_grid[head_slot], v_grid[head_slot], frame
                )
                rows = (
                    seed_table[
                        0,
                        table_slot,
                        ...,
                        router_slot[router_for_head[head]],
                        :,
                    ]
                    .reshape(FRAMES * ROWS, COLS, 4)
                    .permute(1, 0, 2)
                    .contiguous()
                )
                fixed_rows = _fixed_rows(query, key.size(0))
                point = {
                    selector_label: evaluate_rows(
                        query, key, value, dense, rows
                    ),
                    "fixed_c4": evaluate_rows(
                        query, key, value, dense, fixed_rows
                    ),
                }
                point_name = f"{Path(capture).parent.name}:h{head}:f{frame}"
                records[point_name] = point
                for name, metrics in point.items():
                    _add(totals[name], metrics)
                print(
                    f"{point_name} hellinger="
                    f"{point[selector_label]['rel_l2_vs_dense']:.6f} "
                    f"fixed={point['fixed_c4']['rel_l2_vs_dense']:.6f}",
                    flush=True,
                )

    aggregate = {}
    for name, total in totals.items():
        aggregate[name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "points": total["points"],
        }
    result = {
        "settings": vars(args),
        "router_frames": router_frames,
        "hellinger_dominant_flops": _hellinger_flop_budget(
            args.candidate_count,
            args.top_probes,
            args.uniform_probes,
            args.block_j,
            0,
            args.probe_anchor_count,
        ),
        "selector_seconds_total": selector_seconds,
        "selector_ms_per_multi_head_call": 1000.0 * selector_seconds / selector_calls,
        "records": records,
        "aggregate": aggregate,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(aggregate, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
