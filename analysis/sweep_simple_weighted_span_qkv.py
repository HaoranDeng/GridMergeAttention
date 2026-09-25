"""Validate simplified weighted-span selectors on full real Wan QKV captures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sweep_simple_weighted_span_selectors import (
    _fixed_rows,
    _shared_seed_count,
    evaluate_rows,
    select_rows,
)


FRAMES = 21
ROWS = 45
COLS = 80
C4 = 4
DENSE_TFLOPS_PER_CALL = 117.0505728


def _hellinger_flop_budget(
    candidate_count: int,
    top_probes: int,
    uniform_probes: int,
    block_j: int,
    sketch_dim: int,
    probe_anchor_count: int,
) -> dict[str, float]:
    """Dominant matmul FLOPs, normalized by dense QK+AV attention."""
    hard_c4 = 3 * C4 / (2 * ROWS) + C4 / COLS
    effective_dim = sketch_dim if 0 < sketch_dim < 128 else 128
    dimension_ratio = effective_dim / 128
    profile = (
        candidate_count
        * (top_probes + uniform_probes)
        / (2 * ROWS * COLS)
        * dimension_ratio
    )
    adaptive_probe = (
        probe_anchor_count / (2 * ROWS * block_j) * dimension_ratio
        if top_probes
        else 0.0
    )
    projection = effective_dim / (FRAMES * ROWS * COLS) if effective_dim < 128 else 0.0
    selector = profile + adaptive_probe + projection
    total = hard_c4 + selector
    return {
        "hard_c4_ratio_vs_dense": hard_c4,
        "candidate_probe_ratio_vs_dense": profile,
        "adaptive_probe_ratio_vs_dense": adaptive_probe,
        "projection_ratio_vs_dense": projection,
        "selector_ratio_vs_dense": selector,
        "total_ratio_vs_dense": total,
        "ratio_vs_hard_c4": total / hard_c4,
        "tflops_per_40_head_call": total * DENSE_TFLOPS_PER_CALL,
        "passes_1p5x_hard_c4": total <= 1.5 * hard_c4,
    }


def _coarse_to_fine_flop_budget(
    coarse_probes: int,
    shortlist_size: int,
    *,
    full_candidates: int,
    full_probes: int,
    block_j: int,
    probe_anchor_count: int,
) -> dict[str, float]:
    hard_c4 = 3 * C4 / (2 * ROWS) + C4 / COLS
    profile_products = (
        full_candidates * coarse_probes
        + shortlist_size * (full_probes - coarse_probes)
    )
    profile = profile_products / (2 * ROWS * COLS)
    adaptive_probe = probe_anchor_count / (2 * ROWS * block_j)
    total = hard_c4 + profile + adaptive_probe
    return {
        "hard_c4_ratio_vs_dense": hard_c4,
        "candidate_probe_ratio_vs_dense": profile,
        "adaptive_probe_ratio_vs_dense": adaptive_probe,
        "total_ratio_vs_dense": total,
        "ratio_vs_hard_c4": total / hard_c4,
        "tflops_per_40_head_call": total * DENSE_TFLOPS_PER_CALL,
        "passes_1p5x_hard_c4": total <= 1.5 * hard_c4,
    }


def _parse_indices(text: str) -> list[int]:
    return [int(value) for value in text.replace(",", " ").split()]


def _load_heads(path: str, heads: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path) as payload:
        arrays = []
        for name in ("q", "k", "v"):
            arrays.append(torch.from_numpy(payload[name][0, heads].copy()))
    return tuple(arrays)


@torch.inference_mode()
def _dense_frame(
    query_grid: torch.Tensor,
    key_grid: torch.Tensor,
    value_grid: torch.Tensor,
    frame: int,
) -> torch.Tensor:
    query = query_grid[frame].reshape(1, 1, ROWS * COLS, -1)
    key = key_grid.reshape(1, 1, FRAMES * ROWS * COLS, -1)
    value = value_grid.reshape(1, 1, FRAMES * ROWS * COLS, -1)
    output = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )[0, 0]
    return output.reshape(ROWS, COLS, -1).permute(1, 0, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--candidate_count", type=int, default=45)
    parser.add_argument(
        "--candidate_layout",
        choices=("local", "uniform", "local_uniform"),
        default="local",
    )
    parser.add_argument("--block_j", type=int, default=2)
    parser.add_argument("--block_chunk", type=int, default=21)
    parser.add_argument("--top_probes", type=int, default=8)
    parser.add_argument("--uniform_probes", type=int, default=8)
    parser.add_argument("--probe_anchor_count", type=int, default=4)
    parser.add_argument("--sketch_dim", type=int, default=0)
    parser.add_argument("--sketch_seed", type=int, default=20260825)
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    records: dict[str, dict[str, float]] = {}
    totals: dict[str, dict[str, float]] = {}

    for capture in args.qkv:
        q_heads, k_heads, v_heads = _load_heads(capture, heads)
        for head_slot, head in enumerate(heads):
            q_grid = q_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(args.device)
            k_grid = k_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(args.device)
            v_grid = v_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(args.device)
            key_blocks = k_grid.reshape(FRAMES * ROWS, COLS, -1)
            value_blocks = v_grid.reshape(FRAMES * ROWS, COLS, -1)

            for frame in frames:
                query = q_grid[frame].permute(1, 0, 2).contiguous()
                dense = _dense_frame(q_grid, k_grid, v_grid, frame)
                rows, selector_seconds = select_rows(
                    query,
                    key_blocks,
                    value_blocks,
                    alpha=args.alpha,
                    candidate_count=args.candidate_count,
                    block_j=args.block_j,
                    block_chunk=args.block_chunk,
                    top_probes=args.top_probes,
                    uniform_probes=args.uniform_probes,
                    probe_anchor_count=args.probe_anchor_count,
                    sketch_dim=args.sketch_dim,
                    sketch_seed=args.sketch_seed,
                    candidate_layout=args.candidate_layout,
                )
                rows["fixed_c4"] = _fixed_rows(query, key_blocks.size(0))
                reference = rows["weighted_gs"]
                point_name = f"{Path(capture).parent.name}:h{head}:f{frame}"
                point = {}
                for name, seed_rows in rows.items():
                    metrics = evaluate_rows(
                        query, key_blocks, value_blocks, dense, seed_rows
                    )
                    metrics["shared_seeds_vs_weighted_gs"] = _shared_seed_count(
                        seed_rows, reference
                    )
                    if name in selector_seconds:
                        metrics["selection_seconds_python_screen"] = selector_seconds[name]
                    point[name] = metrics
                    total = totals.setdefault(
                        name,
                        {"squared_error": 0.0, "reference_squared": 0.0, "numel": 0.0,
                         "shared": 0.0, "points": 0.0},
                    )
                    total["squared_error"] += metrics["squared_error"]
                    total["reference_squared"] += metrics["reference_squared"]
                    total["numel"] += metrics["numel"]
                    total["shared"] += metrics["shared_seeds_vs_weighted_gs"]
                    total["points"] += 1
                records[point_name] = point
                print(
                    f"{point_name} weighted={point['weighted_gs']['rel_l2_vs_dense']:.6f} "
                    f"joint6={point['joint_volume_m6']['rel_l2_vs_dense']:.6f} "
                    f"anchor={point['anchor_top3']['rel_l2_vs_dense']:.6f}",
                    flush=True,
                )
                del rows, dense
                torch.cuda.empty_cache()
            del q_grid, k_grid, v_grid, key_blocks, value_blocks
            torch.cuda.empty_cache()
        del q_heads, k_heads, v_heads

    aggregate = {}
    for name, total in totals.items():
        aggregate[name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "shared_seeds_vs_weighted_gs": total["shared"] / total["points"],
            "points": int(total["points"]),
        }
    ranking = sorted(aggregate, key=lambda name: aggregate[name]["rel_l2_vs_dense"])
    print("\naggregate")
    for name in ranking:
        value = aggregate[name]
        print(
            f"{name:>22s} rel_l2={value['rel_l2_vs_dense']:.6f} "
            f"mse={value['mse_vs_dense']:.8f} "
            f"shared={value['shared_seeds_vs_weighted_gs']:.3f}/4"
        )

    result = {
        "settings": vars(args),
        "hellinger_dominant_flops": _hellinger_flop_budget(
            args.candidate_count,
            args.top_probes,
            args.uniform_probes,
            args.block_j,
            args.sketch_dim,
            args.probe_anchor_count,
        ),
        "coarse_to_fine_dominant_flops": {
            f"hellinger_coarse{coarse}_short{shortlist}": (
                _coarse_to_fine_flop_budget(
                    coarse,
                    shortlist,
                    full_candidates=args.candidate_count,
                    full_probes=args.top_probes + args.uniform_probes,
                    block_j=args.block_j,
                    probe_anchor_count=args.probe_anchor_count,
                )
            )
            for coarse, shortlist in ((2, 29), (3, 28), (4, 25), (4, 26))
            if args.uniform_probes >= 4 and args.candidate_count >= 29
        },
        "records": records,
        "aggregate": aggregate,
        "ranking": ranking,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
