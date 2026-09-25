#!/usr/bin/env python3
"""Estimate principal attention FLOPs for residual-anchor GridMerge.

The SVG comparison is deliberately conservative: its measured realized
attention-map density is treated as its entire cost, omitting clustering and
routing overhead. GridMerge includes assignment, K/V merging, compressed
QK/AV, exact anchor-query QK/AV, and bilinear residual interpolation.
"""

from __future__ import annotations

import argparse
import json


def estimate(
    *,
    frames: int,
    rows: int,
    cols: int,
    clusters: int,
    anchor_times: int,
    anchor_rows: int,
    total_layers: int,
    anchor_layers: int,
    svg_density: float,
    svg_fraction_limit: float,
) -> dict[str, float | int | bool]:
    if not 0 <= anchor_layers <= total_layers:
        raise ValueError("anchor_layers must be between zero and total_layers")
    anchor_layer_fraction = anchor_layers / total_layers
    gma_contraction_ratio = (
        3 * clusters * cols + 2 * rows * clusters
    ) / (2 * rows * cols)
    active_anchor_query_ratio = anchor_times * anchor_rows / (frames * rows)
    anchor_query_ratio = active_anchor_query_ratio * anchor_layer_fraction

    # Row interpolation is evaluated at every time anchor; temporal
    # interpolation and the final residual add are evaluated at every query.
    # Normalize these scalar operations by dense QK+AV (four FLOPs per
    # query-key-channel pair under the same multiply/add convention).
    active_interpolation_ratio = (3 * anchor_times + 4 * frames) / (
        4 * frames * frames * rows * cols
    )
    interpolation_ratio = active_interpolation_ratio * anchor_layer_fraction
    total_ratio = (
        gma_contraction_ratio + anchor_query_ratio + interpolation_ratio
    )
    budget = svg_fraction_limit * svg_density
    return {
        "frames": frames,
        "rows": rows,
        "cols": cols,
        "clusters": clusters,
        "anchor_times": anchor_times,
        "anchor_rows": anchor_rows,
        "anchor_points": anchor_times * anchor_rows,
        "total_layers": total_layers,
        "anchor_layers": anchor_layers,
        "anchor_layer_fraction": anchor_layer_fraction,
        "gma_contraction_ratio_vs_dense": gma_contraction_ratio,
        "active_layer_anchor_query_ratio_vs_dense": active_anchor_query_ratio,
        "anchor_query_ratio_vs_dense": anchor_query_ratio,
        "active_layer_interpolation_ratio_vs_dense": active_interpolation_ratio,
        "interpolation_ratio_vs_dense": interpolation_ratio,
        "total_ratio_vs_dense": total_ratio,
        "svg_density": svg_density,
        "ratio_vs_svg": total_ratio / svg_density,
        "allowed_ratio_vs_dense": budget,
        "within_svg_fraction_limit": total_ratio <= budget,
    }


def parse_grid(spec: str) -> tuple[int, int]:
    try:
        time_count, row_count = spec.lower().split("x", 1)
        return int(time_count), int(row_count)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid anchor grid {spec!r}; expected TIME_COUNTxROW_COUNT"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--rows", type=int, default=45)
    parser.add_argument("--cols", type=int, default=80)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--total-layers", type=int, default=40)
    parser.add_argument("--anchor-layers", type=int, default=40)
    parser.add_argument(
        "--anchor-grid", type=parse_grid, action="append", required=True
    )
    parser.add_argument("--svg-density", type=float, default=0.37906453310046345)
    parser.add_argument("--svg-fraction-limit", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        f"{time_count}x{row_count}": estimate(
            frames=args.frames,
            rows=args.rows,
            cols=args.cols,
            clusters=args.clusters,
            anchor_times=time_count,
            anchor_rows=row_count,
            total_layers=args.total_layers,
            anchor_layers=args.anchor_layers,
            svg_density=args.svg_density,
            svg_fraction_limit=args.svg_fraction_limit,
        )
        for time_count, row_count in args.anchor_grid
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
