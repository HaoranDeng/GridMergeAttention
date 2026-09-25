#!/usr/bin/env python3
"""Summarize the exact retained attention-map area of the static SVG mask."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg_pattern_density", type=float, required=True)
    parser.add_argument("--requested_density", type=float, required=True)
    parser.add_argument("--num_frame_patches", type=int, default=21)
    parser.add_argument("--tokens_per_frame", type=int, default=3600)
    parser.add_argument("--block_rounding", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def retained_elements(seq_len: int, frame_size: int, half_band: int) -> int:
    total = 0
    for query_index in range(seq_len):
        lower = max(0, query_index - half_band)
        upper = min(seq_len - 1, query_index + half_band)
        band_elements = upper - lower + 1
        sink_overlap = max(0, min(upper, frame_size - 1) - lower + 1)
        total += band_elements + frame_size - sink_overlap
    return total


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.svg_pattern_density <= 1.0:
        raise ValueError("svg_pattern_density must be in [0, 1]")

    seq_len = args.num_frame_patches * args.tokens_per_frame
    continuous_half_band = seq_len * (
        1.0 - math.sqrt(1.0 - args.svg_pattern_density)
    )
    half_band = (
        math.ceil(continuous_half_band / args.block_rounding)
        * args.block_rounding
    )
    retained = retained_elements(seq_len, args.tokens_per_frame, half_band)
    dense = seq_len * seq_len
    density = retained / dense
    payload = {
        "backend": "svg_all",
        "requested_density": args.requested_density,
        "svg_pattern_density": args.svg_pattern_density,
        "realized_density": density,
        "realized_sparsity": 1.0 - density,
        "num_frame_patches": args.num_frame_patches,
        "tokens_per_frame": args.tokens_per_frame,
        "sequence_length": seq_len,
        "continuous_half_band_tokens": continuous_half_band,
        "rounded_half_band_tokens": half_band,
        "retained_attention_elements": retained,
        "dense_attention_elements": dense,
        "density_definition": (
            "Exact static SVG token-mask area including the first-frame sink and "
            "the rounded symmetric diagonal band, divided by dense N squared."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
