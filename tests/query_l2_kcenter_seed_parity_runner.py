"""CUDA parity check for the query-only local L2 k-center selector."""

from __future__ import annotations

import argparse

import torch

from attention.adaptive_seed_pytorch import (
    query_l2_kcenter_seed_rows_pytorch,
)
from attention.adaptive_seed_triton import query_l2_kcenter_seed_rows_triton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--heads", type=int, default=3)
    args = parser.parse_args()

    generator = torch.Generator(device=args.device).manual_seed(20260828)
    query = torch.randn(
        1,
        args.frames,
        45,
        80,
        args.heads,
        128,
        device=args.device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    for sketch_dimensions in (16, 32, 64, 128):
        expected = query_l2_kcenter_seed_rows_pytorch(
            query,
            candidate_count=6,
            sketch_dimensions=sketch_dimensions,
        )
        for block_j in (1, 2, 4, 8, 16, 32):
            actual = query_l2_kcenter_seed_rows_triton(
                query,
                candidate_count=6,
                sketch_dimensions=sketch_dimensions,
                block_j=block_j,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(
                f"D={sketch_dimensions} BLOCK_J={block_j}: "
                f"{actual.numel():,}/{actual.numel():,} rows exact"
            )


if __name__ == "__main__":
    main()
