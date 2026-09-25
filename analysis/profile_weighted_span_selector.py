"""Profile the production per-frame Weighted-Span selector stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import span_seed_rows_triton_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--candidate_count", type=int, default=45)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.qkv) as payload:
        tensors = []
        for name in ("q", "k", "v"):
            source = torch.from_numpy(payload[name][0].copy())
            source = source.reshape(source.shape[0], 21, 45, 80, -1)
            source = source[:1].permute(1, 2, 3, 0, 4).unsqueeze(0)
            tensors.append(source.repeat(1, 1, 1, 1, args.heads, 1).to(args.device))

    def run() -> torch.Tensor:
        return span_seed_rows_triton_split(
            *tensors,
            alpha=0.3,
            candidate_count=args.candidate_count,
            top_probes=8,
            uniform_probes=8,
            value_dims=8,
            block_j=4,
            group_tile=16,
            key_outer_tile=7,
            output_dtype=torch.uint8,
            probe_num_warps=4,
            gs_num_warps=8,
        )

    for _ in range(args.warmup):
        output = run()
        del output
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        output = run()
        del output
        torch.cuda.synchronize()

    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=12
        )
    )


if __name__ == "__main__":
    main()
