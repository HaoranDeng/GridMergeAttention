"""GPU parity check for the production mean-centered refinement kernel."""

from __future__ import annotations

import argparse
import json

import torch

from analysis.probe_hierarchical_split_gma_real_qkv import (
    hierarchical_split_outputs,
)
from analysis.probe_query_centroid_gma_real_qkv import fixed_seed_queries
from attention.hierarchical_c4_wan21_14b_720p_triton import (
    hard_c4_720p_hierarchical_mean_split_attention_triton_blocks,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (1, 21, 45, 80, 1, 128)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25
    key = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25
    value = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.25
    scale = 128**-0.5

    production = hard_c4_720p_hierarchical_mean_split_attention_triton_blocks(
        query,
        key,
        value,
        scale=scale,
        head_count=1,
    )
    query_frame = query[:, :1]
    seeds = fixed_seed_queries(query_frame, key.size(1), key.size(2))
    reference = hierarchical_split_outputs(
        query_frame,
        key,
        value,
        seeds,
        scale,
        refinements=(2,),
    )["tangent_mean_split_r2"]
    actual = production[:, :1]
    difference = (actual.float() - reference.float()).abs()
    torch.testing.assert_close(
        actual.float(), reference.float(), atol=args.atol, rtol=args.rtol
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": list(shape),
                "max_abs_error": float(difference.max().item()),
                "mean_abs_error": float(difference.mean().item()),
                "atol": args.atol,
                "rtol": args.rtol,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
