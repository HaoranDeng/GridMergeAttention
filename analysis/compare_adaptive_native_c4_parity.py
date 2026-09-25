"""Compare adaptive and native Hard-C4 merge/route kernels at fixed seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import adaptive_seed_attention_triton
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_native_seed_rows_attention_triton_blocks,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (1, 21, 45, 80, args.heads, 128)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16)
    key = torch.randn(shape, device=device, dtype=torch.bfloat16)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16)
    seed_rows = torch.randint(
        0,
        45,
        (1, 21, 21, 45, 80, args.heads, 4),
        device=device,
        dtype=torch.int32,
    )

    adaptive_frames = []
    for frame in range(21):
        adaptive_frames.append(
            adaptive_seed_attention_triton(
                query[:, frame : frame + 1],
                key,
                value,
                seed_rows[:, frame : frame + 1],
            )
        )
    adaptive = torch.cat(adaptive_frames, dim=1)
    native = hard_c4_720p_native_seed_rows_attention_triton_blocks(
        query,
        key,
        value,
        seed_rows.to(torch.uint8),
        head_count=args.heads,
    )
    torch.cuda.synchronize()

    difference = (adaptive.float() - native.float()).abs()
    reference_norm = adaptive.float().square().mean().sqrt()
    result = {
        "shape": list(shape),
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
        "relative_l2": float(
            difference.square().mean().sqrt() / reference_norm.clamp_min(1e-12)
        ),
        "adaptive_mean_abs": float(adaptive.float().abs().mean()),
        "native_mean_abs": float(native.float().abs().mean()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
