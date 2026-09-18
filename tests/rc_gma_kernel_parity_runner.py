"""One-head numerical parity check for the specialized 720p RC-GMA kernel."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_responsibility_centroid_attention_triton_blocks,
)
from attention.hard_route_pytorch import (
    gather_seed_queries,
    hard_route_responsibility_centroid_from_seeds_pytorch,
    wide_local_c4_seed_rows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (1, 21, 45, 80, 1, 128)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16)
    key = torch.randn(
        (1, args.key_frames, 45, 80, 1, 128),
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    scale = 128**-0.5

    actual = hard_c4_720p_responsibility_centroid_attention_triton_blocks(
        query, key, value, scale=scale, head_count=1
    )
    seed_rows = wide_local_c4_seed_rows(
        torch.arange(21, device=device),
        torch.arange(args.key_frames, device=device),
        45,
        80,
        45,
    )
    routing_prototypes = gather_seed_queries(query, seed_rows)
    expected = hard_route_responsibility_centroid_from_seeds_pytorch(
        query,
        key,
        value,
        routing_prototypes,
        scale,
        top_r=2,
    )

    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    result = {
        "key_frames": args.key_frames,
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "cosine": float(
            F.cosine_similarity(
                actual_f32.flatten(), expected_f32.flatten(), dim=0
            )
        ),
        "actual_finite": bool(torch.isfinite(actual_f32).all()),
        "expected_finite": bool(torch.isfinite(expected_f32).all()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["actual_finite"] or not result["expected_finite"]:
        raise SystemExit("RC-GMA parity produced non-finite output")


if __name__ == "__main__":
    main()
