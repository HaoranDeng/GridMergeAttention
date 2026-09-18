"""GPU parity check for the production Bi-Axial C3 path."""

from __future__ import annotations

import argparse
import json

import torch

from analysis.probe_biaxial_gma_real_qkv import (
    axis_moments_for_frame,
    fuse_moments,
)
from attention.token_merging import _biaxial_c3_attention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query_frame", type=int, default=0)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (1, 21, 45, 80, 1, 128)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    k = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    v = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.2
    scale = 128**-0.5

    row, row_log_z = axis_moments_for_frame(
        q.reshape(1, -1, 1, 128),
        k.reshape(1, -1, 1, 128),
        v.reshape(1, -1, 1, 128),
        (21, 45, 80),
        args.query_frame,
        scale,
        q_axis="h",
        k_axis="w",
        clusters=3,
    )
    column, column_log_z = axis_moments_for_frame(
        q.reshape(1, -1, 1, 128),
        k.reshape(1, -1, 1, 128),
        v.reshape(1, -1, 1, 128),
        (21, 45, 80),
        args.query_frame,
        scale,
        q_axis="w",
        k_axis="h",
        clusters=3,
    )
    reference, _ = fuse_moments(
        row, row_log_z, column, column_log_z
    )

    actual = _biaxial_c3_attention(
        q.reshape(1, -1, 1, 128),
        k.reshape(1, -1, 1, 128),
        v.reshape(1, -1, 1, 128),
        h=45,
        w=80,
        sm_scale=scale,
        query_outer_chunk=21,
        head_chunk=1,
    )
    frame_tokens = 45 * 80
    start = args.query_frame * frame_tokens
    actual = actual[:, start : start + frame_tokens]
    difference = actual.float() - reference.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1),
        reference.float().reshape(1, -1),
    ).item()
    payload = {
        "device": torch.cuda.get_device_name(0),
        "shape": list(shape),
        "query_frame": args.query_frame,
        "cosine": cosine,
        "mean_abs": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "max_abs": difference.abs().max().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    print(json.dumps(payload, indent=2))
    if not payload["finite"] or cosine < 0.99:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
