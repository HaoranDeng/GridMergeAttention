"""Compile and validate the experimental 720p Hard-C4 grid kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention.hard_c4_grid_wan21_14b_720p_triton import (
    hard_c4_grid24x80_720p_attention_triton_head_major,
    hard_c4_grid48x40_720p_attention_triton_head_major,
    hard_c4_grid48x80_720p_attention_triton_head_major,
)
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_attention_triton_head_major,
)


def timed(function):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return output, start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    shape = (1, 1, 21 * 45 * 80, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    reference, reference_ms = timed(
        lambda: hard_c4_720p_attention_triton_head_major(q, k, v)
    )
    exact, exact_ms = timed(
        lambda: hard_c4_grid48x80_720p_attention_triton_head_major(q, k, v)
    )
    grid24x80, grid24x80_ms = timed(
        lambda: hard_c4_grid24x80_720p_attention_triton_head_major(q, k, v)
    )
    grid48x40, grid48x40_ms = timed(
        lambda: hard_c4_grid48x40_720p_attention_triton_head_major(q, k, v)
    )

    difference = exact.float() - reference.float()
    payload = {
        "device": torch.cuda.get_device_name(),
        "reference_ms": reference_ms,
        "grid48x80_ms": exact_ms,
        "grid48x80_vs_reference": {
            "rel_l2": float(
                difference.norm() / reference.float().norm().clamp_min(1e-20)
            ),
            "mse": float(difference.square().mean()),
            "max_abs": float(difference.abs().max()),
        },
        "grid24x80_ms": grid24x80_ms,
        "grid24x80_nonfinite": int((~grid24x80.isfinite()).sum()),
        "grid48x40_ms": grid48x40_ms,
        "grid48x40_nonfinite": int((~grid48x40.isfinite()).sum()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
