"""Exercise D32 query-only seeds through both external-table C4 kernels."""

from __future__ import annotations

import argparse

import torch

from attention.adaptive_seed_triton import (
    adaptive_seed_attention_triton,
    query_l2_kcenter_seed_rows_triton,
)
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_native_seed_rows_attention_triton_blocks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    generator = torch.Generator(device=args.device).manual_seed(20260828)
    shape = (1, 21, 45, 80, 1, 128)
    query = torch.randn(
        shape,
        device=args.device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        (1, 1, 45, 80, 1, 128),
        device=args.device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    value = torch.randn(
        key.shape,
        device=args.device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    rows = query_l2_kcenter_seed_rows_triton(
        query,
        candidate_count=6,
        sketch_dimensions=32,
        block_j=8,
    ).unsqueeze(2)

    native = hard_c4_720p_native_seed_rows_attention_triton_blocks(
        query, key, value, rows
    )
    reference = adaptive_seed_attention_triton(query, key, value, rows)
    difference = (native.float() - reference.float()).flatten()
    relative_l2 = difference.norm() / reference.float().flatten().norm()
    maximum = difference.abs().max()
    torch.testing.assert_close(native, reference, rtol=3e-2, atol=3e-2)
    print(
        f"native external-table parity: rel_l2={relative_l2.item():.8f} "
        f"max_abs={maximum.item():.8f}"
    )


if __name__ == "__main__":
    main()
