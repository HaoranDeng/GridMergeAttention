"""Check the batched entropy seed oracle against the offline evaluator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from attention.adaptive_seed_pytorch import mass_weighted_reverse_kl_seed_rows
from attention.token_merging import _hard_c4_attention
from evaluate_entropy_seed_oracle import (
    COLS,
    FRAMES,
    ROWS,
    select_entropy_medoid_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--router_frame", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.qkv) as payload:
        query = torch.from_numpy(payload["q"][0, args.head].copy()).reshape(
            FRAMES, ROWS, COLS, -1
        )
        key = torch.from_numpy(payload["k"][0, args.head].copy()).reshape(
            FRAMES, ROWS, COLS, -1
        )
        value = torch.from_numpy(payload["v"][0, args.head].copy()).reshape(
            FRAMES, ROWS, COLS, -1
        )
    query = query.to(args.device)
    key = key.to(args.device)
    value = value.to(args.device)

    native_query = query[args.router_frame : args.router_frame + 1]
    native_query = native_query.unsqueeze(0).unsqueeze(-2)
    native_key = key.unsqueeze(0).unsqueeze(-2)
    torch.cuda.synchronize()
    started = time.perf_counter()
    native = mass_weighted_reverse_kl_seed_rows(
        native_query,
        native_key,
        candidate_count=32,
        candidate_geometry_count=45,
        head_chunk=1,
        block_chunk=21,
    )
    torch.cuda.synchronize()
    native_seconds = time.perf_counter() - started
    native_rows = (
        native[0, 0, ..., 0, :]
        .reshape(FRAMES * ROWS, COLS, 4)
        .permute(1, 0, 2)
        .to(torch.long)
    )

    offline, _, _ = select_entropy_medoid_rows(
        query[args.router_frame].permute(1, 0, 2),
        key.reshape(FRAMES * ROWS, COLS, -1),
        candidate_count=32,
        candidate_geometry_count=45,
        block_chunk=21,
    )
    offline_rows = offline["reverse_kl_mass"]
    exact_fraction = float((native_rows == offline_rows).float().mean())
    set_overlap = float(
        (
            native_rows[..., :, None] == offline_rows[..., None, :]
        ).any(-1).float().sum(-1).mean()
    )
    print(
        {
            "native_seconds_one_head": native_seconds,
            "ordered_seed_fraction": exact_fraction,
            "mean_set_overlap_of_four": set_overlap,
            "native_shape": list(native.shape),
        }
    )
    if exact_fraction != 1.0:
        raise SystemExit("batched selector does not exactly match offline evaluator")

    output = _hard_c4_attention(
        query.reshape(1, FRAMES * ROWS * COLS, 1, -1),
        key.reshape(1, FRAMES * ROWS * COLS, 1, -1),
        value.reshape(1, FRAMES * ROWS * COLS, 1, -1),
        h=ROWS,
        w=COLS,
        h_reduce=1,
        w_reduce=1,
        sm_scale=query.size(-1) ** -0.5,
        query_outer_chunk=FRAMES,
        head_chunk=1,
        hard_route_backend="hard_c4",
        exact_intra_frame_attention=False,
        exact_neighbor_window_columns=0,
        exact_query_rows=0,
        first_frame_full_attention=False,
        full_attention_anchor_time_count=0,
        full_attention_anchor_row_count=0,
        full_attention_anchor_scale=1.0,
        seed_mode="entropy_reverse_kl_mass_c32_t1_rf10_cg45",
    )
    print(
        {
            "hard_c4_output_shape": list(output.shape),
            "hard_c4_output_finite": bool(torch.isfinite(output).all()),
        }
    )
    if output.shape != (1, FRAMES * ROWS * COLS, 1, query.size(-1)):
        raise SystemExit("unexpected hard-C4 oracle output shape")
    if not torch.isfinite(output).all():
        raise SystemExit("hard-C4 oracle output contains non-finite values")


if __name__ == "__main__":
    main()
