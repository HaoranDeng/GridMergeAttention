"""CUDA parity check for the query-response C4 seed selector."""

from __future__ import annotations

import argparse

import torch

from analysis.sweep_wan14b_720p_hard_c4_algorithm import (
    VARIANTS,
    block_response_seeds,
    hard_c4_sample,
)
from attention.hard_c4_wan21_14b_720p_triton import (
    HARD_720_C,
    HARD_720_COLS,
    HARD_720_ROWS,
    _query_response_seed_rows_720p,
    hard_c4_720p_attention_triton_blocks,
    hard_c4_720p_query_response_attention_triton_blocks,
)


def check_seed_rows():
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, query_outer, key_outer, heads, dim = 1, 1, 2, 1, 128
    query = torch.randn(
        batch,
        query_outer,
        HARD_720_ROWS,
        HARD_720_COLS,
        heads,
        dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        batch,
        key_outer,
        HARD_720_ROWS,
        HARD_720_COLS,
        heads,
        dim,
        device=device,
        dtype=dtype,
    )
    scale = dim**-0.5
    config = {
        **VARIANTS["qblock_response15_s8_span_prob_anchor_left"],
        "return_seed_indices": True,
    }
    _, expected = block_response_seeds(query, key, key, scale, config)
    actual = _query_response_seed_rows_720p(query, key, scale=scale).view(
        batch,
        query_outer,
        HARD_720_COLS,
        heads,
        key_outer,
        HARD_720_ROWS,
        HARD_720_C,
    ).permute(0, 1, 4, 5, 2, 3, 6)

    mismatch = actual.long() != expected.long()
    mismatch_count = int(mismatch.sum())
    total = mismatch.numel()
    if mismatch_count:
        location = mismatch.nonzero()[0].tolist()
        raise AssertionError(
            f"seed mismatch {mismatch_count}/{total}; first={location}, "
            f"actual={int(actual[tuple(location)])}, "
            f"expected={int(expected[tuple(location)])}"
        )
    print(f"query-response seed parity: {total}/{total} exact")


def check_full_attention():
    torch.manual_seed(20260823)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, query_outer, key_outer, heads, dim = 1, 21, 1, 1, 128
    shape = (batch, query_outer, HARD_720_ROWS, HARD_720_COLS, heads, dim)
    query = torch.randn(*shape, device=device, dtype=dtype)
    key = torch.randn(
        batch,
        key_outer,
        HARD_720_ROWS,
        HARD_720_COLS,
        heads,
        dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    scale = dim**-0.5
    actual = hard_c4_720p_query_response_attention_triton_blocks(
        query, key, value, scale=scale, head_count=1
    )
    config = {
        **VARIANTS["qblock_response15_s8_span_prob_anchor_left"],
        "bias_scale": 1.2,
    }
    expected, _ = hard_c4_sample(query, key, value, scale, config)
    default_actual = hard_c4_720p_attention_triton_blocks(
        query, key, value, scale=scale, head_count=1
    )
    default_expected, _ = hard_c4_sample(
        query, key, value, scale, VARIANTS["seed_wide"]
    )
    torch.cuda.synchronize()
    default_difference = default_actual.float() - default_expected.float()
    default_mse = float(default_difference.square().mean())
    default_cosine = float(
        torch.nn.functional.cosine_similarity(
            default_actual.float().flatten(),
            default_expected.float().flatten(),
            dim=0,
        )
    )
    print(
        "default wide-C4 full attention parity: "
        f"mse={default_mse:.8g}, cosine={default_cosine:.8g}"
    )
    difference = actual.float() - expected.float()
    mse = float(difference.square().mean())
    max_abs = float(difference.abs().max())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
    )
    if mse > 2.0e-5 or cosine < 0.999:
        raise AssertionError(
            f"full attention mismatch: mse={mse}, max_abs={max_abs}, "
            f"cosine={cosine}"
        )
    print(
        "query-response full attention parity: "
        f"mse={mse:.8g}, max_abs={max_abs:.8g}, cosine={cosine:.8g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    check_seed_rows()
    if args.full:
        check_full_attention()


if __name__ == "__main__":
    main()
