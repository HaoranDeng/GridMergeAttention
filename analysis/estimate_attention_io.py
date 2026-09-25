"""Estimate logical HBM traffic for Wan dense and hard-C4 attention.

This is a first-order model: it counts tensor elements moved between HBM and a
Triton program and ignores L2 cache reuse, masked-lane implementation details,
and compiler scheduling. The goal is to size optimization opportunities before
writing another kernel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import ceil


def parse_int_list(text):
    return [int(value) for value in text.replace(",", " ").split() if value.strip()]


@dataclass(frozen=True)
class Traffic:
    name: str
    bytes: int

    @property
    def gb(self) -> float:
        return self.bytes / 1e9

    def qkv128(self, dtype_bytes: int) -> float:
        return self.bytes / (128 * dtype_bytes)


def hard_c4_traffic(args):
    b, outer, rows, cols, heads, dim, c = (
        args.batch,
        args.outer,
        args.rows,
        args.cols,
        args.heads,
        args.dim,
        args.c,
    )
    dtype_bytes = args.dtype_bytes
    bias_bytes = args.bias_bytes
    merge_q_group = int(args.merge_q_group or outer)
    merge_q_groups = ceil(outer / merge_q_group)

    merge_seed_q = b * heads * outer * outer * rows * cols * c * dim * dtype_bytes
    merge_k = b * heads * merge_q_groups * outer * rows * cols * dim * dtype_bytes
    merge_v = merge_k
    inter_k = b * heads * outer * outer * rows * cols * c * dim * dtype_bytes
    inter_v = inter_k
    inter_bias = b * heads * outer * outer * rows * cols * c * bias_bytes
    route_q = b * heads * outer * cols * rows * dim * dtype_bytes
    output = route_q

    return [
        Traffic("hard.merge.seed_q_load", merge_seed_q),
        Traffic("hard.merge.k_load", merge_k),
        Traffic("hard.merge.v_load", merge_v),
        Traffic("hard.merge.intermediate_store", inter_k + inter_v + inter_bias),
        Traffic("hard.route.q_load", route_q),
        Traffic("hard.route.intermediate_load", inter_k + inter_v + inter_bias),
        Traffic("hard.route.output_store", output),
    ]


def flash_traffic(args):
    b, outer, rows, cols, heads, dim = (
        args.batch,
        args.outer,
        args.rows,
        args.cols,
        args.heads,
        args.dim,
    )
    dtype_bytes = args.dtype_bytes
    seq_len = outer * rows * cols
    q_blocks = ceil(seq_len / args.flash_block_m)
    q = b * heads * seq_len * dim * dtype_bytes
    k = b * heads * q_blocks * seq_len * dim * dtype_bytes
    v = k
    output = q
    return [
        Traffic("flash.q_load", q),
        Traffic("flash.k_load", k),
        Traffic("flash.v_load", v),
        Traffic("flash.output_store", output),
    ]


def fused_hard_c4_traffic(args, i_tile, j_tile):
    """Optimistic no-intermediate hard-C4 traffic for a query-column tile.

    The current merge kernel computes compressed K/V once and shares it across
    all 30 query rows. A fused kernel that consumes compressed K/V immediately
    avoids the intermediate tensor, but if it covers fewer than all query rows,
    it must recompute the same merge work for each query-row tile.
    """

    b, outer, rows, cols, heads, dim, c = (
        args.batch,
        args.outer,
        args.rows,
        args.cols,
        args.heads,
        args.dim,
        args.c,
    )
    dtype_bytes = args.dtype_bytes
    i_tiles = ceil(rows / i_tile)
    j_tiles = ceil(cols / j_tile)

    merge_seed_q = b * heads * outer * outer * rows * cols * c * dim * dtype_bytes * i_tiles
    merge_k = b * heads * outer * outer * rows * cols * dim * dtype_bytes * i_tiles * j_tiles
    merge_v = merge_k
    route_q = b * heads * outer * cols * rows * dim * dtype_bytes
    output = route_q
    return [
        Traffic(f"fused.i{i_tile}.j{j_tile}.seed_q_load", merge_seed_q),
        Traffic(f"fused.i{i_tile}.j{j_tile}.k_load", merge_k),
        Traffic(f"fused.i{i_tile}.j{j_tile}.v_load", merge_v),
        Traffic(f"fused.i{i_tile}.j{j_tile}.route_q_load", route_q),
        Traffic(f"fused.i{i_tile}.j{j_tile}.output_store", output),
    ]


def two_pass_hard_c4_traffic(args, i_tile, j_tile, d_tile):
    """D-tiled two-pass no-merged-KV traffic.

    Pass 1 computes the second-stage online-softmax row max/sum and stores only
    those two FP32 tensors. Pass 2 recomputes the hard-C4 merge and consumes V
    to form the output. This avoids the large merged K/V/bias tensors at the
    cost of repeated route work.

    If pass 2 is split into D-sized output chunks to avoid a full ``I x J x D``
    FP32 accumulator, each D chunk must recompute the full-D route logits unless
    another intermediate is introduced. Therefore seed-Q, K, route-Q, and stats
    loads are multiplied by the number of D chunks. V is only counted once in
    total because the chunks partition the value dimension.
    """

    b, outer, rows, cols, heads, dim, c = (
        args.batch,
        args.outer,
        args.rows,
        args.cols,
        args.heads,
        args.dim,
        args.c,
    )
    dtype_bytes = args.dtype_bytes
    f32 = 4
    i_tiles = ceil(rows / i_tile)
    j_tiles = ceil(cols / j_tile)
    d_tiles = ceil(dim / d_tile)

    seed_q = b * heads * outer * outer * rows * cols * c * dim * dtype_bytes * i_tiles
    key = b * heads * outer * outer * rows * cols * dim * dtype_bytes * i_tiles * j_tiles
    value = key
    route_q = b * heads * outer * cols * rows * dim * dtype_bytes
    output = route_q
    stats = b * heads * outer * rows * cols * 2 * f32

    return [
        Traffic(f"twopass.i{i_tile}.j{j_tile}.pass1.seed_q_load", seed_q),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.pass1.k_load", key),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.pass1.route_q_load", route_q),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.stats_store", stats),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.pass2.seed_q_load", seed_q * d_tiles),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.pass2.k_load", key * d_tiles),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.pass2.v_load", value),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.pass2.route_q_load", route_q * d_tiles),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.stats_load", stats * d_tiles),
        Traffic(f"twopass.i{i_tile}.j{j_tile}.output_store", output),
    ]


def summarize(rows, dtype_bytes):
    total = sum(row.bytes for row in rows)
    return {
        "total_gb": total / 1e9,
        "total_qkv128_m": total / (128 * dtype_bytes) / 1e6,
        "parts": [
            {
                "name": row.name,
                "gb": row.gb,
                "qkv128_m": row.qkv128(dtype_bytes) / 1e6,
                "fraction": row.bytes / total if total else 0.0,
            }
            for row in rows
        ],
    }


def fused_live_state(args, i_tile, j_tile):
    """Approximate per-program live tensor footprint for fused hard-C4.

    This is not an exact register/shared-memory report. It counts the main
    logical live tensors a fused design would want to keep at once:

    - seed queries for C seeds and J query columns,
    - one 52-key K/V row,
    - C seed-score matrices over J query columns and 52 keys,
    - query tile and FP32 route accumulator for I x J queries.
    """

    cols, dim, c = args.cols, args.dim, args.c
    dtype_bytes = args.dtype_bytes
    f32 = 4
    parts = [
        ("seed_q", c * j_tile * dim * dtype_bytes),
        ("key_row", cols * dim * dtype_bytes),
        ("value_row", cols * dim * dtype_bytes),
        ("seed_scores", c * j_tile * cols * f32),
        ("query_tile", i_tile * j_tile * dim * dtype_bytes),
        ("route_accumulator", i_tile * j_tile * dim * f32),
        ("route_stats", i_tile * j_tile * 2 * f32),
    ]
    total = sum(bytes_ for _, bytes_ in parts)
    total_kb = total / 1024
    return {
        "i_tile": i_tile,
        "j_tile": j_tile,
        "total_kb": total_kb,
        "h200_smem_fraction": total_kb / args.h200_smem_kb,
        "parts": [
            {"name": name, "kb": bytes_ / 1024, "fraction": bytes_ / total}
            for name, bytes_ in parts
        ],
    }


def two_pass_live_state(args, i_tile, j_tile, d_tile):
    """Approximate live tensors for the two-pass no-merged-KV design.

    The output pass can accumulate V in D-sized chunks while reusing the same
    route probabilities. That replaces the full FP32 ``I x J x D`` accumulator
    with an ``I x J x D_TILE`` accumulator, at the cost of more inner-loop work.
    """

    cols, dim, c = args.cols, args.dim, args.c
    dtype_bytes = args.dtype_bytes
    f32 = 4
    common = [
        ("seed_q", c * j_tile * dim * dtype_bytes),
        ("key_row", cols * dim * dtype_bytes),
        ("seed_scores", c * j_tile * cols * f32),
        ("query_tile", i_tile * j_tile * dim * dtype_bytes),
        ("merged_key", c * j_tile * dim * dtype_bytes),
        ("route_logits", i_tile * j_tile * c * f32),
        ("route_stats", i_tile * j_tile * 2 * f32),
    ]
    stats_pass = common
    output_pass = common + [
        ("value_row_chunk", cols * d_tile * dtype_bytes),
        ("merged_value_chunk", c * j_tile * d_tile * dtype_bytes),
        ("output_accumulator_chunk", i_tile * j_tile * d_tile * f32),
    ]

    def summarize_parts(parts):
        total = sum(bytes_ for _, bytes_ in parts)
        return {
            "total_kb": total / 1024,
            "h200_smem_fraction": (total / 1024) / args.h200_smem_kb,
            "parts": [
                {"name": name, "kb": bytes_ / 1024, "fraction": bytes_ / total}
                for name, bytes_ in parts
            ],
        }

    stats_summary = summarize_parts(stats_pass)
    output_summary = summarize_parts(output_pass)
    return {
        "i_tile": i_tile,
        "j_tile": j_tile,
        "d_tile": d_tile,
        "stats_pass": stats_summary,
        "output_pass": output_summary,
        "max_live_kb": max(stats_summary["total_kb"], output_summary["total_kb"]),
    }


def fused_tile_summary(args, flash_total_gb, hard_total_gb, i_tile, j_tile):
    traffic = summarize(fused_hard_c4_traffic(args, i_tile, j_tile), args.dtype_bytes)
    live = fused_live_state(args, i_tile, j_tile)
    return {
        "i_tile": i_tile,
        "j_tile": j_tile,
        "i_tiles": ceil(args.rows / i_tile),
        "j_tiles": ceil(args.cols / j_tile),
        "traffic_gb": traffic["total_gb"],
        "traffic_over_flash": traffic["total_gb"] / flash_total_gb,
        "traffic_over_hard": traffic["total_gb"] / hard_total_gb,
        "live_kb": live["total_kb"],
        "h200_smem_fraction": live["h200_smem_fraction"],
    }


def two_pass_tile_summary(
    args, flash_total_gb, hard_total_gb, i_tile, j_tile, d_tile
):
    traffic = summarize(two_pass_hard_c4_traffic(args, i_tile, j_tile, d_tile), args.dtype_bytes)
    live = two_pass_live_state(args, i_tile, j_tile, d_tile)
    output_live = live["output_pass"]
    return {
        "i_tile": i_tile,
        "j_tile": j_tile,
        "d_tile": d_tile,
        "i_tiles": ceil(args.rows / i_tile),
        "j_tiles": ceil(args.cols / j_tile),
        "traffic_gb": traffic["total_gb"],
        "traffic_over_flash": traffic["total_gb"] / flash_total_gb,
        "traffic_over_hard": traffic["total_gb"] / hard_total_gb,
        "max_live_kb": live["max_live_kb"],
        "h200_smem_fraction": output_live["h200_smem_fraction"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--outer", type=int, default=21)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--cols", type=int, default=52)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--c", type=int, default=4)
    parser.add_argument("--dtype_bytes", type=int, default=2)
    parser.add_argument("--bias_bytes", type=int, default=4)
    parser.add_argument("--flash_block_m", type=int, default=128)
    parser.add_argument(
        "--merge_q_group",
        type=int,
        default=0,
        help=(
            "Query frames sharing one original K/V load in the maintained merge "
            "kernel. Zero means all query frames."
        ),
    )
    parser.add_argument("--fused_j_tiles", default="1,4,8,16,32,52")
    parser.add_argument("--fused_i_tiles", default="8,16,30")
    parser.add_argument("--two_pass_d_tiles", default="8,16,32,64,128")
    parser.add_argument(
        "--h200_smem_kb",
        type=float,
        default=228.0,
        help="Approximate max shared memory per H200 SM available to one block.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    i_tiles = parse_int_list(args.fused_i_tiles)
    j_tiles = parse_int_list(args.fused_j_tiles)
    d_tiles = parse_int_list(args.two_pass_d_tiles)
    hard_rows = hard_c4_traffic(args)
    flash_rows = flash_traffic(args)
    hard = summarize(hard_rows, args.dtype_bytes)
    flash = summarize(flash_rows, args.dtype_bytes)
    hard_intermediate = sum(
        row.bytes for row in hard_rows if "intermediate" in row.name
    )
    payload = {
        "config": vars(args),
        "flash": flash,
        "hard_c4_two_stage": hard,
        "fused_hard_c4_no_intermediate": {},
        "two_pass_hard_c4_no_merged_kv": {},
        "fused_live_state": [
            fused_live_state(args, i_tile, j_tile)
            for i_tile in i_tiles
            for j_tile in j_tiles
        ],
        "two_pass_live_state": [
            two_pass_live_state(args, i_tile, j_tile, d_tile)
            for i_tile in i_tiles
            for j_tile in j_tiles
            for d_tile in d_tiles
        ],
        "fused_tile_summary": [],
        "plausible_fused_tiles": [],
        "two_pass_tile_summary": [],
        "plausible_two_pass_tiles": [],
        "ratios": {
            "hard_over_flash": hard["total_gb"] / flash["total_gb"],
            "hard_intermediate_fraction": hard_intermediate
            / sum(row.bytes for row in hard_rows),
            "hard_without_intermediate_over_flash": (
                sum(row.bytes for row in hard_rows) - hard_intermediate
            )
            / sum(row.bytes for row in flash_rows),
        },
    }
    for i_tile in i_tiles:
        for j_tile in j_tiles:
            key = f"i{i_tile}_j{j_tile}"
            payload["fused_hard_c4_no_intermediate"][key] = summarize(
                fused_hard_c4_traffic(args, i_tile, j_tile),
                args.dtype_bytes,
            )
            payload["fused_tile_summary"].append(
                fused_tile_summary(
                    args, flash["total_gb"], hard["total_gb"], i_tile, j_tile
                )
            )
            for d_tile in d_tiles:
                two_pass_key = f"{key}_d{d_tile}"
                payload["two_pass_hard_c4_no_merged_kv"][two_pass_key] = summarize(
                    two_pass_hard_c4_traffic(args, i_tile, j_tile, d_tile),
                    args.dtype_bytes,
                )
                payload["two_pass_tile_summary"].append(
                    two_pass_tile_summary(
                        args,
                        flash["total_gb"],
                        hard["total_gb"],
                        i_tile,
                        j_tile,
                        d_tile,
                    )
                )
    payload["fused_tile_summary"].sort(
        key=lambda row: (row["traffic_over_flash"], row["h200_smem_fraction"])
    )
    payload["plausible_fused_tiles"] = [
        row
        for row in payload["fused_tile_summary"]
        if row["traffic_over_hard"] <= 1.0 and row["h200_smem_fraction"] <= 1.0
    ]
    payload["two_pass_tile_summary"].sort(
        key=lambda row: (row["traffic_over_flash"], row["h200_smem_fraction"])
    )
    payload["plausible_two_pass_tiles"] = [
        row
        for row in payload["two_pass_tile_summary"]
        if row["traffic_over_hard"] <= 1.0 and row["h200_smem_fraction"] <= 1.0
    ]

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(json.dumps(payload["config"], indent=2, sort_keys=True))
    for section in ("flash", "hard_c4_two_stage"):
        print(f"\n{section}: {payload[section]['total_gb']:.3f} GB")
        for part in payload[section]["parts"]:
            print(
                f"  {part['name']:36s}"
                f"{part['gb']:8.3f} GB"
                f"{part['qkv128_m']:9.3f} M qkv128"
                f"{part['fraction'] * 100:7.2f}%"
            )
    print("\nfused_hard_c4_no_intermediate:")
    for key, section in payload["fused_hard_c4_no_intermediate"].items():
        print(
            f"  {key:8s}"
            f"  {section['total_gb']:8.3f} GB"
            f"  {section['total_qkv128_m']:8.3f} M qkv128"
            f"  vs_flash={section['total_gb'] / flash['total_gb']:.3f}x"
        )
    print("\nfused_live_state_approx:")
    for row in payload["fused_live_state"]:
        print(
            f"  I_TILE={row['i_tile']:2d}"
            f" J_TILE={row['j_tile']:2d}"
            f"  live={row['total_kb']:8.1f} KiB"
            f"  h200_smem={row['h200_smem_fraction'] * 100:6.1f}%"
        )
    print("\nplausible_fused_tiles:")
    if not payload["plausible_fused_tiles"]:
        print("  none under both traffic<=retained-hard-C4 and live<=H200 smem")
    for row in payload["plausible_fused_tiles"]:
        print(
            f"  I_TILE={row['i_tile']:2d}"
            f" J_TILE={row['j_tile']:2d}"
            f"  traffic={row['traffic_gb']:8.3f} GB"
            f"  vs_flash={row['traffic_over_flash']:6.3f}x"
            f"  vs_hard={row['traffic_over_hard']:6.3f}x"
            f"  live={row['live_kb']:8.1f} KiB"
            f"  h200_smem={row['h200_smem_fraction'] * 100:6.1f}%"
        )
    print("\nplausible_two_pass_tiles:")
    if not payload["plausible_two_pass_tiles"]:
        print("  none under both traffic<=retained-hard-C4 and live<=H200 smem")
    for row in payload["plausible_two_pass_tiles"]:
        print(
            f"  I_TILE={row['i_tile']:2d}"
            f" J_TILE={row['j_tile']:2d}"
            f" D_TILE={row['d_tile']:2d}"
            f"  traffic={row['traffic_gb']:8.3f} GB"
            f"  vs_flash={row['traffic_over_flash']:6.3f}x"
            f"  vs_hard={row['traffic_over_hard']:6.3f}x"
            f"  live={row['max_live_kb']:8.1f} KiB"
            f"  h200_smem={row['h200_smem_fraction'] * 100:6.1f}%"
        )
    print("\nratios:")
    for name, value in payload["ratios"].items():
        print(f"  {name:40s}{value:.4f}")


if __name__ == "__main__":
    main()
