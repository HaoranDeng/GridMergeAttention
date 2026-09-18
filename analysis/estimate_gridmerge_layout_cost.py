"""Estimate IO and MAC costs for candidate GridMerge layouts.

This is a static feasibility model for Wan 480p. It does not replace measured
H200 benchmarks; it helps decide whether a new layout is worth a tuned kernel.
The model covers layouts of the forms ``qcX_krY_cZ`` and ``qfX_kfY_cZ``:

``qc``: adjacent query columns grouped into one query block.
``kr``: adjacent key rows grouped into one key block.
``qf``: adjacent query frames grouped into one query block.
``kf``: adjacent key frames grouped into one key block.
``c``:  compressed K/V tokens per grid block.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path


ROWCOL_VARIANT_RE = re.compile(r"qc(?P<qcols>\d+)_kr(?P<krows>\d+)_c(?P<c>\d+)$")
TEMPORAL_VARIANT_RE = re.compile(r"qf(?P<qframes>\d+)_kf(?P<kframes>\d+)_c(?P<c>\d+)$")


@dataclass(frozen=True)
class Shape:
    batch: int
    outer: int
    rows: int
    cols: int
    heads: int
    dim: int
    dtype_bytes: int
    bias_bytes: int
    flash_block_m: int
    merge_group_tile: int

    @property
    def tokens(self) -> int:
        return self.outer * self.rows * self.cols


def parse_variant(name: str) -> dict:
    if name.startswith("c") and name[1:].isdigit():
        return {"name": name, "kind": "rowcol", "qcols": 1, "krows": 1, "c": int(name[1:])}
    rowcol = ROWCOL_VARIANT_RE.fullmatch(name)
    if rowcol:
        return {
            "name": name,
            "kind": "rowcol",
            "qcols": int(rowcol.group("qcols")),
            "krows": int(rowcol.group("krows")),
            "c": int(rowcol.group("c")),
        }
    temporal = TEMPORAL_VARIANT_RE.fullmatch(name)
    if temporal:
        return {
            "name": name,
            "kind": "temporal",
            "qframes": int(temporal.group("qframes")),
            "kframes": int(temporal.group("kframes")),
            "c": int(temporal.group("c")),
        }
    raise ValueError(f"Variant must look like c4, qc1_kr2_c8, or qf1_kf7_c7, got {name!r}")


def gb(bytes_: float) -> float:
    return bytes_ / 1e9


def mac_t(mac: float) -> float:
    return mac / 1e12


def flash(shape: Shape) -> dict:
    q_blocks = ceil(shape.tokens / shape.flash_block_m)
    q = shape.batch * shape.heads * shape.tokens * shape.dim * shape.dtype_bytes
    kv = shape.batch * shape.heads * q_blocks * shape.tokens * shape.dim * shape.dtype_bytes
    mac = 2 * shape.batch * shape.heads * shape.tokens * shape.tokens * shape.dim
    return {
        "io_bytes": q + 2 * kv + q,
        "mac": mac,
        "q_load_bytes": q,
        "k_load_bytes": kv,
        "v_load_bytes": kv,
        "out_store_bytes": q,
    }


def gridmerge(shape: Shape, spec: dict) -> dict:
    qcols, krows, c = spec["qcols"], spec["krows"], spec["c"]
    valid = shape.cols % qcols == 0 and shape.rows % krows == 0
    if not valid:
        return {**spec, "valid": False}

    query_groups = shape.cols // qcols
    key_groups = shape.rows // krows
    query_tokens_per_block = shape.rows * qcols
    key_tokens_per_block = shape.cols * krows
    merge_tiles = ceil(query_groups / shape.merge_group_tile)
    block_tiles = shape.batch * shape.heads * shape.outer * shape.outer * key_groups * merge_tiles
    blocks = shape.batch * shape.heads * shape.outer * shape.outer * query_groups * key_groups
    compressed_tokens = blocks * c
    vector_bytes = shape.dim * shape.dtype_bytes

    merge_seed_q = blocks * c * vector_bytes
    merge_k = block_tiles * key_tokens_per_block * vector_bytes
    merge_v = merge_k
    intermediate = compressed_tokens * (2 * vector_bytes + shape.bias_bytes)
    route_q = shape.batch * shape.heads * shape.tokens * vector_bytes
    out = route_q

    merge_seed_mac = blocks * c * key_tokens_per_block * shape.dim
    merge_kv_mac = 2 * merge_seed_mac
    route_mac = (
        2
        * shape.batch
        * shape.heads
        * shape.outer
        * query_groups
        * query_tokens_per_block
        * shape.outer
        * key_groups
        * c
        * shape.dim
    )

    return {
        **spec,
        "valid": True,
        "query_block_tokens": query_tokens_per_block,
        "key_block_tokens": key_tokens_per_block,
        "query_groups_per_frame": query_groups,
        "key_groups_per_frame": key_groups,
        "compressed_token_density": c / (krows * shape.cols),
        "paper_density_c": c
        * (query_tokens_per_block + key_tokens_per_block)
        / (query_tokens_per_block * key_tokens_per_block),
        "two_stage_io_bytes": merge_seed_q + merge_k + merge_v + 2 * intermediate + route_q + out,
        "no_intermediate_io_bytes": merge_seed_q + merge_k + merge_v + route_q + out,
        "intermediate_io_bytes": 2 * intermediate,
        "merge_mac": merge_seed_mac + merge_kv_mac,
        "route_mac": route_mac,
        "total_mac": merge_seed_mac + merge_kv_mac + route_mac,
        "parts": {
            "merge_seed_q_load_gb": gb(merge_seed_q),
            "merge_k_load_gb": gb(merge_k),
            "merge_v_load_gb": gb(merge_v),
            "intermediate_store_load_gb": gb(2 * intermediate),
            "route_q_load_gb": gb(route_q),
            "out_store_gb": gb(out),
        },
    }


def temporal_gridmerge(shape: Shape, spec: dict) -> dict:
    qframes, kframes, c = spec["qframes"], spec["kframes"], spec["c"]
    valid = shape.outer % qframes == 0 and shape.outer % kframes == 0
    if not valid:
        return {**spec, "valid": False}

    query_frame_groups = shape.outer // qframes
    key_frame_groups = shape.outer // kframes
    query_groups = query_frame_groups * shape.cols
    key_groups = key_frame_groups * shape.rows
    query_tokens_per_block = qframes * shape.rows
    key_tokens_per_block = kframes * shape.cols
    merge_tiles = ceil(shape.cols / shape.merge_group_tile)
    block_tiles = shape.batch * shape.heads * query_frame_groups * key_frame_groups * shape.rows * merge_tiles
    blocks = shape.batch * shape.heads * query_groups * key_groups
    compressed_tokens = blocks * c
    vector_bytes = shape.dim * shape.dtype_bytes

    merge_seed_q = blocks * c * vector_bytes
    merge_k = block_tiles * key_tokens_per_block * vector_bytes
    merge_v = merge_k
    intermediate = compressed_tokens * (2 * vector_bytes + shape.bias_bytes)
    route_q = shape.batch * shape.heads * shape.tokens * vector_bytes
    out = route_q

    merge_seed_mac = blocks * c * key_tokens_per_block * shape.dim
    merge_kv_mac = 2 * merge_seed_mac
    route_mac = (
        2
        * shape.batch
        * shape.heads
        * query_groups
        * query_tokens_per_block
        * key_groups
        * c
        * shape.dim
    )

    return {
        **spec,
        "valid": True,
        "query_block_tokens": query_tokens_per_block,
        "key_block_tokens": key_tokens_per_block,
        "query_groups_per_video": query_groups,
        "key_groups_per_video": key_groups,
        "compressed_token_density": c / (kframes * shape.cols),
        "paper_density_c": c
        * (query_tokens_per_block + key_tokens_per_block)
        / (query_tokens_per_block * key_tokens_per_block),
        "two_stage_io_bytes": merge_seed_q + merge_k + merge_v + 2 * intermediate + route_q + out,
        "no_intermediate_io_bytes": merge_seed_q + merge_k + merge_v + route_q + out,
        "intermediate_io_bytes": 2 * intermediate,
        "merge_mac": merge_seed_mac + merge_kv_mac,
        "route_mac": route_mac,
        "total_mac": merge_seed_mac + merge_kv_mac + route_mac,
        "parts": {
            "merge_seed_q_load_gb": gb(merge_seed_q),
            "merge_k_load_gb": gb(merge_k),
            "merge_v_load_gb": gb(merge_v),
            "intermediate_store_load_gb": gb(2 * intermediate),
            "route_q_load_gb": gb(route_q),
            "out_store_gb": gb(out),
        },
    }


def estimate_variant(shape: Shape, spec: dict) -> dict:
    if spec.get("kind") == "temporal":
        return temporal_gridmerge(shape, spec)
    return gridmerge(shape, spec)


def add_ratios(row: dict, flash_row: dict, c4_row: dict) -> dict:
    if not row.get("valid"):
        return row
    row = dict(row)
    row.update(
        {
            "two_stage_io_gb": gb(row["two_stage_io_bytes"]),
            "no_intermediate_io_gb": gb(row["no_intermediate_io_bytes"]),
            "intermediate_io_gb": gb(row["intermediate_io_bytes"]),
            "two_stage_io_vs_flash": row["two_stage_io_bytes"] / flash_row["io_bytes"],
            "no_intermediate_io_vs_flash": row["no_intermediate_io_bytes"] / flash_row["io_bytes"],
            "merge_mac_t": mac_t(row["merge_mac"]),
            "route_mac_t": mac_t(row["route_mac"]),
            "total_mac_t": mac_t(row["total_mac"]),
            "merge_mac_vs_c4": row["merge_mac"] / c4_row["merge_mac"],
            "route_mac_vs_c4": row["route_mac"] / c4_row["route_mac"],
            "total_mac_vs_c4": row["total_mac"] / c4_row["total_mac"],
            "total_mac_vs_flash": row["total_mac"] / flash_row["mac"],
        }
    )
    return row


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown(rows: list[dict], output_md: Path) -> None:
    header = (
        "| Variant | q x k block | C | density | paper density | two-stage IO / flash | "
        "no-intermediate IO / flash | merge MAC / C4 | route MAC / C4 | total MAC / flash |\n"
    )
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    body = []
    for row in rows:
        if not row.get("valid"):
            body.append(f"| `{row['name']}` | invalid | {row['c']} | - | - | - | - | - | - | - |")
            continue
        body.append(
            f"| `{row['name']}` | {row['query_block_tokens']} x {row['key_block_tokens']} | "
            f"{row['c']} | {fmt(row['compressed_token_density'], 5)} | "
            f"{fmt(row['paper_density_c'], 5)} | {fmt(row['two_stage_io_vs_flash'], 3)}x | "
            f"{fmt(row['no_intermediate_io_vs_flash'], 3)}x | {fmt(row['merge_mac_vs_c4'], 2)}x | "
            f"{fmt(row['route_mac_vs_c4'], 2)}x | {fmt(row['total_mac_vs_flash'], 3)}x |"
        )

    text = [
        "# GridMerge Layout Cost Estimate",
        "",
        "Static Wan 480p estimate for candidate GridMerge layouts. This is a feasibility",
        "model, not a measured runtime report. It ignores cache reuse, occupancy,",
        "mask overhead, and scalar softmax costs.",
        "",
        header + sep + "\n".join(body),
        "",
        "Interpretation:",
        "",
        "- `two-stage IO / flash` includes current intermediate compressed K/V/bias store and load.",
        "- `no-intermediate IO / flash` is an optimistic lower bound for a perfect fused design.",
        "- `merge MAC / C4` and `route MAC / C4` show why some better-quality layouts are hard to speed up.",
        "",
        "Main signal:",
        "",
        "- `qc1_kr2_c8` has strong analysis-only quality, but it doubles merge MAC versus C4 and has no IO advantage in this model.",
        "- `qc2_kr2_c8` keeps C4-like main MAC and lower idealized IO, but the tuned H200 kernel did not realize the speedup.",
        "- Temporal-key grouping is quality-interesting at very low compressed-token density: `qf1_kf7_c7` keeps C1 density while reaching hard-C4-like error in PyTorch screening.",
        "- The temporal-key tradeoff is merge-heavy. `qf1_kf7_c7` has only `0.25x` C4 route MAC, but `1.75x` C4 merge MAC; `qf1_kf7_c5` has `1.25x` C4 merge MAC and only `0.18x` route MAC.",
        "- For the current two-stage implementation, temporal-key grouping is unlikely to reach the 3x-vs-Flash speed target unless the merge kernel is redesigned around the large 364-key block.",
        "",
        "## Hard-C4 Fused Tile Constraint",
        "",
        "`analysis/estimate_attention_io.py` also models an optimistic no-intermediate",
        "hard-C4 kernel that computes compressed K/V and immediately consumes it, instead",
        "of writing `merged_k`, `merged_v`, and `bias` to HBM and reading them back.",
        "",
        "The tradeoff is sharp:",
        "",
        "- Small query-column tiles reduce live state, but reload original K/V too many",
        "  times. For example, `I_TILE=8,J_TILE=16` is only about `152` KiB live state,",
        "  but its logical IO is still `1.965x` FlashAttention.",
        "- Large query-column tiles reduce IO, but the live state becomes too large.",
        "  `I_TILE=30,J_TILE=16` is only `0.494x` FlashAttention logical IO, but needs",
        "  about `419` KiB of live tensors, well above a rough H200 one-block shared",
        "  memory budget of `228` KiB.",
        "- With the default candidate list, the only tile that is both below",
        "  FlashAttention logical IO and within the approximate H200 shared-memory budget",
        "  is `I_TILE=30,J_TILE=8`: `0.739x` FlashAttention IO with about `222` KiB live",
        "  state (`97.5%` of the 228 KiB budget).",
        "",
        "The estimator also includes a more conservative two-pass no-merged-KV option:",
        "pass 1 computes only the second-stage softmax row max/sum, then pass 2",
        "recomputes the merge and consumes V. After accounting for D-tiled output",
        "accumulation honestly, this is not a clean win. If pass 2 uses a small",
        "`D_TILE`, it must recompute the full-D route logits for every D chunk, repeating",
        "seed-Q and K traffic. If it uses `D_TILE=128`, the live state again exceeds the",
        "rough H200 shared-memory budget. With the default sweep, no two-pass tile is",
        "both below FlashAttention logical IO and within the 228 KiB live-state budget.",
        "",
        "This is only a feasibility filter, not a runtime prediction. The live-state",
        "estimate mixes tensors that may live in registers, shared memory, or be",
        "recomputed, and it ignores occupancy. But it makes the next kernel target",
        "clear: the only current no-intermediate candidate is one-pass full-query-rows",
        "`J_TILE=8`; avoid many tiny column tiles and avoid D-tiled two-pass unless a",
        "new way is found to reuse route logits without storing another large",
        "intermediate.",
        "",
        "A synthetic H200 Triton probe in",
        "`analysis/probe_hard_c4_fused_tile_feasibility.py` confirms that the padded",
        "accumulator sizes for this direction compile. With `D=128`, logical query tiles",
        "`30/60/120/240` pad to `32/64/128/256` rows and all ran on H200. The largest",
        "case, `q_tile=240`, uses a padded fp32 accumulator of `256 x 128 = 128` KiB.",
        "This does not prove the full one-pass hard-C4 kernel will be fast, but it",
        "removes the first compile/live-state blocker for the `I_TILE=30,J_TILE=8`",
        "target.",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--outer", type=int, default=21)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--cols", type=int, default=52)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype_bytes", type=int, default=2)
    parser.add_argument("--bias_bytes", type=int, default=4)
    parser.add_argument("--flash_block_m", type=int, default=128)
    parser.add_argument(
        "--merge_group_tile",
        type=int,
        default=64,
        help="Number of query-column groups covered by one merge tile.",
    )
    parser.add_argument(
        "--variants",
        default=(
            "c1,c2,c3,c4,"
            "qc1_kr2_c4,qc1_kr2_c6,qc1_kr2_c7,qc1_kr2_c8,"
            "qc1_kr3_c5,qc1_kr3_c6,"
            "qc1_kr5_c6,qc1_kr5_c7,qc1_kr5_c8,qc1_kr5_c9,qc1_kr5_c10,qc1_kr5_c15,"
            "qc2_kr1_c6,qc2_kr1_c7,qc2_kr1_c8,"
            "qc4_kr1_c8,qc2_kr2_c8,qc2_kr3_c12,qc4_kr2_c16,"
            "qf1_kf3_c3,qf1_kf7_c4,qf1_kf7_c5,qf1_kf7_c7,qf1_kf7_c10"
        ),
    )
    parser.add_argument("--output_json")
    parser.add_argument("--output_md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = Shape(
        batch=args.batch,
        outer=args.outer,
        rows=args.rows,
        cols=args.cols,
        heads=args.heads,
        dim=args.dim,
        dtype_bytes=args.dtype_bytes,
        bias_bytes=args.bias_bytes,
        flash_block_m=args.flash_block_m,
        merge_group_tile=args.merge_group_tile,
    )
    flash_row = flash(shape)
    specs = [parse_variant(name) for name in args.variants.replace(",", " ").split()]
    c4_row = estimate_variant(shape, parse_variant("c4"))
    rows = [add_ratios(estimate_variant(shape, spec), flash_row, c4_row) for spec in specs]
    payload = {
        "shape": shape.__dict__,
        "flash": {
            "io_gb": gb(flash_row["io_bytes"]),
            "mac_t": mac_t(flash_row["mac"]),
        },
        "rows": rows,
    }

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        markdown(rows, Path(args.output_md))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
