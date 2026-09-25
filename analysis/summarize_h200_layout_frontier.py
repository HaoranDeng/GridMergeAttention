"""Summarize H200 GridMerge layout experiments into a Pareto-style report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TUNED_NAMES = {
    "monarch_rt_triton",
    "hard_c2",
    "hard_c3",
    "hard_c4",
    "hard_c4_triton",
    "hard_kr2_c4",
    "hard_kr2_c6",
    "hard_kr3_c5",
    "hard_qc2_c6",
    "hard_qc2_kr2_c8",
    "hard_qc4_c8",
}


def value(item, *names):
    for name in names:
        if name in item and item[name] is not None:
            return item[name]
    return None


def read_metric_files(root):
    for path in sorted(Path(root).glob("*/metrics.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        variants = payload.get("variants_vs_dense") or {}
        if variants:
            yield path, payload, variants


def collect_rows(root, min_calls):
    rows = []
    for path, payload, variants in read_metric_files(root):
        calls = payload.get("calls_recorded") or payload.get("calls_total") or 0
        if calls < min_calls:
            continue
        metadata = payload.get("layout_metadata") or {}
        flash_ms = None
        if "flash_attn_triton" in variants:
            flash_ms = value(variants["flash_attn_triton"], "mean_ms_per_call", "mean_ms")
        for name, stats in variants.items():
            if name in {"dense", "sdpa", "flash_attn_triton"}:
                continue
            mean_ms = value(stats, "mean_ms_per_call", "mean_ms")
            rel_l2 = value(stats, "rel_l2_vs_dense", "rel_l2")
            mse = value(stats, "mse_vs_dense", "mse")
            if mean_ms is None or rel_l2 is None:
                continue
            meta = metadata.get(name, {})
            rows.append(
                {
                    "experiment": path.parent.name,
                    "calls": calls,
                    "name": name,
                    "mean_ms": mean_ms,
                    "speedup_vs_flash": flash_ms / mean_ms if flash_ms else None,
                    "rel_l2": rel_l2,
                    "mse": mse,
                    "density": meta.get("compressed_token_density"),
                    "query_block_tokens": meta.get("query_block_tokens") or meta.get("query_inner_tokens"),
                    "key_block_tokens": meta.get("key_block_tokens") or meta.get("key_inner_tokens"),
                    "kind": "tuned_kernel" if name in TUNED_NAMES else "analysis_only",
                }
            )
    return rows


def best_by_name(rows, kind=None):
    selected = {}
    for row in rows:
        if kind and row["kind"] != kind:
            continue
        old = selected.get(row["name"])
        if old is None or (row["calls"], -row["rel_l2"]) > (old["calls"], -old["rel_l2"]):
            selected[row["name"]] = row
    return list(selected.values())


def pareto(rows):
    out = []
    for row in sorted(rows, key=lambda item: (item["mean_ms"], item["rel_l2"])):
        if not any(other["mean_ms"] <= row["mean_ms"] and other["rel_l2"] <= row["rel_l2"] for other in out):
            out.append(row)
    return out


def dedupe(rows):
    seen = set()
    unique = []
    for row in rows:
        key = (
            row["experiment"],
            row["name"],
            round(float(row["mean_ms"]), 6),
            round(float(row["rel_l2"]), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def fmt(value_, digits=3):
    if value_ is None:
        return "-"
    return f"{value_:.{digits}f}"


def markdown_table(rows, include_speed=True):
    header = "| Variant | Experiment | Calls | Mean ms |"
    if include_speed:
        header += " Speedup vs Flash |"
    header += " Rel-L2 | MSE | Density | Block q x k |\n"
    sep = "| --- | --- | ---: | ---: |"
    if include_speed:
        sep += " ---: |"
    sep += " ---: | ---: | ---: | ---: |\n"
    body = []
    for row in rows:
        block = "-"
        if row.get("query_block_tokens") and row.get("key_block_tokens"):
            block = f"{int(row['query_block_tokens'])} x {int(row['key_block_tokens'])}"
        line = (
            f"| {row['name']} | {row['experiment']} | {row['calls']} | "
            f"{fmt(row['mean_ms'])} |"
        )
        if include_speed:
            line += f" {fmt(row['speedup_vs_flash'], 2)} |"
        line += (
            f" {fmt(row['rel_l2'], 5)} | {fmt(row['mse'], 6)} | "
            f"{fmt(row['density'], 5)} | {block} |"
        )
        body.append(line)
    return header + sep + "\n".join(body) + "\n"


def write_report(rows, output_md, output_json):
    tuned = [row for row in rows if row["kind"] == "tuned_kernel"]
    analysis = best_by_name(rows, "analysis_only")
    tuned_frontier = pareto(dedupe(tuned))
    quality_candidates = sorted(analysis, key=lambda item: item["rel_l2"])[:12]

    report = [
        "# H200 GridMerge Layout Frontier",
        "",
        "This report summarizes existing scai7/H200 `metrics.json` files with at least the requested call count.",
        "Tuned-kernel rows are useful for speed comparison. Analysis-only rows are PyTorch screening paths; use them for quality signals only.",
        "",
        "## Tuned Kernel Frontier",
        "",
        markdown_table(tuned_frontier, include_speed=True),
        "## Best Analysis-Only Quality Candidates",
        "",
        markdown_table(quality_candidates, include_speed=False),
        "## Takeaway",
        "",
        "- The only measured setting around `3x` versus FlashAttention is C1/MonarchRT, but its rel-L2 is around `0.17`.",
        "- C2 is around `2x` versus FlashAttention with rel-L2 around `0.118`.",
        "- C3/C4 improve error but fall to roughly `1.3x` or below versus FlashAttention.",
        "- Some analysis-only row/key-group layouts, especially `qc1_kr2_c8`, have much better dense error, but existing kr2 Triton evidence says the larger merge working set is not a speed path.",
        "",
    ]
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(report), encoding="utf-8")
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_root", required=True)
    parser.add_argument("--min_calls", type=int, default=90)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.metrics_root, args.min_calls)
    write_report(rows, args.output_md, args.output_json)
    print(f"wrote {args.output_md} with {len(rows)} rows")


if __name__ == "__main__":
    main()
