"""Summarize explored GridMergeAttention block-layout families.

This script consumes existing experiment JSON files. It does not load models or
touch GPUs. Use it to keep the layout search honest: before adding another
kernel, check whether the corresponding block family already failed on real
Wan Q/K/V.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


TUNED_NAMES = {
    "monarch_rt_triton",
    "current_hard_c4",
    "hard_c2",
    "hard_c3",
    "hard_c4",
    "hard_c4_triton",
    "hard_c2_rt_triton",
    "hard_c4_rt_triton",
    "fused_triton",
    "hard_kr2_c4",
    "hard_kr2_c6",
    "hard_kr3_c5",
    "hard_qc2_c6",
    "hard_qc2_kr2_c8",
    "hard_qc4_c8",
}

REFERENCE_NAME_RE = re.compile(r"(?:^|_)flash(?:_|$)|dense|sdpa|grid_ref")


def first_value(item, *names):
    for name in names:
        value = item.get(name)
        if value is not None:
            return value
    return None


def load_json(path):
    try:
        if path.stat().st_size == 0:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def family_for(name):
    if re.search(r"onepass|chunk\d+|pytorch", name):
        return "kernel_prototype"
    if name in {
        "monarch_rt_triton",
        "current_hard_c4",
        "hard_c2",
        "hard_c3",
        "hard_c4",
        "hard_c4_triton",
        "hard_c2_rt_triton",
        "hard_c4_rt_triton",
        "fused_triton",
    }:
        return "original_row_query_col_keyrow"
    if re.match(r"hard_kr\d+_c\d+", name):
        return "wider_key_rows"
    if re.match(r"hard_qc\d+_kr\d+_c\d+", name):
        return "query_columns_and_key_rows"
    if re.match(r"hard_qc\d+_c\d+", name):
        return "query_column_group"
    if "transpose" in name:
        return "transpose_rowcol"
    if re.match(r"qf\d+_kf\d+_c\d+", name):
        return "temporal_frame_group"
    if re.match(r"hard_kf\d+_c\d+", name):
        return "temporal_frame_group"
    if re.match(r"ta_", name):
        return "temporal_adaptive_c"
    if re.match(r"monarchrt_h\d+", name):
        return "original_row_query_col_keyrow"
    if re.match(r"q[thw]_k[thw]_c\d+", name):
        q_axis = name[1]
        k_axis = name[4]
        if q_axis == "h" and k_axis == "w":
            return "original_row_query_col_keyrow"
        if q_axis == "w" and k_axis == "h":
            return "transpose_rowcol"
        return "same_axis" if q_axis == k_axis else "mixed_axis"
    if re.match(r"qp\d+x\d+_kp\d+x\d+_c\d+", name):
        return "query_key_patch"
    if re.match(r"kp\d+x\d+_c\d+", name):
        return "key_patch"
    if re.match(r"kg\d+_c\d+", name):
        return "key_column_group"
    match = re.match(r"qc(?P<qc>\d+)_kr(?P<kr>\d+)_c(?P<c>\d+)(?:_p[a-z0-9]+)?(?:_g(?P<group>[a-z0-9]+))?", name)
    if match:
        qc = int(match.group("qc"))
        kr = int(match.group("kr"))
        group = match.group("group") or ""
        if group == "interleave":
            return "interleaved_key_rows"
        if qc > 1 and kr > 1:
            return "query_columns_and_key_rows"
        if qc > 1:
            return "query_column_group"
        if kr > 1:
            return "wider_key_rows"
        return "original_row_query_col_keyrow"
    lowered = name.lower()
    if re.match(r"c\d+_local", lowered) or any(token in lowered for token in ("seed", "greedy", "prefilter", "top_", "top")):
        return "seed_selection"
    return "other"


def device_matches(payload, path, device_filter):
    if not device_filter:
        return True
    wanted = device_filter.lower()
    path_text = str(path).lower()
    if wanted == "h200" and any(token in path_text for token in ("mi210", "mi300", "mi250")):
        return False
    device = payload.get("device")
    if device is None:
        return True
    return wanted in str(device).lower()


def inferred_layout(name):
    """Infer Wan-480p block metadata for common kernel names.

    Older benchmark JSON files often predate `layout_metadata`. These fallback
    rules keep the coverage report readable without changing the source data.
    """

    if name == "monarch_rt_triton":
        return {"density": 1 / 52, "query_block_tokens": 30, "key_block_tokens": 52}
    if name in {"current_hard_c4", "hard_c4", "hard_c4_triton", "hard_c4_rt_triton", "fused_triton"}:
        return {"density": 4 / 52, "query_block_tokens": 30, "key_block_tokens": 52}
    if name == "hard_c2_rt_triton":
        return {"density": 2 / 52, "query_block_tokens": 30, "key_block_tokens": 52}
    match = re.fullmatch(r"hard_c(?P<c>\d+)", name)
    if match:
        c = int(match.group("c"))
        return {"density": c / 52, "query_block_tokens": 30, "key_block_tokens": 52}
    match = re.fullmatch(r"hard_kr(?P<kr>\d+)_c(?P<c>\d+)", name)
    if match:
        kr = int(match.group("kr"))
        c = int(match.group("c"))
        return {"density": c / (kr * 52), "query_block_tokens": 30, "key_block_tokens": kr * 52}
    match = re.fullmatch(r"hard_qc(?P<qc>\d+)_c(?P<c>\d+)", name)
    if match:
        qc = int(match.group("qc"))
        c = int(match.group("c"))
        return {"density": c / 52, "query_block_tokens": qc * 30, "key_block_tokens": 52}
    match = re.fullmatch(r"hard_qc(?P<qc>\d+)_kr(?P<kr>\d+)_c(?P<c>\d+)", name)
    if match:
        qc = int(match.group("qc"))
        kr = int(match.group("kr"))
        c = int(match.group("c"))
        return {"density": c / (kr * 52), "query_block_tokens": qc * 30, "key_block_tokens": kr * 52}
    match = re.fullmatch(r"transpose_c(?P<c>\d+).*", name)
    if match:
        c = int(match.group("c"))
        return {"density": c / 30, "query_block_tokens": 52, "key_block_tokens": 30}
    return {}


def collect_rows(root, device_filter=None):
    rows = []
    skipped = []
    for path in sorted(Path(root).rglob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            skipped.append(str(path))
            continue
        if not device_matches(payload, path, device_filter):
            continue
        variants = payload.get("variants_vs_dense") or payload.get("variants")
        if not isinstance(variants, dict):
            continue
        metadata = payload.get("layout_metadata") or payload.get("variant_metadata") or {}
        payload_calls = payload.get("calls_recorded") or payload.get("calls_total")
        flash = variants.get("flash_attn_triton") or {}
        flash_ms = first_value(flash, "mean_ms_per_call", "mean_ms")
        for name, stats in variants.items():
            if name in {"dense", "sdpa", "flash_attn_triton"} or REFERENCE_NAME_RE.search(name) or not isinstance(stats, dict):
                continue
            rel_l2 = first_value(stats, "rel_l2_vs_dense", "rel_l2", "relative_l2")
            mse = first_value(stats, "mse_vs_dense", "mse")
            mean_ms = first_value(stats, "mean_ms_per_call", "mean_ms")
            if rel_l2 is None:
                continue
            meta = metadata.get(name, {}) if isinstance(metadata, dict) else {}
            inferred = inferred_layout(name)
            calls = first_value(stats, "calls", "count") or payload_calls or 0
            density = (
                first_value(stats, "compressed_token_density", "density")
                or meta.get("compressed_token_density")
                or inferred.get("density")
            )
            query_block = meta.get("query_block_tokens") or meta.get("query_inner_tokens") or inferred.get("query_block_tokens")
            key_block = meta.get("key_block_tokens") or meta.get("key_inner_tokens") or inferred.get("key_block_tokens")
            row = {
                "name": name,
                "family": family_for(name),
                "experiment": path.parent.name if path.name == "metrics.json" else path.stem,
                "path": str(path),
                "calls": int(calls or 0),
                "mean_ms": float(mean_ms) if mean_ms is not None else None,
                "speedup_vs_flash": float(flash_ms) / float(mean_ms) if flash_ms and mean_ms else None,
                "rel_l2": float(rel_l2),
                "mse": float(mse) if mse is not None else None,
                "density": float(density) if density is not None else None,
                "query_block_tokens": int(query_block) if query_block is not None else None,
                "key_block_tokens": int(key_block) if key_block is not None else None,
                "kind": "tuned_kernel" if name in TUNED_NAMES else "analysis_only",
            }
            rows.append(row)
    return rows, skipped


def best_rows(rows, *, min_calls=1, max_density=None, kind=None, limit=20):
    candidates = [row for row in rows if row["calls"] >= min_calls]
    if max_density is not None:
        candidates = [row for row in candidates if row["density"] is not None and row["density"] <= max_density]
    if kind is not None:
        candidates = [row for row in candidates if row["kind"] == kind]
    return sorted(candidates, key=lambda row: (row["rel_l2"], row["density"] or math.inf))[:limit]


def best_variant_rows(rows, *, min_calls=1, kind=None, limit=20):
    grouped = defaultdict(list)
    for row in rows:
        if row["calls"] < min_calls:
            continue
        if kind is not None and row["kind"] != kind:
            continue
        grouped[row["name"]].append(row)

    selected = []
    for items in grouped.values():
        if kind == "tuned_kernel":
            selected.append(
                min(
                    items,
                    key=lambda row: (
                        -row["calls"],
                        row["mean_ms"] if row["mean_ms"] is not None else math.inf,
                        row["rel_l2"],
                    ),
                )
            )
        else:
            selected.append(min(items, key=lambda row: (-row["calls"], row["rel_l2"])))
    return sorted(selected, key=lambda row: (row["rel_l2"], row["mean_ms"] or math.inf))[:limit]


def best_by_family(rows, min_calls):
    grouped = defaultdict(list)
    for row in rows:
        if row["calls"] >= min_calls:
            grouped[row["family"]].append(row)
    out = []
    for family, items in grouped.items():
        best_quality = min(items, key=lambda row: row["rel_l2"])
        low_density = [row for row in items if row["density"] is not None and row["density"] <= 0.08]
        best_low_density = min(low_density, key=lambda row: row["rel_l2"]) if low_density else None
        out.append((family, best_quality, best_low_density, len(items)))
    return sorted(out, key=lambda item: item[0])


def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def row_line(row, include_family=False, include_speed=False):
    block = "-"
    if row["query_block_tokens"] and row["key_block_tokens"]:
        block = f"{row['query_block_tokens']} x {row['key_block_tokens']}"
    parts = [
        row["name"],
        row["experiment"],
        str(row["calls"]),
        fmt(row["density"], 5),
        block,
        fmt(row["rel_l2"], 5),
        fmt(row["mse"], 6),
    ]
    if include_family:
        parts.insert(1, row["family"])
    if include_speed:
        parts.extend([fmt(row["mean_ms"], 3), fmt(row["speedup_vs_flash"], 2)])
    return "| " + " | ".join(parts) + " |"


def rows_table(rows, include_family=False, include_speed=False):
    header = ["Variant"]
    if include_family:
        header.append("Family")
    header.extend(["Experiment", "Calls", "Density", "Block q x k", "Rel-L2", "MSE"])
    if include_speed:
        header.extend(["Mean ms", "Speedup vs Flash"])
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend(row_line(row, include_family, include_speed) for row in rows)
    return "\n".join(lines)


def family_table(items):
    lines = [
        "| Family | Rows | Best Quality | Rel-L2 | Density | Best <=0.08 Density | Rel-L2 | Density |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for family, best, low, count in items:
        low_name = low["name"] if low else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    str(count),
                    best["name"],
                    fmt(best["rel_l2"], 5),
                    fmt(best["density"], 5),
                    low_name,
                    fmt(low["rel_l2"], 5) if low else "-",
                    fmt(low["density"], 5) if low else "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows, skipped, output_md, output_json, min_calls):
    tuned_min_calls = max(90, min_calls)
    tuned = best_variant_rows(rows, min_calls=tuned_min_calls, kind="tuned_kernel", limit=20)
    low_density = best_rows(rows, min_calls=min_calls, max_density=0.04, limit=20)
    c4_density = best_rows(rows, min_calls=min_calls, max_density=0.08, limit=25)
    families = best_by_family(rows, min_calls)

    text = [
        "# H200 Layout Coverage",
        "",
        "Generated from local experiment JSON files. Tuned rows can be used for speed; analysis-only rows are quality screens.",
        "",
        f"- Parsed rows: {len(rows)}",
        f"- Skipped non-JSON/empty files: {len(skipped)}",
        f"- Minimum calls in tables: {min_calls}",
        "",
        "## Best Tuned Rows",
        "",
        f"Tuned speed rows require at least {tuned_min_calls} recorded calls.",
        "",
        rows_table(tuned, include_family=True, include_speed=True),
        "",
        "## Best Low-Density Rows",
        "",
        "These are at or below C2 compressed-token density (`<= 0.04`).",
        "",
        rows_table(low_density, include_family=True, include_speed=False),
        "",
        "## Best C4-Density-Or-Lower Rows",
        "",
        "These are at or below hard-C4 compressed-token density (`<= 0.08`).",
        "",
        rows_table(c4_density, include_family=True, include_speed=False),
        "",
        "## Best By Family",
        "",
        family_table(families),
        "",
        "## Current Search Implication",
        "",
        "- The original full query-column plus full key-row geometry remains the only tuned speed/quality frontier.",
        "- Multi-row key blocks and temporal-key blocks give strong quality signals, but direct two-stage kernels are merge-bound.",
        "- Query grouping, transpose, same-axis, key-column grouping, and 2D patch layouts have not recovered enough quality at matched density.",
        "- The next useful speed work is kernel-level IO reduction for a quality-preserving layout, not another fixed geometric split with weaker alignment.",
        "",
    ]
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(text), encoding="utf-8")
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps({"rows": rows, "skipped": skipped}, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_root", required=True)
    parser.add_argument("--min_calls", type=int, default=6)
    parser.add_argument("--device_contains", default="H200")
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows, skipped = collect_rows(args.metrics_root, device_filter=args.device_contains)
    write_report(rows, skipped, args.output_md, args.output_json, args.min_calls)
    print(f"wrote {args.output_md} with {len(rows)} rows; skipped {len(skipped)} files")


if __name__ == "__main__":
    main()
