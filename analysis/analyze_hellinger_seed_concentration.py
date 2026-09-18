"""Measure where production Hellinger-Span C4 selects query rows.

The production ``t1`` configuration computes row indices from one router
frame and reuses those indices for every query frame.  This script separates
the mandatory matched-row anchor from the three adaptive rows, measures how
concentrated those adaptive choices are, and optionally evaluates a post-hoc
shared-C4 control.  The control takes the four most frequently selected rows
per query column/head and then shares them over every key block.

The shared control is an oracle diagnostic, not a deployable Q-only selector:
its rows are inferred from a Hellinger table that was itself computed with K.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
from analysis.evaluate_query_only_seed_kl import (
    C4,
    COLS,
    FRAMES,
    ROWS,
    _global_farthest_order,
    _production_wide_rows,
    evaluate_attention_kl,
)


def _parse_indices(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def _load_qk(path: str, heads: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as payload:
        tensors = tuple(
            torch.from_numpy(payload[name][0, heads].copy()).reshape(
                len(heads), FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k")
        )
    return tensors


def _histogram_summary(histogram: torch.Tensor) -> dict[str, object]:
    histogram = histogram.to(torch.float64)
    total = float(histogram.sum())
    if total <= 0:
        return {
            "total": 0,
            "normalized_entropy": 0.0,
            "effective_support": 0.0,
            "top1_mass": 0.0,
            "top4_mass": 0.0,
            "top8_mass": 0.0,
        }
    probability = histogram / total
    support = probability > 0
    entropy = float(-(probability[support] * probability[support].log()).sum())
    ordered = probability.sort(descending=True).values
    return {
        "total": int(total),
        "normalized_entropy": entropy / math.log(histogram.numel()),
        "effective_support": math.exp(entropy),
        "top1_mass": float(ordered[:1].sum()),
        "top4_mass": float(ordered[:4].sum()),
        "top8_mass": float(ordered[:8].sum()),
    }


def _top_entries(
    histogram: torch.Tensor, count: int, *, index_offset: int = 0
) -> list[dict[str, float | int]]:
    total = max(int(histogram.sum()), 1)
    values, indices = histogram.topk(min(count, histogram.numel()))
    return [
        {
            "index": int(index) + index_offset,
            "count": int(value),
            "mass": int(value) / total,
        }
        for value, index in zip(values, indices)
    ]


def _vote_shared_rows(table: torch.Tensor) -> torch.Tensor:
    """Return top-four adaptive row modes as [column, head, C4]."""

    if table.ndim != 5 or table.size(-1) != C4:
        raise ValueError("table must have shape [key_frame,row,column,head,C4]")
    dynamic = table[..., 1:].to(torch.long)
    histogram = F.one_hot(dynamic, num_classes=ROWS).sum(dim=(0, 1, 4))
    return histogram.topk(C4, dim=-1).indices


def _accumulate_distribution(
    table: torch.Tensor,
    accumulator: dict[str, object],
    block_j: int,
) -> None:
    """Accumulate concentration metrics from one CPU seed table."""

    table = table.to(torch.long)
    representatives = table[:, :, ::block_j]
    dynamic = representatives[..., 1:]
    key_row = torch.arange(ROWS).view(1, ROWS, 1, 1, 1)
    offset = representatives - key_row

    accumulator["row_hist"] += torch.bincount(
        representatives.flatten(), minlength=ROWS
    )
    accumulator["dynamic_row_hist"] += torch.bincount(
        dynamic.flatten(), minlength=ROWS
    )
    accumulator["offset_hist"] += torch.bincount(
        (offset + ROWS - 1).flatten(), minlength=2 * ROWS - 1
    )
    for slot in range(C4):
        accumulator["slot_row_hist"][slot] += torch.bincount(
            representatives[..., slot].flatten(), minlength=ROWS
        )
        accumulator["slot_offset_hist"][slot] += torch.bincount(
            (offset[..., slot] + ROWS - 1).flatten(),
            minlength=2 * ROWS - 1,
        )

    sorted_rows = representatives.sort(dim=-1).values
    unique_count = 1 + (sorted_rows[..., 1:] != sorted_rows[..., :-1]).sum(-1)
    accumulator["unique_hist"] += torch.bincount(
        unique_count.flatten(), minlength=C4 + 1
    )
    span = sorted_rows[..., -1] - sorted_rows[..., 0]
    accumulator["span_hist"] += torch.bincount(span.flatten(), minlength=ROWS)
    pairwise = []
    for left in range(C4):
        for right in range(left + 1, C4):
            pairwise.append(
                (representatives[..., left] - representatives[..., right]).abs()
            )
    pairwise_distance = torch.stack(pairwise, dim=-1)
    accumulator["pairwise_hist"] += torch.bincount(
        pairwise_distance.flatten(), minlength=ROWS
    )

    for radius in (1, 2, 4, 8):
        accumulator["dynamic_local_hits"][radius] += int(
            ((dynamic - key_row).abs() <= radius).sum()
        )
    accumulator["dynamic_count"] += dynamic.numel()

    contexts = dynamic.permute(0, 2, 3, 1, 4).reshape(-1, ROWS * (C4 - 1))
    context_histogram = F.one_hot(contexts, num_classes=ROWS).sum(dim=1)
    context_probability = context_histogram / context_histogram.sum(
        dim=-1, keepdim=True
    ).clamp_min(1)
    context_support = context_probability > 0
    context_entropy = -torch.where(
        context_support,
        context_probability * context_probability.clamp_min(1.0e-30).log(),
        torch.zeros_like(context_probability),
    ).sum(-1)
    ordered = context_probability.sort(dim=-1, descending=True).values
    accumulator["context_unique"].extend(
        (context_histogram > 0).sum(-1).tolist()
    )
    accumulator["context_effective"].extend(context_entropy.exp().tolist())
    accumulator["context_top4"].extend(ordered[:, :4].sum(-1).tolist())
    accumulator["context_top8"].extend(ordered[:, :8].sum(-1).tolist())

    group_matches = 0
    group_total = 0
    group_leading_total = 0
    for column in range(COLS):
        group_start = (column // block_j) * block_j
        group_matches += int(
            (table[:, :, column] == table[:, :, group_start]).all(dim=-1).sum()
        )
        group_total += table.size(0) * table.size(1) * table.size(3)
        if column == group_start:
            group_leading_total += table.size(0) * table.size(1) * table.size(3)
    accumulator["group_matches"] += group_matches
    accumulator["group_total"] += group_total
    accumulator["group_leading_total"] += group_leading_total


def _new_accumulator() -> dict[str, object]:
    return {
        "row_hist": torch.zeros(ROWS, dtype=torch.long),
        "dynamic_row_hist": torch.zeros(ROWS, dtype=torch.long),
        "offset_hist": torch.zeros(2 * ROWS - 1, dtype=torch.long),
        "slot_row_hist": torch.zeros(C4, ROWS, dtype=torch.long),
        "slot_offset_hist": torch.zeros(C4, 2 * ROWS - 1, dtype=torch.long),
        "unique_hist": torch.zeros(C4 + 1, dtype=torch.long),
        "span_hist": torch.zeros(ROWS, dtype=torch.long),
        "pairwise_hist": torch.zeros(ROWS, dtype=torch.long),
        "dynamic_local_hits": {1: 0, 2: 0, 4: 0, 8: 0},
        "dynamic_count": 0,
        "context_unique": [],
        "context_effective": [],
        "context_top4": [],
        "context_top8": [],
        "group_matches": 0,
        "group_total": 0,
        "group_leading_total": 0,
    }


def _weighted_quantiles(histogram: torch.Tensor) -> dict[str, float]:
    total = int(histogram.sum())
    if total == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
    values = torch.arange(histogram.numel(), dtype=torch.float64)
    cumulative = histogram.cumsum(0)

    def percentile(q: float) -> float:
        target = math.ceil(q * total)
        return float(torch.searchsorted(cumulative, torch.tensor(target)).item())

    return {
        "mean": float((values * histogram).sum() / total),
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
    }


def _list_summary(values: list[float | int]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.1)),
        "p90": float(torch.quantile(tensor, 0.9)),
    }


def _finalize_distribution(accumulator: dict[str, object]) -> dict[str, object]:
    dynamic_count = max(int(accumulator["dynamic_count"]), 1)
    group_total = max(int(accumulator["group_total"]), 1)
    leading_columns = int(accumulator["group_leading_total"])
    non_leading_total = max(group_total - leading_columns, 1)
    non_leading_matches = max(
        int(accumulator["group_matches"]) - leading_columns, 0
    )
    fixed = _production_wide_rows(torch.device("cpu"))
    fixed_sorted = fixed.sort(dim=-1).values
    fixed_pairwise = torch.stack(
        [
            (fixed[:, left] - fixed[:, right]).abs()
            for left in range(C4)
            for right in range(left + 1, C4)
        ],
        dim=-1,
    )
    return {
        "all_seed_rows": {
            **_histogram_summary(accumulator["row_hist"]),
            "top_rows": _top_entries(accumulator["row_hist"], 10),
            "histogram": accumulator["row_hist"].tolist(),
        },
        "adaptive_seed_rows_only": {
            **_histogram_summary(accumulator["dynamic_row_hist"]),
            "top_rows": _top_entries(accumulator["dynamic_row_hist"], 10),
            "histogram": accumulator["dynamic_row_hist"].tolist(),
        },
        "relative_offsets": {
            "top_offsets": _top_entries(
                accumulator["offset_hist"], 12, index_offset=-(ROWS - 1)
            ),
            "histogram_minus44_to_plus44": accumulator["offset_hist"].tolist(),
            "adaptive_local_fraction": {
                str(radius): accumulator["dynamic_local_hits"][radius]
                / dynamic_count
                for radius in (1, 2, 4, 8)
            },
        },
        "per_slot_row_histogram": accumulator["slot_row_hist"].tolist(),
        "per_slot_offset_histogram": accumulator["slot_offset_hist"].tolist(),
        "seed_set": {
            "unique_row_count_histogram": accumulator["unique_hist"].tolist(),
            "duplicate_seed_fraction": 1.0
            - int(accumulator["unique_hist"][C4])
            / max(int(accumulator["unique_hist"].sum()), 1),
            "row_span": _weighted_quantiles(accumulator["span_hist"]),
            "pairwise_absolute_row_distance": _weighted_quantiles(
                accumulator["pairwise_hist"]
            ),
        },
        "within_column_group_context": {
            "definition": (
                "one key-frame/query-column-group/head context; adaptive rows "
                "collected over all 45 target key rows"
            ),
            "unique_rows": _list_summary(accumulator["context_unique"]),
            "effective_rows": _list_summary(accumulator["context_effective"]),
            "top4_mass": _list_summary(accumulator["context_top4"]),
            "top8_mass": _list_summary(accumulator["context_top8"]),
        },
        "column_group_sharing": {
            "all_columns_exact_match_fraction": accumulator["group_matches"]
            / group_total,
            "non_leading_columns_exact_match_fraction": non_leading_matches
            / non_leading_total,
            "note": (
                "the leading column always matches itself; Hellinger shares "
                "probe positions over four columns, not necessarily seed rows"
            ),
        },
        "fixed_wide_reference": {
            "mean_row_span": float(
                (fixed_sorted[:, -1] - fixed_sorted[:, 0]).float().mean()
            ),
            "mean_pairwise_absolute_row_distance": float(
                fixed_pairwise.float().mean()
            ),
        },
    }


def _add_metrics(
    totals: dict[str, dict[str, float]], name: str, metrics: dict[str, float]
) -> None:
    entry = totals[name]
    queries = int(metrics["queries"])
    entry["queries"] += queries
    entry["cases"] += 1
    for key in (
        "mean_key_kl",
        "mean_cluster_kl",
        "mean_conditional_kl",
        "active_cluster_fraction",
    ):
        entry[key] += float(metrics[key]) * queries


def _finalize_metrics(
    totals: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    output = {}
    for name, values in totals.items():
        queries = max(int(values["queries"]), 1)
        output[name] = {
            key: values[key] / queries
            for key in (
                "mean_key_kl",
                "mean_cluster_kl",
                "mean_conditional_kl",
                "active_cluster_fraction",
            )
        }
        output[name]["cases"] = int(values["cases"])
        output[name]["queries"] = int(values["queries"])
    baseline = output.get("fixed_wide", {}).get("mean_key_kl")
    if baseline:
        for values in output.values():
            values["mean_key_kl_ratio_vs_fixed"] = values["mean_key_kl"] / baseline
    return output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--eval_frames", default="0,10,20")
    parser.add_argument("--router_frame", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--candidate_count", type=int, default=22)
    parser.add_argument("--top_probes", type=int, default=8)
    parser.add_argument("--uniform_probes", type=int, default=8)
    parser.add_argument("--probe_anchor_count", type=int, default=2)
    parser.add_argument("--block_j", type=int, default=4)
    parser.add_argument("--evaluate_kl", action="store_true")
    parser.add_argument("--column_chunk", type=int, default=4)
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    eval_frames = _parse_indices(args.eval_frames)
    if not 0 <= args.router_frame < FRAMES:
        raise ValueError(f"router_frame must be in [0, {FRAMES})")
    accumulator = _new_accumulator()
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    records: dict[str, dict[str, dict[str, float]]] = {}

    for capture_index, capture in enumerate(args.qkv, start=1):
        q_grid, k_grid = _load_qk(capture, heads)
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        router_query = (
            q_grid[:, args.router_frame]
            .permute(1, 2, 0, 3)
            .unsqueeze(0)
            .unsqueeze(0)
            .contiguous()
        )
        production_key = (
            k_grid.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        )
        table = hellinger_span_seed_rows_triton(
            router_query,
            production_key,
            alpha=args.alpha,
            candidate_count=args.candidate_count,
            top_probes=args.top_probes,
            uniform_probes=args.uniform_probes,
            probe_anchor_count=args.probe_anchor_count,
            block_j=args.block_j,
            output_dtype=torch.uint8,
            gs_num_warps=8,
            group_tile=32,
            selection_mode="gs",
        )[0, 0]
        table_cpu = table.cpu()
        _accumulate_distribution(table_cpu, accumulator, args.block_j)

        if args.evaluate_kl:
            vote_rows = _vote_shared_rows(table).to(torch.long)
            block_key_row = torch.arange(
                FRAMES * ROWS, device=q_grid.device
            ) % ROWS
            fixed = _production_wide_rows(q_grid.device)[block_key_row]
            fixed = fixed.unsqueeze(0).expand(COLS, -1, -1)
            capture_records: dict[str, dict[str, float]] = {}
            for head_slot, head in enumerate(heads):
                key = k_grid[head_slot].reshape(FRAMES * ROWS, COLS, -1)
                hellinger_rows = (
                    table[..., head_slot, :]
                    .reshape(FRAMES * ROWS, COLS, C4)
                    .permute(1, 0, 2)
                    .to(torch.long)
                )
                shared_vote = vote_rows[:, head_slot]
                shared_vote = shared_vote.unsqueeze(1).expand(
                    -1, FRAMES * ROWS, -1
                )
                router_points = q_grid[
                    head_slot, args.router_frame
                ].permute(1, 0, 2)
                query_global = _global_farthest_order(
                    router_points[..., :32], C4
                )
                query_global = query_global.unsqueeze(1).expand(
                    -1, FRAMES * ROWS, -1
                )
                for frame in eval_frames:
                    query = q_grid[head_slot, frame].permute(1, 0, 2).contiguous()
                    case_name = f"{Path(capture).stem}:h{head}:f{frame}"
                    case = {}
                    for name, rows in (
                        ("fixed_wide", fixed),
                        ("hellinger_key_dependent", hellinger_rows),
                        ("hellinger_vote_shared_oracle", shared_vote),
                        ("query_global_c4_d32", query_global),
                    ):
                        metrics = evaluate_attention_kl(
                            query, key, rows, args.column_chunk
                        )
                        case[name] = metrics
                        _add_metrics(totals, name, metrics)
                    capture_records[case_name] = case
                    print(
                        f"{case_name} fixed={case['fixed_wide']['mean_key_kl']:.6f} "
                        f"hellinger={case['hellinger_key_dependent']['mean_key_kl']:.6f} "
                        f"vote={case['hellinger_vote_shared_oracle']['mean_key_kl']:.6f} "
                        f"qglobal={case['query_global_c4_d32']['mean_key_kl']:.6f}",
                        flush=True,
                    )
            records[Path(capture).stem] = capture_records

        print(
            f"processed {capture_index}/{len(args.qkv)} {Path(capture).name}",
            flush=True,
        )

    result = {
        "settings": {
            **vars(args),
            "heads": heads,
            "eval_frames": eval_frames,
            "device_name": torch.cuda.get_device_name(args.device),
            "production_top1_semantics": (
                "router-frame row indices are reused for every query frame"
            ),
        },
        "interpretation": {
            "hellinger_key_dependent": True,
            "hellinger_vote_shared_oracle_key_block_independent": True,
            "hellinger_vote_shared_oracle_uses_key_offline": True,
            "hellinger_vote_shared_oracle_deployable_q_only": False,
        },
        "concentration": _finalize_distribution(accumulator),
        "aggregate_kl": _finalize_metrics(totals) if totals else {},
        "cases": records,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["concentration"], indent=2))
    if totals:
        print(json.dumps(result["aggregate_kl"], indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
