"""Find oracle per-call C schedules from a recorded C sweep.

This script is a planning tool: it does not change model execution. Given
per-self-attention-call timings/errors for C1/C2/C3/C4, it solves a small
multiple-choice knapsack problem:

    minimize summed per-call MSE subject to an attention-time budget.

The result answers whether a layer/call-specific C choice has enough headroom
to approach a target speedup versus FlashAttention while improving over fixed C1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TARGETS = "3.5,3.0,2.5,2.0,1.5"


def load_records(path):
    payload = json.loads(Path(path).read_text())
    return payload, payload["records"]


def summarize_fixed(records, variants):
    rows = {}
    for name in variants:
        seconds = sum(record["variants"][name]["seconds"] for record in records)
        mse = sum(record["variants"][name].get("mse_vs_dense", 0.0) for record in records) / len(records)
        rel_l2 = sum(record["variants"][name].get("rel_l2_vs_dense", 0.0) for record in records) / len(records)
        rows[name] = {
            "seconds": seconds,
            "mean_ms": 1000.0 * seconds / len(records),
            "mean_mse": mse,
            "mean_rel_l2": rel_l2,
        }
    return rows


def best_schedule(records, variants, budget_seconds, tick_seconds):
    budget_ticks = int(round(budget_seconds / tick_seconds))
    dp = {0: (0.0, [])}
    for call_index, record in enumerate(records):
        next_dp = {}
        for used_ticks, (error, choices) in dp.items():
            for name in variants:
                variant = record["variants"][name]
                ticks = max(1, int(round(variant["seconds"] / tick_seconds)))
                new_ticks = used_ticks + ticks
                if new_ticks > budget_ticks:
                    continue
                new_error = error + float(variant["mse_vs_dense"])
                old = next_dp.get(new_ticks)
                if old is None or new_error < old[0]:
                    next_dp[new_ticks] = (new_error, [*choices, (call_index, name)])
        if not next_dp:
            return None

        # Keep only Pareto-improving states as time increases.
        pruned = {}
        best_error = float("inf")
        for ticks in sorted(next_dp):
            error, choices = next_dp[ticks]
            if error < best_error:
                pruned[ticks] = (error, choices)
                best_error = error
        dp = pruned

    best_ticks, (best_error, choices) = min(dp.items(), key=lambda item: item[1][0])
    schedule = [name for _call_index, name in choices]
    return {
        "seconds": best_ticks * tick_seconds,
        "mean_ms": 1000.0 * best_ticks * tick_seconds / len(records),
        "mean_mse": best_error / len(records),
        "counts": dict(Counter(schedule)),
        "schedule": schedule,
    }


def module_counts(records, schedule):
    counts = defaultdict(Counter)
    for record, choice in zip(records, schedule):
        counts[record["module"]][choice] += 1
    return {module: dict(counter) for module, counter in sorted(counts.items())}


def ref_mean_from_variant(variant):
    mse = variant.get("mse_vs_dense")
    rel_l2 = variant.get("rel_l2_vs_dense")
    if mse is None or rel_l2 in (None, 0.0):
        return None
    return float(mse) / (float(rel_l2) ** 2)


def aggregate_schedule_error(records, schedule):
    mse_sum = 0.0
    ref_mean_sum = 0.0
    max_abs = 0.0
    for record, choice in zip(records, schedule):
        variant = record["variants"][choice]
        mse_sum += float(variant["mse_vs_dense"])
        ref_mean = ref_mean_from_variant(variant)
        if ref_mean is not None:
            ref_mean_sum += ref_mean
        max_abs = max(max_abs, float(variant.get("max_abs_vs_dense", 0.0)))
    mean_mse = mse_sum / len(records)
    return {
        "mean_mse": mean_mse,
        "aggregate_rel_l2": (mse_sum / ref_mean_sum) ** 0.5 if ref_mean_sum else None,
        "max_abs_vs_dense": max_abs,
    }


def choice_rows(records, schedule):
    rows = []
    for record, choice in zip(records, schedule):
        prompt_call = int(record.get("prompt_call_index", record["global_call_index"]))
        rows.append(
            {
                "global_call_index": record["global_call_index"],
                "prompt_call_index": prompt_call,
                "denoise_step_index": (prompt_call - 1) // 30,
                "block_index": (prompt_call - 1) % 30,
                "module": record["module"],
                "choice": choice,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--variants", default="c1,c2,c3,c4")
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--tick_ms", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    payload, records = load_records(args.input_json)
    variants = [name.strip() for name in args.variants.replace(",", " ").split() if name.strip()]
    targets = [float(value) for value in args.targets.replace(",", " ").split() if value.strip()]
    tick_seconds = args.tick_ms / 1000.0

    fixed = summarize_fixed(records, ["flash_attn_triton", *variants])
    flash_seconds = fixed["flash_attn_triton"]["seconds"]
    sdpa_seconds = sum(record["variants"]["sdpa"]["seconds"] for record in records)

    schedules = {}
    for target in targets:
        budget = flash_seconds / target
        result = best_schedule(records, variants, budget, tick_seconds)
        if result is None:
            schedules[f"{target:g}x_vs_flash"] = {"target_speedup_vs_flash": target, "feasible": False}
            continue
        result["feasible"] = True
        result["target_speedup_vs_flash"] = target
        result["actual_speedup_vs_flash"] = flash_seconds / result["seconds"]
        result["actual_speedup_vs_sdpa"] = sdpa_seconds / result["seconds"]
        schedule = result.pop("schedule")
        result.update(aggregate_schedule_error(records, schedule))
        result["module_counts"] = module_counts(records, schedule)
        result["choices"] = choice_rows(records, schedule)
        schedules[f"{target:g}x_vs_flash"] = result

    output = {
        "source": args.input_json,
        "device": payload.get("device"),
        "num_records": len(records),
        "targets": targets,
        "tick_ms": args.tick_ms,
        "fixed": fixed,
        "sdpa": {
            "seconds": sdpa_seconds,
            "mean_ms": 1000.0 * sdpa_seconds / len(records),
        },
        "schedules": schedules,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
