"""Compute a finite-portfolio oracle over strictly Q-only seed selectors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _case_records(payload: dict) -> dict[str, dict]:
    output = {}
    for capture in payload["cases"].values():
        output.update(capture)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases: dict[str, dict[str, dict]] = defaultdict(dict)
    baselines: dict[str, dict] = {}
    for source in args.input:
        payload = json.loads(Path(source).read_text())
        if not payload["settings"].get("selector_reads_key", False) is False:
            raise ValueError(f"{source} is not marked as a key-free selector run")
        for case_name, methods in _case_records(payload).items():
            if "fixed_wide" in methods:
                baselines[case_name] = methods["fixed_wide"]
            for method_name, metrics in methods.items():
                if method_name.startswith("query_"):
                    cases[case_name][method_name] = metrics

    common_cases = sorted(set(cases) & set(baselines))
    if not common_cases:
        raise ValueError("no common cases found")
    winners = Counter()
    totals = defaultdict(float)
    baseline_total = 0.0
    query_total = 0
    per_case = {}
    for case_name in common_cases:
        winner, metrics = min(
            cases[case_name].items(), key=lambda item: item[1]["mean_key_kl"]
        )
        queries = int(metrics["queries"])
        winners[winner] += 1
        query_total += queries
        baseline_total += float(baselines[case_name]["mean_key_kl"]) * queries
        for key in (
            "mean_key_kl",
            "mean_cluster_kl",
            "mean_conditional_kl",
            "active_cluster_fraction",
        ):
            totals[key] += float(metrics[key]) * queries
        per_case[case_name] = {
            "winner": winner,
            "mean_key_kl": metrics["mean_key_kl"],
            "fixed_wide_mean_key_kl": baselines[case_name]["mean_key_kl"],
        }

    baseline = baseline_total / query_total
    aggregate = {key: value / query_total for key, value in totals.items()}
    aggregate.update(
        {
            "cases": len(common_cases),
            "queries": query_total,
            "fixed_wide_mean_key_kl": baseline,
            "mean_key_kl_ratio_vs_fixed": aggregate["mean_key_kl"] / baseline,
            "improvement_vs_fixed_percent": 100.0
            * (1.0 - aggregate["mean_key_kl"] / baseline),
        }
    )
    result = {
        "settings": {
            "inputs": args.input,
            "candidate_selectors_read_key": False,
            "oracle_uses_key_derived_kl_to_choose_winner": True,
            "deployable_q_only": False,
            "scope": "finite portfolio, one winning selector per QKV case",
        },
        "aggregate": aggregate,
        "winner_counts": dict(winners.most_common()),
        "cases": per_case,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({"aggregate": aggregate, "winner_counts": result["winner_counts"]}, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
