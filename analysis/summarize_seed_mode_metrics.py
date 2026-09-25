import argparse
import json
from pathlib import Path


def fmt(value, digits=6):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def main():
    parser = argparse.ArgumentParser(description="Print a compact table from compare_hard_c4_seed_modes.py JSON.")
    parser.add_argument("metrics_json")
    args = parser.parse_args()

    payload = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    rows = []
    for name, stats in payload["variants_vs_dense"].items():
        rows.append(
            (
                stats.get("rel_l2", -1.0) if stats.get("rel_l2") is not None else -1.0,
                name,
                stats,
            )
        )

    print("| mode | rel_l2 | mse | max_abs | mean_ms | speedup_vs_dense_attention | calls |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for _, name, stats in sorted(rows, key=lambda item: (item[0] if item[0] >= 0 else -999)):
        print(
            "| "
            + " | ".join(
                [
                    name,
                    fmt(stats.get("rel_l2")),
                    fmt(stats.get("mse")),
                    fmt(stats.get("max_abs")),
                    fmt(stats.get("mean_ms")),
                    fmt(stats.get("speedup_vs_dense_attention")),
                    fmt(stats.get("calls"), digits=0),
                ]
            )
            + " |"
        )

    print()
    print(
        "meta:",
        json.dumps(
            {
                "device": payload.get("device"),
                "grid_size": payload.get("grid_size"),
                "num_prompts": payload.get("num_prompts"),
                "max_calls_per_prompt": payload.get("max_calls_per_prompt"),
                "wall_seconds": payload.get("wall_seconds"),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
