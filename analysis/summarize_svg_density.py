"""Summarize realized SVG-2 or SVG-EAR attention-map density logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_density(path: Path) -> list[float]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        density = float(record["avg_density"])
        if not math.isfinite(density) or not 0.0 <= density <= 1.0:
            raise ValueError(f"Invalid density at {path}:{line_number}: {density}")
        values.append(density)
    return values


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--calls_per_video", type=int, default=160)
    args = parser.parse_args()

    densities = read_density(args.input)
    calls_per_video = args.calls_per_video
    complete_videos = len(densities) // calls_per_video
    per_video = []
    for index in range(complete_videos):
        start = index * calls_per_video
        video_density = mean(densities[start : start + calls_per_video])
        per_video.append(
            {
                "video_index": index,
                "calls": calls_per_video,
                "density": video_density,
                "sparsity": 1.0 - video_density,
            }
        )

    density = mean(densities)
    payload = {
        "backend": args.backend,
        "definition": (
            "retained attention-map area / dense attention-map area, weighted "
            "by query-cluster and key-cluster sizes"
        ),
        "aggregation": "call-weighted mean of per-call, per-head area-weighted density",
        "calls": len(densities),
        "calls_per_video": calls_per_video,
        "complete_videos": complete_videos,
        "incomplete_video_calls": len(densities) % calls_per_video,
        "density": density,
        "sparsity": 1.0 - density if density is not None else None,
        "min_call_density": min(densities) if densities else None,
        "max_call_density": max(densities) if densities else None,
        "per_video": per_video,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
