"""Merge two prompt shards per backend into one eight-video experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


METHODS = ("dense", "svg", "svg2", "gma")


def summarize(videos):
    ordered = sorted(videos, key=lambda item: item["index"])
    steady = ordered[1:]
    total_attention_ms = sum(item["attention_total_ms"] for item in ordered)
    total_calls = sum(item["attention_calls"] for item in ordered)
    steady_attention_ms = sum(item["attention_total_ms"] for item in steady)
    steady_calls = sum(item["attention_calls"] for item in steady)
    return {
        "mp4_count": len(ordered),
        "attention_calls": total_calls,
        "mean_attention_ms": total_attention_ms / total_calls,
        "mean_inference_seconds": float(
            np.mean([item["inference_seconds"] for item in ordered])
        ),
        "steady_state_videos": len(steady),
        "steady_state_mean_attention_ms": steady_attention_ms / steady_calls,
        "steady_state_mean_inference_seconds": float(
            np.mean([item["inference_seconds"] for item in steady])
        ),
        "mean_export_seconds": float(
            np.mean([item["export_seconds"] for item in ordered])
        ),
        "max_memory_gib": max(item["max_memory_gib"] for item in ordered),
    }


def merge_error(errors):
    calls = sum(error["calls"] for error in errors)
    numel = sum(error["numel"] for error in errors)
    squared_error = sum(error["squared_error_sum"] for error in errors)
    reference_squared = sum(error["reference_squared_sum"] for error in errors)
    return {
        "calls": calls,
        "numel": numel,
        "squared_error_sum": squared_error,
        "reference_squared_sum": reference_squared,
        "mse_vs_dense": squared_error / numel if numel else 0.0,
        "rel_l2_vs_dense": (
            math.sqrt(squared_error / max(reference_squared, 1e-20))
            if calls
            else 0.0
        ),
        "max_abs_vs_dense": max(error["max_abs_vs_dense"] for error in errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--starts", type=int, nargs="+", default=(0, 4))
    args = parser.parse_args()

    root = Path(args.experiment_root)
    for method in METHODS:
        shards = [
            json.loads(
                (root / "chunks" / method / f"start_{start}" / "runtime.json").read_text(
                    encoding="utf-8"
                )
            )
            for start in args.starts
        ]
        videos = sorted(
            [video for shard in shards for video in shard["videos"]],
            key=lambda item: item["index"],
        )
        if [video["index"] for video in videos] != list(range(8)):
            raise ValueError(f"{method} does not contain exactly prompt indices 0-7")

        output_dir = root / method
        output_dir.mkdir(parents=True, exist_ok=True)
        for start in args.starts:
            chunk_dir = root / "chunks" / method / f"start_{start}"
            for source in chunk_dir.glob("*.mp4"):
                destination = output_dir / source.name
                if not destination.exists():
                    os.link(source, destination)

        payload = {
            "protocol": shards[0]["protocol"]
            | {"merged_prompt_shards": list(args.starts)},
            "backend": method,
            "videos": videos,
            "summary": summarize(videos),
            "attention_error_vs_dense": merge_error(
                [shard["attention_error_vs_dense"] for shard in shards]
            ),
        }
        if method == "dense":
            payload["attention_error_vs_dense"]["baseline"] = "exact dense SDPA"
        (output_dir / "runtime.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps({"method": method, "videos": len(videos)}))


if __name__ == "__main__":
    main()
