#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


BACKENDS = ("cudnn", "flash")


def mean(records, key):
    values = [record[key] for record in records]
    return sum(values) / len(values)


def summarize(root, backend):
    stage = root / backend
    runtime_path = stage / "runtime.json"
    manifest_path = stage / "manifest.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = runtime.get("records", [])
    if len(records) != 20:
        raise ValueError(f"{backend}: expected 20 records, found {len(records)}")
    outputs = [record["output"] for record in records]
    expected = [f"{index:03d}-00.mp4" for index in range(20)]
    if outputs != expected:
        raise ValueError(f"{backend}: output order or names do not match")
    calls = {record["attention_calls"] for record in records}
    full_calls = {record["full_attention_calls"] for record in records}
    if calls != {160} or full_calls != {160}:
        raise ValueError(
            f"{backend}: unexpected calls attention={calls}, full={full_calls}")

    steady = records[1:]
    core_ms = mean(steady, "mean_attention_ms")
    full_ms = mean(steady, "mean_full_attention_ms")
    inference_seconds = mean(steady, "inference_seconds")
    return {
        "backend": backend,
        "videos": len(records),
        "steady_state_videos": len(steady),
        "attention_calls_per_video": 160,
        "core_attention_ms_per_call": core_ms,
        "full_attention_ms_per_call": full_ms,
        "inference_seconds_per_video": inference_seconds,
        "attention_share": core_ms * 160 / (1000 * inference_seconds),
        "first_video": {
            "core_attention_ms_per_call": records[0]["mean_attention_ms"],
            "full_attention_ms_per_call": records[0]["mean_full_attention_ms"],
            "inference_seconds": records[0]["inference_seconds"],
        },
        "software": {
            key: manifest[key]
            for key in ("git_commit", "python", "torch", "cuda", "cudnn", "triton")
        },
        "setting": {
            key: manifest[key]
            for key in (
                "checkpoint",
                "config",
                "prompts",
                "seed",
                "resolution",
                "frames",
                "denoise_steps",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    methods = {backend: summarize(args.root, backend) for backend in BACKENDS}
    cudnn = methods["cudnn"]
    for method in methods.values():
        method["core_speed_vs_cudnn"] = (
            cudnn["core_attention_ms_per_call"]
            / method["core_attention_ms_per_call"]
        )
        method["full_speed_vs_cudnn"] = (
            cudnn["full_attention_ms_per_call"]
            / method["full_attention_ms_per_call"]
        )

    result = {
        "comparison": "same H200, code, checkpoint, prompts, seed, and QKV stream",
        "methods": methods,
    }
    results_dir = args.root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    fields = [
        "backend",
        "videos",
        "steady_state_videos",
        "attention_calls_per_video",
        "core_attention_ms_per_call",
        "full_attention_ms_per_call",
        "inference_seconds_per_video",
        "attention_share",
        "core_speed_vs_cudnn",
        "full_speed_vs_cudnn",
    ]
    with (results_dir / "summary.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for backend in BACKENDS:
            writer.writerow({key: methods[backend][key] for key in fields})


if __name__ == "__main__":
    main()
