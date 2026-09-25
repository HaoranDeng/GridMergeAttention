#!/usr/bin/env python
"""Aggregate per-worker Hunyuan timing JSON into one config-level report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Worker timing JSON files.")
    parser.add_argument("--label", default=None, help="Config/run name.")
    parser.add_argument("--output", required=True, help="Summary JSON path.")
    return parser.parse_args()


def load_report(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema_version") != 1:
        raise ValueError(f"unsupported timing schema in {path}")
    generated = int(report.get("generated_videos", -1))
    if generated < 0:
        raise ValueError(f"invalid generated_videos in {path}")
    if len(report.get("samples", [])) != generated:
        raise ValueError(f"sample count does not match generated_videos in {path}")
    return report


def summarize_reports(reports, label=None, source_reports=None):
    """Compute weighted latency and actual parallel-window throughput."""

    if not reports:
        raise ValueError("at least one worker timing report is required")

    generated = sum(int(report["generated_videos"]) for report in reports)
    generation_total = sum(
        float(report["pipeline_generation_seconds_total"])
        for report in reports
    )
    export_total = sum(
        float(report["video_export_seconds_total"]) for report in reports
    )
    model_load_total = sum(
        float(report["model_load_seconds"]) for report in reports
    )

    active = [report for report in reports if report["generated_videos"] > 0]
    if active:
        work_start = min(float(report["work_started_unix"]) for report in active)
        work_finish = max(float(report["work_finished_unix"]) for report in active)
        parallel_work_wall = max(0.0, work_finish - work_start)
    else:
        work_start = None
        work_finish = None
        parallel_work_wall = 0.0

    process_start = min(float(report["process_started_unix"]) for report in reports)
    process_finish = max(float(report["process_finished_unix"]) for report in reports)
    process_wall = max(0.0, process_finish - process_start)

    return {
        "schema_version": 1,
        "label": label,
        "worker_count": len(reports),
        "active_worker_count": len(active),
        "generated_videos": generated,
        "pipeline_generation_seconds_total_worker_time": generation_total,
        "average_pipeline_generation_seconds_per_video": (
            generation_total / generated if generated else None
        ),
        "pipeline_generation_videos_per_second_worker_equivalent": (
            generated / generation_total
            if generated and generation_total > 0
            else None
        ),
        "video_export_seconds_total_worker_time": export_total,
        "average_video_export_seconds_per_video": (
            export_total / generated if generated else None
        ),
        "effective_multi_gpu_work_wall_seconds": parallel_work_wall,
        "effective_multi_gpu_seconds_per_video": (
            parallel_work_wall / generated if generated else None
        ),
        "effective_multi_gpu_videos_per_second": (
            generated / parallel_work_wall
            if generated and parallel_work_wall > 0
            else None
        ),
        "worker_process_wall_seconds_including_model_load": process_wall,
        "model_load_seconds_mean_per_worker": model_load_total / len(reports),
        "model_load_seconds_max_worker": max(
            float(report["model_load_seconds"]) for report in reports
        ),
        "work_started_unix": work_start,
        "work_finished_unix": work_finish,
        "process_started_unix": process_start,
        "process_finished_unix": process_finish,
        "source_reports": list(source_reports or []),
    }


def write_json_atomic(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def print_summary(summary):
    label = summary.get("label") or "Hunyuan run"
    count = summary["generated_videos"]
    print(
        f"Timing {label}: {count} newly generated videos across "
        f"{summary['worker_count']} workers."
    )
    if not count:
        print("  No timing average is available because every video was skipped.")
        return
    print(
        "  Average pipeline generation latency: "
        f"{summary['average_pipeline_generation_seconds_per_video']:.3f} s/video "
        "(text encoding + denoising + VAE decode; excludes load/export)."
    )
    effective_seconds = summary["effective_multi_gpu_seconds_per_video"]
    effective_rate = summary["effective_multi_gpu_videos_per_second"]
    if effective_seconds is None or effective_rate is None:
        print("  Effective multi-GPU throughput: unavailable (zero timing span).")
    else:
        print(
            "  Effective multi-GPU throughput: "
            f"{effective_seconds:.3f} s/video, "
            f"{effective_rate:.6f} videos/s "
            f"over {summary['effective_multi_gpu_work_wall_seconds']:.3f} s."
        )
    print(
        "  Worker-process wall time including model loading: "
        f"{summary['worker_process_wall_seconds_including_model_load']:.3f} s."
    )


def main():
    args = parse_args()
    reports = [load_report(path) for path in args.reports]
    summary = summarize_reports(
        reports,
        label=args.label,
        source_reports=[str(Path(path)) for path in args.reports],
    )
    write_json_atomic(summary, args.output)
    print_summary(summary)
    print(f"  Summary report: {args.output}")


if __name__ == "__main__":
    main()
