"""Incrementally evaluate completed videos against a dense reference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluate_psnr_ssim import aggregate, evaluate_pair


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--method", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial_json")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--poll_seconds", type=int, default=60)
    return parser.parse_args()


def completed_videos(method_dir):
    runtime_path = Path(method_dir) / "runtime.json"
    if not runtime_path.exists():
        return set()
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if records is not None:
        return {record["output"] for record in records if record.get("output")}
    return {record["video"] for record in payload.get("videos", [])}


def save_result(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = output if output.exists() else Path(args.initial_json or "")
    if initial.is_file():
        result = json.loads(initial.read_text(encoding="utf-8"))
    else:
        result = {
            "reference": str(Path(args.reference_dir).resolve()),
            "protocol": "same filename/prompt/seed; framewise RGB PSNR and SSIM",
            "methods": {},
        }

    methods = [(name, Path(directory)) for name, directory in args.method]
    for name, _ in methods:
        result["methods"].setdefault(name, {"summary": {}, "videos": []})
    save_result(output, result)

    while True:
        added = False
        references = sorted(Path(args.reference_dir).glob("*.mp4"))[: args.count]
        for name, method_dir in methods:
            method_result = result["methods"][name]
            existing = {video["video"] for video in method_result["videos"]}
            completed = completed_videos(method_dir)
            for reference_path in references:
                if reference_path.name in existing or reference_path.name not in completed:
                    continue
                video = evaluate_pair(reference_path, method_dir / reference_path.name)
                method_result["videos"].append(video)
                method_result["videos"].sort(key=lambda item: item["video"])
                method_result["summary"] = aggregate(method_result["videos"])
                save_result(output, result)
                print(json.dumps({"method": name, **video}), flush=True)
                added = True

        if all(len(result["methods"][name]["videos"]) >= args.count for name, _ in methods):
            break
        if not added:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
