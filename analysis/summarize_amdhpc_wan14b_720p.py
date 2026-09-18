"""Combine AMDHPC runtime, local attention error, PSNR, and SSIM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("dense", "svg", "svg2", "gma")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--quality_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_tsv", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    quality = json.loads(Path(args.quality_json).read_text(encoding="utf-8"))
    runtimes = {
        method: json.loads((root / method / "runtime.json").read_text(encoding="utf-8"))
        for method in METHODS
    }
    dense_attention_ms = runtimes["dense"]["summary"]["steady_state_mean_attention_ms"]

    rows = []
    for method in METHODS:
        runtime = runtimes[method]
        summary = runtime["summary"]
        error = runtime["attention_error_vs_dense"]
        fidelity = (
            {"psnr": "Inf (reference)", "ssim": 1.0}
            if method == "dense"
            else quality["methods"][method]["summary"]
        )
        attention_ms = summary["steady_state_mean_attention_ms"]
        rows.append(
            {
                "method": method,
                "videos": summary["mp4_count"],
                "psnr": fidelity["psnr"],
                "ssim": fidelity["ssim"],
                "attention_error_calls": error["calls"],
                "attention_mse_vs_dense": error["mse_vs_dense"],
                "attention_rel_l2_vs_dense": error["rel_l2_vs_dense"],
                "attention_max_abs_vs_dense": error["max_abs_vs_dense"],
                "attention_ms_per_call": attention_ms,
                "attention_speedup_vs_dense": dense_attention_ms / attention_ms,
                "seconds_per_video": summary["steady_state_mean_inference_seconds"],
                "timing_videos": summary["steady_state_videos"],
            }
        )

    payload = {
        "protocol": {
            "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "resolution": "1280x720",
            "frames": 81,
            "denoising_steps": 50,
            "videos_per_method": runtimes["dense"]["summary"]["mp4_count"],
            "seed": 0,
            "timing": (
                "videos 2-N; model loading, export, and error pass excluded"
            ),
            "attention_error": (
                "separate untimed inference; local method output versus dense SDPA "
                "on the same call-local QKV"
            ),
        },
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with Path(args.output_tsv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
