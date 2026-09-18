"""Combine runtime, paired fidelity, and VBench custom-input results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = {
    "dense": "Dense",
    "svg": "SVG",
    "svg2": "SVG-2",
    "ear": "SVG-EAR",
    "gma": "GMA",
}
SPARSITY = {
    "dense": "0%",
    "svg": "30% target (24% schedule avg)",
    "svg2": "Adaptive (top-p=0.9)",
    "ear": "Estimated adaptive routing (top-p=0.9)",
    "gma": "95% KV-token (80->4)",
}
VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)


def vbench_scores(method_dir):
    result_paths = sorted(Path(method_dir).glob("*_eval_results.json"))
    if not result_paths:
        return None, {}
    raw = json.loads(result_paths[-1].read_text(encoding="utf-8"))
    scores = {}
    for dimension in VBENCH_DIMENSIONS:
        value = raw[dimension]
        scores[dimension] = float(value[0] if isinstance(value, list) else value)
    return sum(scores.values()) / len(scores), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--quality_json", required=True)
    parser.add_argument("--vbench_root", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument(
        "--methods",
        default="dense,svg,svg2,gma",
        help="Comma-separated backends to summarize.",
    )
    args = parser.parse_args()

    selected = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = [method for method in selected if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    experiment_root = Path(args.experiment_root)
    quality = json.loads(Path(args.quality_json).read_text(encoding="utf-8"))
    dense_runtime = json.loads(
        (experiment_root / "dense" / "runtime.json").read_text(encoding="utf-8")
    )["summary"]
    dense_attention_ms = dense_runtime["steady_state_mean_attention_ms"]
    dense_full_attention_ms = dense_runtime.get("steady_state_mean_full_attention_ms")
    rows = []
    prompt_count = min(
        json.loads(
            (experiment_root / backend / "runtime.json").read_text(encoding="utf-8")
        )["summary"]["mp4_count"]
        for backend in selected
    )

    for backend in selected:
        method = METHODS[backend]
        runtime = json.loads(
            (experiment_root / backend / "runtime.json").read_text(encoding="utf-8")
        )["summary"]
        if backend == "dense":
            fidelity = {"psnr": "Inf (reference)", "ssim": 1.0, "lpips": 0.0}
        else:
            fidelity = quality["methods"][backend]["summary"]
        vbench, dimensions = vbench_scores(Path(args.vbench_root) / backend)
        attention_ms = runtime["steady_state_mean_attention_ms"]
        full_attention_ms = runtime.get("steady_state_mean_full_attention_ms")
        rows.append(
            {
                "method": method,
                "model": "Wan2.1-14B",
                "resolution": "1280x720",
                "sparsity": SPARSITY[backend],
                "psnr": fidelity["psnr"],
                "ssim": fidelity["ssim"],
                "lpips": fidelity["lpips"],
                "vbench_custom_six_mean": vbench,
                "vbench_dimensions": dimensions,
                "mean_attention_ms": attention_ms,
                "speedup_vs_sdpa": dense_attention_ms / attention_ms,
                "mean_full_attention_ms": full_attention_ms,
                "full_attention_speedup_vs_sdpa": (
                    dense_full_attention_ms / full_attention_ms
                    if dense_full_attention_ms is not None
                    and full_attention_ms is not None
                    else None
                ),
                "end_to_end_seconds": runtime["steady_state_mean_inference_seconds"],
                "timing_videos": runtime["steady_state_videos"],
            }
        )

    payload = {
        "protocol": {
            "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            "resolution": "1280x720",
            "frames": 81,
            "denoising_steps": 50,
            "prompts": prompt_count,
            "seed": 0,
            "timing": (
                "per-method video 1 excluded as warmup; model loading and video "
                "export excluded"
            ),
            "vbench": "mean of six official custom-input dimensions",
            "vbench_dimensions": VBENCH_DIMENSIONS,
            "sparsity": {
                "svg": (
                    "30% target attention-map sparsity during sparse steps; "
                    "24% call-weighted average because steps 1-10 of 50 are dense."
                ),
                "svg2": "Adaptive sparsity; top-p=0.9 with min_kc_ratio=0.10.",
                "ear": (
                    "Estimated adaptive routing; QC=300, KC=1000, top-p=0.9, "
                    "min_kc_ratio=0.10."
                ),
                "gma": "Compressed-token sparsity: 1 - 4/80 = 95%.",
            },
        },
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with Path(args.output_tsv).open("w", newline="", encoding="utf-8") as handle:
        columns = [key for key in rows[0] if key != "vbench_dimensions"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


if __name__ == "__main__":
    main()
