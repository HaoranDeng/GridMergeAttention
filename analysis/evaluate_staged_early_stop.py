"""Evaluate a candidate against SVG-EAR and make a staged early-stop decision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(frozen=True)
class Gate:
    min_psnr_delta: float
    min_ssim_delta: float
    max_lpips_delta: float
    max_latency_ratio: float


STAGE_GATES = {
    1: Gate(-0.60, -0.040, 0.050, 1.00),
    3: Gate(-0.35, -0.025, 0.035, 0.98),
    5: Gate(-0.15, -0.015, 0.020, 0.95),
    20: Gate(0.00, 0.000, 0.000, 1.00),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_videos", type=int, choices=STAGE_GATES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--theoretical_ratio_vs_svg_ear", type=float, required=True)
    parser.add_argument("--max_theoretical_ratio", type=float, default=0.70)
    return parser.parse_args()


def video_names(candidate_dir: Path, expected_videos: int):
    names = sorted(path.name for path in candidate_dir.glob("*.mp4"))
    if len(names) < expected_videos:
        raise RuntimeError(
            f"expected {expected_videos} candidate videos, found {len(names)}"
        )
    return names[:expected_videos]


def evaluate_rgb(reference_dir: Path, method_dir: Path, names: list[str]):
    from evaluate_psnr_ssim import aggregate, evaluate_pair

    videos = [
        evaluate_pair(reference_dir / name, method_dir / name)
        for name in names
    ]
    return {"summary": aggregate(videos), "videos": videos}


def evaluate_lpips(metric, reference_dir, method_dir, names, device):
    import imageio.v3 as iio
    import numpy as np
    import torch

    from evaluate_lpips import frame_tensor

    videos = []
    with torch.inference_mode():
        for name in names:
            values = []
            references = iio.imiter(reference_dir / name, plugin="FFMPEG")
            methods = iio.imiter(method_dir / name, plugin="FFMPEG")
            for reference, method in zip(references, methods, strict=True):
                value = metric(
                    frame_tensor(reference, device),
                    frame_tensor(method, device),
                )
                values.append(float(value.item()))
            if not values:
                raise RuntimeError(f"no frames decoded from {name}")
            videos.append(
                {"video": name, "frames": len(values), "lpips": float(np.mean(values))}
            )
    weights = np.asarray([video["frames"] for video in videos], dtype=np.float64)
    return {
        "summary": {
            "lpips": float(
                np.average([video["lpips"] for video in videos], weights=weights)
            ),
            "videos": len(videos),
            "frames": int(weights.sum()),
        },
        "videos": videos,
    }


def runtime_latency(runtime_path: Path, names: list[str]):
    payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    raw_records = payload.get("videos", payload.get("records"))
    if raw_records is None:
        raise KeyError(f"runtime has neither videos nor records: {runtime_path}")
    records = {
        record.get("video", record.get("output")): record
        for record in raw_records
    }
    selected = [records[name] for name in names]
    steady = selected[1:] if len(selected) > 1 else selected
    calls = sum(record["full_attention_calls"] for record in steady)
    total_ms = sum(record["full_attention_total_ms"] for record in steady)
    if calls <= 0:
        raise RuntimeError(f"no attention calls in {runtime_path}")
    return {
        "mean_full_attention_ms": total_ms / calls,
        "attention_calls": calls,
        "videos": len(steady),
        "excludes_first_video": len(selected) > 1,
    }


def gate_decision(
    candidate,
    baseline,
    runtime_ratio,
    theoretical_ratio,
    gate,
    max_theoretical_ratio=0.70,
):
    deltas = {
        "psnr": candidate["rgb"]["summary"]["psnr"]
        - baseline["rgb"]["summary"]["psnr"],
        "ssim": candidate["rgb"]["summary"]["ssim"]
        - baseline["rgb"]["summary"]["ssim"],
        "lpips": candidate["lpips"]["summary"]["lpips"]
        - baseline["lpips"]["summary"]["lpips"],
    }
    checks = {
        "psnr": deltas["psnr"] >= gate.min_psnr_delta,
        "ssim": deltas["ssim"] >= gate.min_ssim_delta,
        "lpips": deltas["lpips"] <= gate.max_lpips_delta,
        "latency": runtime_ratio <= gate.max_latency_ratio,
        "theoretical_flops": theoretical_ratio <= max_theoretical_ratio,
    }
    return deltas, checks, all(checks.values())


def main():
    args = parse_args()
    reference_dir = Path(args.reference_dir)
    candidate_dir = Path(args.candidate_dir)
    baseline_dir = Path(args.baseline_dir)
    names = video_names(candidate_dir, args.expected_videos)
    for directory in (reference_dir, baseline_dir):
        missing = [name for name in names if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"missing paired videos in {directory}: {missing}")

    import lpips

    metric = lpips.LPIPS(net="alex").to(args.device).eval()
    candidate = {
        "rgb": evaluate_rgb(reference_dir, candidate_dir, names),
        "lpips": evaluate_lpips(
            metric, reference_dir, candidate_dir, names, args.device
        ),
        "runtime": runtime_latency(candidate_dir / "runtime.json", names),
    }
    baseline = {
        "rgb": evaluate_rgb(reference_dir, baseline_dir, names),
        "lpips": evaluate_lpips(
            metric, reference_dir, baseline_dir, names, args.device
        ),
        "runtime": runtime_latency(baseline_dir / "runtime.json", names),
    }
    runtime_ratio = (
        candidate["runtime"]["mean_full_attention_ms"]
        / baseline["runtime"]["mean_full_attention_ms"]
    )
    gate = STAGE_GATES[args.expected_videos]
    deltas, checks, passed = gate_decision(
        candidate,
        baseline,
        runtime_ratio,
        args.theoretical_ratio_vs_svg_ear,
        gate,
        args.max_theoretical_ratio,
    )
    result = {
        "candidate": args.candidate_name,
        "baseline": "SVG-EAR all-layer",
        "videos": names,
        "stage": args.expected_videos,
        "gate": asdict(gate),
        "candidate_metrics": candidate,
        "baseline_metrics": baseline,
        "deltas_candidate_minus_baseline": deltas,
        "full_attention_latency_ratio": runtime_ratio,
        "theoretical_flops_ratio_vs_svg_ear": args.theoretical_ratio_vs_svg_ear,
        "max_theoretical_flops_ratio_vs_svg_ear": args.max_theoretical_ratio,
        "checks": checks,
        "decision": "promote" if passed else "reject",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
