"""Measure paired framewise PSNR and SSIM against dense videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.prompt_contract import assert_matching_prompt_manifests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--method_dir", action="append", required=True)
    parser.add_argument("--method_name", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_videos", type=int)
    return parser.parse_args()


def paired_paths(reference_dir, method_dir, max_videos=None):
    references = sorted(Path(reference_dir).glob("*.mp4"))
    if max_videos is not None:
        references = references[:max_videos]
    pairs = [(path, Path(method_dir) / path.name) for path in references]
    missing = [str(method) for _, method in pairs if not method.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} paired videos: {missing[:3]}")
    return pairs


def evaluate_pair(reference_path, method_path):
    psnr_values = []
    ssim_values = []
    reference_frames = iio.imiter(reference_path, plugin="FFMPEG")
    method_frames = iio.imiter(method_path, plugin="FFMPEG")
    for reference, method in zip(reference_frames, method_frames, strict=True):
        if reference.shape != method.shape:
            raise ValueError(
                f"Shape mismatch for {reference_path.name}: "
                f"{reference.shape} vs {method.shape}"
            )
        psnr_values.append(peak_signal_noise_ratio(reference, method, data_range=255))
        ssim_values.append(
            structural_similarity(reference, method, channel_axis=2, data_range=255)
        )
    if not psnr_values:
        raise ValueError(f"No frames decoded from {reference_path}")
    return {
        "video": reference_path.name,
        "frames": len(psnr_values),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def aggregate(videos):
    weights = np.asarray([video["frames"] for video in videos], dtype=np.float64)
    return {
        metric: float(np.average([video[metric] for video in videos], weights=weights))
        for metric in ("psnr", "ssim")
    } | {"videos": len(videos), "frames": int(weights.sum())}


def main():
    args = parse_args()
    if len(args.method_dir) != len(args.method_name):
        raise ValueError("--method_dir and --method_name counts must match")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "reference": str(Path(args.reference_dir).resolve()),
        "protocol": "same filename/prompt/seed; framewise RGB PSNR and SSIM",
        "methods": {},
    }
    for name, directory in zip(args.method_name, args.method_dir, strict=True):
        prompt_contract = assert_matching_prompt_manifests(
            args.reference_dir, directory
        )
        videos = []
        for reference_path, method_path in paired_paths(
            args.reference_dir, directory, args.max_videos
        ):
            video = evaluate_pair(reference_path, method_path)
            videos.append(video)
            result["methods"][name] = {
                "prompt_contract": prompt_contract,
                "summary": aggregate(videos),
                "videos": videos,
            }
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"method": name, **video}), flush=True)


if __name__ == "__main__":
    main()
