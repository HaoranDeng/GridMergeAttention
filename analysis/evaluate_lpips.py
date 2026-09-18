"""Measure paired framewise LPIPS against reference videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.prompt_contract import assert_matching_prompt_manifests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--method_dir", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_videos", type=int)
    return parser.parse_args()


def frame_tensor(frame, device):
    tensor = torch.from_numpy(np.asarray(frame).copy()).to(device=device)
    return tensor.permute(2, 0, 1).unsqueeze(0).float().div_(127.5).sub_(1.0)


def main():
    args = parse_args()
    import lpips

    reference_dir = Path(args.reference_dir)
    method_dir = Path(args.method_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_contract = assert_matching_prompt_manifests(reference_dir, method_dir)
    references = sorted(reference_dir.glob("*.mp4"))
    if args.max_videos is not None:
        references = references[: args.max_videos]
    metric = lpips.LPIPS(net="alex").to(args.device).eval()
    videos = []
    with torch.inference_mode():
        for reference_path in references:
            method_path = method_dir / reference_path.name
            if not method_path.exists():
                raise FileNotFoundError(method_path)
            values = []
            reference_frames = iio.imiter(reference_path, plugin="FFMPEG")
            method_frames = iio.imiter(method_path, plugin="FFMPEG")
            for reference, method in zip(reference_frames, method_frames, strict=True):
                value = metric(
                    frame_tensor(reference, args.device),
                    frame_tensor(method, args.device),
                )
                values.append(float(value.item()))
            video = {
                "video": reference_path.name,
                "frames": len(values),
                "lpips": float(np.mean(values)),
            }
            videos.append(video)
            weights = np.asarray([item["frames"] for item in videos])
            result = {
                "reference": str(reference_dir.resolve()),
                "method": args.method_name,
                "prompt_contract": prompt_contract,
                "summary": {
                    "lpips": float(np.average(
                        [item["lpips"] for item in videos], weights=weights
                    )),
                    "videos": len(videos),
                    "frames": int(weights.sum()),
                },
                "videos": videos,
            }
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(video), flush=True)


if __name__ == "__main__":
    main()
