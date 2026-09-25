"""Measure paired video fidelity against a dense reference directory."""

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

from utils.prompt_contract import (
    assert_matching_prompt_manifests,
    load_output_prompt_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--method_dir", action="append", required=True)
    parser.add_argument("--method_name", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lpips_batch_size", type=int, default=8)
    parser.add_argument("--max_videos", type=int)
    parser.add_argument(
        "--expected_videos", type=int,
        help="Require exactly the canonical 000-00..NNN-00 MP4 set in every directory.",
    )
    parser.add_argument(
        "--expected_frames", type=int,
        help="Require exactly this many decoded frames in every paired video.",
    )
    return parser.parse_args()


def paired_paths(reference_dir, method_dir, max_videos=None):
    reference_paths = sorted(Path(reference_dir).glob("*.mp4"))
    if max_videos is not None:
        reference_paths = reference_paths[:max_videos]
    pairs = [(path, Path(method_dir) / path.name) for path in reference_paths]
    missing = [str(method) for _, method in pairs if not method.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} paired videos: {missing[:3]}")
    if not pairs:
        raise ValueError("No paired videos; refusing empty quality comparison")
    return pairs


def prepare_comparison(reference_dir, method_dir, max_videos=None, expected_videos=None):
    """Validate every pair before importing/loading the LPIPS model.

    Strict expected-count mode is intentionally limited to one sample per
    prompt with numeric filenames. Without it, existing filename and
    ``max_videos`` behavior is preserved, while empty comparisons are rejected.
    """
    prompt_indices = None
    if expected_videos is not None:
        if expected_videos < 1:
            raise ValueError("expected_videos must be positive")
        if max_videos is not None and max_videos != expected_videos:
            raise ValueError("max_videos must equal expected_videos in strict mode")
        expected_names = {f"{index:03d}-00.mp4" for index in range(expected_videos)}
        for directory in (reference_dir, method_dir):
            manifest = load_output_prompt_manifest(directory)
            if manifest.get("num_samples") != 1 or manifest.get("vbench_filenames") is not False:
                raise ValueError(
                    "expected_videos strict mode requires num_samples=1 "
                    f"and vbench_filenames=false: {directory}"
                )
            actual_names = {path.name for path in Path(directory).glob("*.mp4")}
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                extra = sorted(actual_names - expected_names)
                raise ValueError(
                    f"Expected exactly {expected_videos} canonical paired videos in {directory}; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
        prompt_indices = range(expected_videos)
    contract = assert_matching_prompt_manifests(
        reference_dir, method_dir, prompt_indices=prompt_indices
    )
    return paired_paths(reference_dir, method_dir, max_videos), contract


def frames(path):
    yield from iio.imiter(path, plugin="FFMPEG")


def paired_frames(reference_path, method_path, expected_frames=None):
    """Decode every paired frame and reject missing, extra, or mismatched frames."""
    if expected_frames is not None and expected_frames < 1:
        raise ValueError("expected_frames must be positive")
    count = 0
    for count, (reference, method) in enumerate(
        zip(frames(reference_path), frames(method_path), strict=True), start=1
    ):
        if reference.shape != method.shape:
            raise ValueError(
                f"Shape mismatch for {reference_path.name}: "
                f"{reference.shape} vs {method.shape}"
            )
        if expected_frames is not None and count > expected_frames:
            raise ValueError(
                f"Frame count mismatch for {reference_path.name}: "
                f"expected {expected_frames}, decoded more than {expected_frames}"
            )
        yield reference, method
    if not count:
        raise ValueError(f"No frames decoded from {reference_path}")
    if expected_frames is not None and count != expected_frames:
        raise ValueError(
            f"Frame count mismatch for {reference_path.name}: "
            f"expected {expected_frames}, decoded {count}"
        )


def lpips_tensor(batch, device):
    import torch

    array = np.stack(batch).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).to(device)


def evaluate_pair(reference_path, method_path, lpips_model, batch_size, device, expected_frames=None):
    psnr_values = []
    ssim_values = []
    lpips_values = []
    reference_batch = []
    method_batch = []

    def flush_lpips():
        if not reference_batch:
            return
        import torch

        with torch.inference_mode():
            distance = lpips_model(
                lpips_tensor(reference_batch, device),
                lpips_tensor(method_batch, device),
            )
        lpips_values.extend(distance.flatten().float().cpu().tolist())
        reference_batch.clear()
        method_batch.clear()

    frame_count = 0
    for reference, method in paired_frames(reference_path, method_path, expected_frames):
        psnr_values.append(peak_signal_noise_ratio(reference, method, data_range=255))
        ssim_values.append(
            structural_similarity(reference, method, channel_axis=2, data_range=255)
        )
        reference_batch.append(reference)
        method_batch.append(method)
        frame_count += 1
        if len(reference_batch) == batch_size:
            flush_lpips()
    flush_lpips()

    return {
        "video": reference_path.name,
        "frames": frame_count,
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "lpips": float(np.mean(lpips_values)),
    }


def aggregate(videos):
    weights = np.array([video["frames"] for video in videos], dtype=np.float64)
    return {
        metric: float(np.average([video[metric] for video in videos], weights=weights))
        for metric in ("psnr", "ssim", "lpips")
    } | {"videos": len(videos), "frames": int(weights.sum())}


def main():
    args = parse_args()
    if len(args.method_dir) != len(args.method_name):
        raise ValueError("--method_dir and --method_name counts must match")
    if args.expected_frames is not None and args.expected_frames < 1:
        raise ValueError("expected_frames must be positive")
    if args.lpips_batch_size < 1:
        raise ValueError("lpips_batch_size must be positive")
    comparisons = []
    for name, directory in zip(args.method_name, args.method_dir, strict=True):
        pairs, contract = prepare_comparison(
            args.reference_dir, directory, args.max_videos, args.expected_videos
        )
        comparisons.append((name, pairs, contract))

    import lpips
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    result = {
        "reference": str(Path(args.reference_dir).resolve()),
        "protocol": {
            "pairing": "same filename/prompt/seed; framewise comparison",
            "psnr_data_range": 255,
            "ssim_channel_axis": 2,
            "lpips_backbone": "alex",
            "expected_videos": args.expected_videos,
            "expected_frames": args.expected_frames,
        },
        "methods": {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for name, pairs, prompt_contract in comparisons:
        videos = []
        for reference_path, method_path in pairs:
            video = evaluate_pair(
                reference_path,
                method_path,
                lpips_model,
                args.lpips_batch_size,
                device,
                expected_frames=args.expected_frames,
            )
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
