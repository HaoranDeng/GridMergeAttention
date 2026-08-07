#!/usr/bin/env python
"""Compare latents saved by this repository and MonarchRT.

Run both dense causal pipelines with Token Merging disabled, the same EMA
checkpoint, prompt embeddings, initial noise, and per-step re-noising tensors.
Save each returned latent tensor with ``torch.save`` and pass the two paths to
this script. Comparing latents is much more informative than comparing encoded
MP4 files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path, help="Latents from this repository")
    parser.add_argument("monarchrt", type=Path, help="Reference latents from MonarchRT")
    parser.add_argument("--prefix-frames", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser.parse_args()


def load_latents(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("latents", "output", "sample"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor or a recognized latent dictionary")
    return payload.float()


def main() -> None:
    args = parse_args()
    current = load_latents(args.current)
    reference = load_latents(args.monarchrt)
    if current.shape != reference.shape:
        raise AssertionError(
            f"latent shape mismatch: current={tuple(current.shape)}, "
            f"MonarchRT={tuple(reference.shape)}"
        )
    if args.prefix_frames is not None:
        current = current[:, : args.prefix_frames]
        reference = reference[:, : args.prefix_frames]

    difference = current - reference
    print(f"shape: {tuple(current.shape)}")
    print(f"mean absolute error: {difference.abs().mean().item():.8g}")
    print(f"maximum absolute error: {difference.abs().max().item():.8g}")
    print(f"root mean square error: {difference.square().mean().sqrt().item():.8g}")
    print(
        "cosine similarity: "
        f"{torch.nn.functional.cosine_similarity(current.flatten(), reference.flatten(), dim=0).item():.8g}"
    )
    torch.testing.assert_close(
        current,
        reference,
        rtol=args.rtol,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()

