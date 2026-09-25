"""Fit projection-foldable GMA calibrators on captured Wan QKV.

For each attention head, this script solves

    min_T ||Y_gma T - Y_dense||_F^2 + lambda ||T - I||_F^2.

The calibrated head output can be folded into the existing attention output
projection.  If PyTorch's projection is ``z = y W^T``, replacing each head
block of ``W`` by ``W T^T`` gives ``z = (y T) W^T`` with no inference-time
operator or FLOP increase.  This probe evaluates transfer on disjoint captured
QKV before any production integration or video generation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_attention_triton_head_major,
)


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.replace(",", " ").split()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def sample_indices(length: int, count: int, seed: int, device: torch.device):
    if count >= length:
        return torch.arange(length, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(length, generator=generator)[:count].sort().values.to(
        device
    )


def load_qkv(path: Path, device: torch.device):
    with np.load(path, allow_pickle=False) as capture:
        tensors = tuple(
            torch.from_numpy(np.asarray(capture[name])).to(
                device=device, dtype=torch.float16
            )
            for name in ("q", "k", "v")
        )
    return tensors


@torch.inference_mode()
def sampled_outputs(
    path: Path,
    *,
    sample_count: int,
    sample_seed: int,
    dense_query_chunk: int,
    device: torch.device,
):
    q, k, v = load_qkv(path, device)
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expected matching [B,H,N,D] QKV, got {q.shape}")
    if q.size(0) != 1:
        raise ValueError("the calibration probe currently expects batch size one")

    indices = sample_indices(q.size(2), sample_count, sample_seed, device)
    gma = hard_c4_720p_attention_triton_head_major(q, k, v)
    gma = gma.index_select(2, indices).squeeze(0).float()

    dense_chunks = []
    for start in range(0, indices.numel(), dense_query_chunk):
        chunk_indices = indices[start : start + dense_query_chunk]
        dense_chunks.append(
            F.scaled_dot_product_attention(
                q.index_select(2, chunk_indices),
                k,
                v,
                scale=q.size(-1) ** -0.5,
            )
        )
    dense = torch.cat(dense_chunks, dim=2).squeeze(0).float()
    del q, k, v
    torch.cuda.empty_cache()
    return gma, dense


def sufficient_statistics(pairs):
    first_x, _ = pairs[0]
    heads, _, dim = first_x.shape
    device = first_x.device
    xtx = torch.zeros(heads, dim, dim, device=device, dtype=torch.float64)
    xty = torch.zeros_like(xtx)
    samples = 0
    for x, y in pairs:
        x64, y64 = x.double(), y.double()
        xtx.add_(torch.einsum("hnd,hne->hde", x64, x64))
        xty.add_(torch.einsum("hnd,hne->hde", x64, y64))
        samples += x.size(1)
    return xtx, xty, samples


def fit_map(xtx, xty, ridge_relative: float):
    heads, dim, _ = xtx.shape
    identity = torch.eye(dim, device=xtx.device, dtype=xtx.dtype).expand(
        heads, -1, -1
    )
    mean_energy = xtx.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / dim
    ridge = float(ridge_relative) * mean_energy.clamp_min(
        torch.finfo(xtx.dtype).tiny
    )
    return torch.linalg.solve(
        xtx + ridge[:, None, None] * identity,
        xty + ridge[:, None, None] * identity,
    ).float()


def truncate_residual(transform: torch.Tensor, rank: int):
    if rank <= 0 or rank >= transform.size(-1):
        return transform
    identity = torch.eye(
        transform.size(-1), device=transform.device, dtype=transform.dtype
    )
    residual = transform - identity
    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    residual = torch.einsum(
        "hdr,hr,hre->hde", u[..., :rank], s[..., :rank], vh[..., :rank, :]
    )
    return identity + residual


def error_metrics(x, y, transform=None):
    prediction = (
        x
        if transform is None
        else torch.einsum("hnd,hde->hne", x, transform)
    )
    error = prediction - y
    squared = error.square()
    target_energy = y.square().sum().clamp_min(torch.finfo(y.dtype).tiny)
    return {
        "mse": float(squared.mean().item()),
        "relative_l2": float(torch.sqrt(squared.sum() / target_energy).item()),
        "cosine": float(
            F.cosine_similarity(
                prediction.reshape(-1, prediction.size(-1)),
                y.reshape(-1, y.size(-1)),
                dim=-1,
            )
            .mean()
            .item()
        ),
        "per_head_mse": squared.mean(dim=(1, 2)).tolist(),
    }


def aggregate_metrics(rows):
    keys = ("mse", "relative_l2", "cosine")
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--dense-query-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--ridge", type=parse_floats, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--ranks", type=parse_ints, default=[1, 2, 4, 8, 16, 32, 0])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native Hard-C4 probe")
    device = torch.device("cuda")

    train_pairs = []
    train_rows = []
    for index, path in enumerate(args.train):
        x, y = sampled_outputs(
            path,
            sample_count=args.sample_count,
            sample_seed=args.seed + index,
            dense_query_chunk=args.dense_query_chunk,
            device=device,
        )
        train_pairs.append((x, y))
        train_rows.append({"path": str(path), "base": error_metrics(x, y)})

    xtx, xty, train_samples = sufficient_statistics(train_pairs)
    transforms = {
        str(ridge): fit_map(xtx, xty, ridge) for ridge in args.ridge
    }

    eval_pairs = []
    eval_rows = []
    for index, path in enumerate(args.eval):
        x, y = sampled_outputs(
            path,
            sample_count=args.sample_count,
            sample_seed=args.seed + 1000 + index,
            dense_query_chunk=args.dense_query_chunk,
            device=device,
        )
        eval_pairs.append((x, y))
        eval_rows.append({"path": str(path), "base": error_metrics(x, y)})

    candidates = {}
    for ridge_label, transform in transforms.items():
        for rank in args.ranks:
            calibrated = truncate_residual(transform, rank)
            label = f"ridge_{ridge_label}_rank_{rank or 'full'}"
            train_metrics = [
                error_metrics(x, y, calibrated) for x, y in train_pairs
            ]
            eval_metrics = [
                error_metrics(x, y, calibrated) for x, y in eval_pairs
            ]
            base_eval = [row["base"] for row in eval_rows]
            aggregate_eval = aggregate_metrics(eval_metrics)
            aggregate_base = aggregate_metrics(base_eval)
            candidates[label] = {
                "ridge_relative": float(ridge_label),
                "rank": rank or transform.size(-1),
                "train": aggregate_metrics(train_metrics),
                "eval": aggregate_eval,
                "eval_mse_ratio_vs_c4": (
                    aggregate_eval["mse"] / aggregate_base["mse"]
                ),
                "per_capture_eval": eval_metrics,
            }

    best_label = min(
        candidates, key=lambda label: candidates[label]["eval_mse_ratio_vs_c4"]
    )
    result = {
        "method": "projection-folded per-head GMA calibration",
        "objective": "min ||Y_gma T - Y_dense||^2 + lambda ||T-I||^2",
        "folding": "replace each input-head block of W_o by W_o T_h^T",
        "inference_extra_flops": 0,
        "attention_ratio_vs_dense": (3 * 4 * 80 + 2 * 45 * 4) / (2 * 45 * 80),
        "attention_ratio_vs_svg_ear_density_0.3790645331": (
            (3 * 4 * 80 + 2 * 45 * 4)
            / (2 * 45 * 80)
            / 0.37906453310046345
        ),
        "train_samples_per_head": train_samples,
        "train": train_rows,
        "eval": eval_rows,
        "best": {"label": best_label, **candidates[best_label]},
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
