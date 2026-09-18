"""Probe a query-conditioned, hardware-regular correction for pure Hard C4.

For each attention head, fit the centered ridge objective

    min_B ||[Y_gma, Q] B + b - Y_dense||_F^2
          + lambda ||B - [I, 0]^T||_F^2.

The first block can be folded into the output projection.  The second block is
one regular head-wise Q projection, so the method remains GPU friendly and its
extra arithmetic is negligible compared with dense attention.  This script is
only an offline real-QKV screen; it does not modify generation behavior.
"""

from __future__ import annotations

import argparse
import json
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
        return tuple(
            torch.from_numpy(np.asarray(capture[name])).to(
                device=device, dtype=torch.float16
            )
            for name in ("q", "k", "v")
        )


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
        raise ValueError("the query-conditioned probe expects batch size one")

    indices = sample_indices(q.size(2), sample_count, sample_seed, device)
    gma = hard_c4_720p_attention_triton_head_major(q, k, v)
    gma = gma.index_select(2, indices).squeeze(0).float()
    query = q.index_select(2, indices).squeeze(0).float()

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
    return query, gma, dense


def concatenate(pairs):
    return tuple(torch.cat(items, dim=1) for items in zip(*pairs))


def fit_centered_ridge(
    features: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
    ridge_relative: float,
):
    feature_mean = features.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    x = (features - feature_mean).double()
    y = (target - target_mean).double()
    xtx = torch.einsum("hnp,hnq->hpq", x, x)
    xty = torch.einsum("hnp,hnd->hpd", x, y)
    feature_dim = features.size(-1)
    identity = torch.eye(
        feature_dim, device=features.device, dtype=torch.float64
    ).expand(features.size(0), -1, -1)
    mean_energy = xtx.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / feature_dim
    ridge = float(ridge_relative) * mean_energy.clamp_min(
        torch.finfo(torch.float64).tiny
    )
    weight = torch.linalg.solve(
        xtx + ridge[:, None, None] * identity,
        xty + ridge[:, None, None] * prior.double(),
    ).float()
    bias = target_mean - torch.einsum("hnp,hpd->hnd", feature_mean, weight)
    return weight, bias.squeeze(1)


def truncate_rank(weight: torch.Tensor, rank: int):
    if rank <= 0 or rank >= min(weight.shape[-2:]):
        return weight
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    return torch.einsum(
        "hpr,hr,hrd->hpd", u[..., :rank], s[..., :rank], vh[..., :rank, :]
    )


def make_bias(
    query: torch.Tensor,
    gma: torch.Tensor,
    dense: torch.Tensor,
    gma_weight: torch.Tensor,
    query_weight: torch.Tensor,
):
    return (
        dense.mean(dim=1)
        - torch.einsum("hnd,hde->hne", gma.mean(dim=1, keepdim=True), gma_weight)
        .squeeze(1)
        - torch.einsum(
            "hnd,hde->hne", query.mean(dim=1, keepdim=True), query_weight
        ).squeeze(1)
    )


def predict(pair, model):
    query, gma, _ = pair
    output = torch.einsum("hnd,hde->hne", gma, model["gma_weight"])
    output.add_(torch.einsum("hnd,hde->hne", query, model["query_weight"]))
    output.add_(model["bias"][:, None, :])
    return output


def error_metrics(prediction, target):
    error = prediction - target
    squared = error.square()
    target_energy = target.square().sum().clamp_min(
        torch.finfo(target.dtype).tiny
    )
    return {
        "mse": float(squared.mean().item()),
        "relative_l2": float(torch.sqrt(squared.sum() / target_energy).item()),
        "cosine": float(
            F.cosine_similarity(
                prediction.reshape(-1, prediction.size(-1)),
                target.reshape(-1, target.size(-1)),
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


def evaluate(model, pairs):
    return [error_metrics(predict(pair, model), pair[2]) for pair in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--dense-query-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--ridge", type=parse_floats, default=[1e-3, 1e-2, 1e-1])
    parser.add_argument("--query-ranks", type=parse_ints, default=[4, 8, 16, 32, 64, 0])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native Hard-C4 probe")
    device = torch.device("cuda")

    train_pairs = [
        sampled_outputs(
            path,
            sample_count=args.sample_count,
            sample_seed=args.seed + index,
            dense_query_chunk=args.dense_query_chunk,
            device=device,
        )
        for index, path in enumerate(args.train)
    ]
    eval_pairs = [
        sampled_outputs(
            path,
            sample_count=args.sample_count,
            sample_seed=args.seed + 1000 + index,
            dense_query_chunk=args.dense_query_chunk,
            device=device,
        )
        for index, path in enumerate(args.eval)
    ]
    train_query, train_gma, train_dense = concatenate(train_pairs)
    heads, _, dim = train_gma.shape
    identity = torch.eye(dim, device=device).expand(heads, -1, -1)
    zero = torch.zeros_like(identity)

    base_model = {
        "gma_weight": identity,
        "query_weight": zero,
        "bias": torch.zeros(heads, dim, device=device),
    }
    base_eval = evaluate(base_model, eval_pairs)
    aggregate_base = aggregate_metrics(base_eval)
    candidates = {}

    for ridge in args.ridge:
        y_weight, y_bias = fit_centered_ridge(
            train_gma, train_dense, identity, ridge
        )
        y_model = {
            "gma_weight": y_weight,
            "query_weight": zero,
            "bias": y_bias,
        }
        candidates[f"y_only_ridge_{ridge}"] = y_model

        joint_features = torch.cat((train_gma, train_query), dim=-1)
        joint_prior = torch.cat((identity, zero), dim=1)
        joint_weight, _ = fit_centered_ridge(
            joint_features, train_dense, joint_prior, ridge
        )
        joint_gma_weight = joint_weight[:, :dim]
        joint_query_weight = joint_weight[:, dim:]

        residual_weight, _ = fit_centered_ridge(
            train_query, train_dense - train_gma, zero, ridge
        )
        for rank in args.query_ranks:
            rank_label = rank or dim
            joint_query_ranked = truncate_rank(joint_query_weight, rank)
            candidates[f"joint_ridge_{ridge}_qrank_{rank_label}"] = {
                "gma_weight": joint_gma_weight,
                "query_weight": joint_query_ranked,
                "bias": make_bias(
                    train_query,
                    train_gma,
                    train_dense,
                    joint_gma_weight,
                    joint_query_ranked,
                ),
            }
            residual_ranked = truncate_rank(residual_weight, rank)
            candidates[f"q_residual_ridge_{ridge}_qrank_{rank_label}"] = {
                "gma_weight": identity,
                "query_weight": residual_ranked,
                "bias": make_bias(
                    train_query,
                    train_gma,
                    train_dense,
                    identity,
                    residual_ranked,
                ),
            }

    result_candidates = {}
    for label, model in candidates.items():
        train_metrics = evaluate(model, train_pairs)
        eval_metrics = evaluate(model, eval_pairs)
        aggregate_eval = aggregate_metrics(eval_metrics)
        result_candidates[label] = {
            "train": aggregate_metrics(train_metrics),
            "eval": aggregate_eval,
            "eval_mse_ratio_vs_c4": aggregate_eval["mse"] / aggregate_base["mse"],
            "per_capture_eval": eval_metrics,
        }

    best_label = min(
        result_candidates,
        key=lambda label: result_candidates[label]["eval_mse_ratio_vs_c4"],
    )
    c4_ratio = (3 * 4 * 80 + 2 * 45 * 4) / (2 * 45 * 80)
    query_projection_ratio = dim / (2 * 21 * 45 * 80)
    result = {
        "method": "query-conditioned pure Hard-C4 calibration",
        "objective": "min ||Y_gma T_y + Q T_q + b - Y_dense||^2 + ridge-to-[I,0]",
        "train_samples_per_head": train_gma.size(1),
        "base_eval": aggregate_base,
        "per_capture_base_eval": base_eval,
        "attention_ratio_vs_dense": c4_ratio + query_projection_ratio,
        "attention_ratio_vs_svg_ear_density_0.3790645331": (
            (c4_ratio + query_projection_ratio) / 0.37906453310046345
        ),
        "best": {"label": best_label, **result_candidates[best_label]},
        "candidates": result_candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["best"], indent=2))


if __name__ == "__main__":
    main()
