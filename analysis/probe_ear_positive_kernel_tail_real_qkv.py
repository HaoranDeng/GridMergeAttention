"""Probe exact EAR blocks plus a positive-kernel GridMerge tail moment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from probe_ear_keyconditioned_gridmerge_real_qkv import (
    cluster_labels,
    load_qkv,
    metrics,
    mixed_measure_output,
    parse_floats,
    parse_ints,
)


def positive_feature_directions(rank: int, dim: int, seed: int, device):
    if rank % 2:
        raise ValueError("positive-kernel rank must be even")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(rank // 2, dim, generator=generator, dtype=torch.float64)
    directions = torch.stack((base, -base), dim=1).reshape(rank, dim)
    return directions.to(device=device)


def positive_features(tensor: torch.Tensor, directions: torch.Tensor):
    dim = tensor.size(-1)
    scaled = tensor.double() * dim ** -0.25
    exponent = torch.matmul(scaled, directions.T)
    exponent -= 0.5 * scaled.square().sum(dim=-1, keepdim=True)
    if not torch.isfinite(exponent).all() or exponent.max() > 700:
        raise FloatingPointError("positive softmax features overflow")
    return exponent.exp()


def clustered_feature_moments(features, values, labels, cluster_count, chunk=4096):
    rank = features.size(1)
    dim = values.size(1)
    feature_sum = torch.zeros(
        cluster_count, rank, device=features.device, dtype=torch.float64
    )
    value_moment = torch.zeros(
        cluster_count, rank, dim, device=features.device, dtype=torch.float64
    )
    for start in range(0, features.size(0), chunk):
        stop = min(start + chunk, features.size(0))
        local_labels = labels[start:stop]
        local_features = features[start:stop]
        local_values = values[start:stop].double()
        feature_sum.index_add_(0, local_labels, local_features)
        value_moment.index_add_(
            0,
            local_labels,
            local_features[:, :, None] * local_values[:, None, :],
        )
    return feature_sum, value_moment


def exact_sparse_output(query, keys, values, exact_token_mask):
    scale = query.size(-1) ** -0.5
    exact_keys = keys[exact_token_mask].double()
    exact_values = values[exact_token_mask].double()
    logits = torch.mv(exact_keys, query.double()) * scale
    probabilities = torch.softmax(logits, dim=0)
    return torch.mv(exact_values.T, probabilities), torch.logsumexp(logits, dim=0)


def mixed_positive_kernel_output(
    query,
    keys,
    values,
    exact_token_mask,
    tail_clusters,
    directions,
    cluster_feature_sum,
    cluster_value_moment,
    rank,
):
    scale = query.size(-1) ** -0.5
    exact_keys = keys[exact_token_mask].double()
    exact_values = values[exact_token_mask].double()
    exact_logits = torch.mv(exact_keys, query.double()) * scale

    query_feature = positive_features(query[None], directions[:rank])[0]
    tail_feature_sum = cluster_feature_sum[tail_clusters, :rank].sum(dim=0)
    tail_value_moment = cluster_value_moment[tail_clusters, :rank].sum(dim=0)
    tail_denominator = torch.dot(query_feature, tail_feature_sum) / rank
    tail_numerator = torch.matmul(query_feature, tail_value_moment) / rank

    exact_logz = torch.logsumexp(exact_logits, dim=0)
    tail_logz = tail_denominator.clamp_min(1e-300).log()
    shift = torch.maximum(exact_logz, tail_logz)
    exact_weight = (exact_logits - shift).exp()
    tail_scale = (-shift).exp()
    denominator = exact_weight.sum() + tail_denominator * tail_scale
    numerator = torch.mv(exact_values.T, exact_weight)
    numerator += tail_numerator * tail_scale
    return numerator / denominator.clamp_min(1e-300), shift + denominator.log()


def mixed_control_variate_output(
    query,
    keys,
    values,
    exact_token_mask,
    tail_clusters,
    key_centroids,
    value_centroids,
    key_sizes,
    directions,
    cluster_feature_sum,
    cluster_value_moment,
    representative_feature_sum,
    representative_value_moment,
    rank,
):
    scale = query.size(-1) ** -0.5
    exact_keys = keys[exact_token_mask].double()
    exact_values = values[exact_token_mask].double()
    exact_logits = torch.mv(exact_keys, query.double()) * scale
    tail_keys = key_centroids[tail_clusters].double()
    tail_values = value_centroids[tail_clusters].double()
    tail_sizes = key_sizes[tail_clusters].double()
    representative_logits = torch.mv(tail_keys, query.double()) * scale
    representative_logits += tail_sizes.clamp_min(1).log()

    query_feature = positive_features(query[None], directions[:rank])[0]
    residual_feature_sum = (
        cluster_feature_sum[tail_clusters, :rank]
        - representative_feature_sum[tail_clusters, :rank]
    ).sum(dim=0)
    residual_value_moment = (
        cluster_value_moment[tail_clusters, :rank]
        - representative_value_moment[tail_clusters, :rank]
    ).sum(dim=0)
    residual_denominator = torch.dot(query_feature, residual_feature_sum) / rank
    residual_numerator = torch.matmul(query_feature, residual_value_moment) / rank

    shift = torch.maximum(exact_logits.max(), representative_logits.max())
    exact_weight = (exact_logits - shift).exp()
    representative_weight = (representative_logits - shift).exp()
    residual_scale = (-shift).exp()
    denominator = exact_weight.sum() + representative_weight.sum()
    denominator += residual_denominator * residual_scale
    if not torch.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("control-variate denominator is not positive")
    numerator = torch.mv(exact_values.T, exact_weight)
    numerator += torch.mv(tail_values.T, representative_weight)
    numerator += residual_numerator * residual_scale
    return numerator / denominator, shift + denominator.log()


@torch.inference_mode()
def evaluate_head(
    query,
    key,
    value,
    query_centroids,
    key_centroids,
    value_centroids,
    query_sizes,
    key_sizes,
    dynamic_maps,
    sample_count,
    seed,
    ranks,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(query.size(0), generator=generator)[:sample_count]
    indices = indices.sort().values.to(query.device)
    query_clusters = torch.bucketize(
        indices, torch.cumsum(query_sizes.long(), dim=0), right=True
    )
    key_labels = cluster_labels(key_sizes)
    sampled_query = query.index_select(0, indices).float()
    dense_logits = torch.matmul(sampled_query, key.float().T) * query.size(-1) ** -0.5
    dense_probability = torch.softmax(dense_logits, dim=-1)
    dense_output = torch.matmul(dense_probability, value.float())
    dense_lse = torch.logsumexp(dense_logits, dim=-1)

    directions = positive_feature_directions(
        max(ranks), key.size(-1), seed, key.device
    )
    key_features = positive_features(key, directions)
    feature_sum, value_moment = clustered_feature_moments(
        key_features,
        value,
        key_labels,
        key_sizes.numel(),
    )
    representative_features = positive_features(key_centroids, directions)
    representative_feature_sum = representative_features * key_sizes.double()[:, None]
    representative_value_moment = (
        representative_feature_sum[:, :, None] * value_centroids.double()[:, None, :]
    )
    log_sizes = key_sizes.float().clamp_min(1).log()
    valid_clusters = torch.ones(
        key_sizes.numel(), device=key.device, dtype=torch.bool
    )
    cluster_ids = torch.arange(key_sizes.numel(), device=key.device)

    result = {}
    for top_p, dynamic_map in dynamic_maps.items():
        names = ["exact_sparse", "ear_mean_tail"]
        names += [f"positive_kernel_tail_r{rank}" for rank in ranks]
        names += [f"control_variate_tail_r{rank}" for rank in ranks]
        predictions = {name: [] for name in names}
        lse = {name: [] for name in names}
        for sample_idx, query_cluster in enumerate(query_clusters):
            exact_clusters = dynamic_map[int(query_cluster)]
            exact_token_mask = exact_clusters.index_select(0, key_labels)
            tail_clusters = ~exact_clusters
            output, output_lse = exact_sparse_output(
                sampled_query[sample_idx], key, value, exact_token_mask
            )
            predictions["exact_sparse"].append(output)
            lse["exact_sparse"].append(output_lse)

            output, output_lse = mixed_measure_output(
                sampled_query[sample_idx],
                key,
                value,
                key_labels,
                exact_clusters,
                key_centroids.float(),
                value_centroids.float(),
                log_sizes,
                cluster_ids,
                valid_clusters,
            )
            predictions["ear_mean_tail"].append(output)
            lse["ear_mean_tail"].append(output_lse)

            for rank in ranks:
                name = f"positive_kernel_tail_r{rank}"
                output, output_lse = mixed_positive_kernel_output(
                    sampled_query[sample_idx],
                    key,
                    value,
                    exact_token_mask,
                    tail_clusters,
                    directions,
                    feature_sum,
                    value_moment,
                    rank,
                )
                predictions[name].append(output)
                lse[name].append(output_lse)
                name = f"control_variate_tail_r{rank}"
                output, output_lse = mixed_control_variate_output(
                    sampled_query[sample_idx],
                    key,
                    value,
                    exact_token_mask,
                    tail_clusters,
                    key_centroids,
                    value_centroids,
                    key_sizes,
                    directions,
                    feature_sum,
                    value_moment,
                    representative_feature_sum,
                    representative_value_moment,
                    rank,
                )
                predictions[name].append(output)
                lse[name].append(output_lse)
        result[top_p] = {
            name: metrics(
                torch.stack(predictions[name]).float(),
                dense_output,
                torch.stack(lse[name]).float(),
                dense_lse,
            )
            for name in names
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=parse_ints, default=[0, 10, 20, 30])
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--ranks", type=parse_ints, default=[8, 16, 32])
    parser.add_argument("--top-ps", type=parse_floats, default=[0.9, 0.82, 0.75])
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--svg-root", default="/home/denghaoran/workspace/Sparse-VideoGen"
    )
    args = parser.parse_args()
    if any(rank <= 0 or rank % 2 for rank in args.ranks):
        parser.error("all positive-kernel ranks must be positive and even")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["SPARSE_VIDEO_GEN_ROOT"] = args.svg_root
    device = torch.device("cuda")
    query, key, value = load_qkv(args.qkv, args.heads, device)

    from attention.svg_family_wan import _configure

    modules = _configure("ear", (21, 45, 80), query)
    from svg.kmeans_utils import identify_dynamic_map_estimated

    processor = modules["ear"](layer_idx=0)
    (
        query_permuted,
        key_permuted,
        value_permuted,
        _,
        _,
        query_sizes,
        key_sizes,
        _,
        query_centroids,
        key_centroids,
        value_centroids,
    ) = processor.semantic_aware_permutation(query, key, value)
    dynamic_maps = {
        top_p: identify_dynamic_map_estimated(
            query_permuted,
            key_permuted,
            value_permuted,
            query_sizes,
            key_sizes,
            query_centroids,
            key_centroids,
            value_centroids,
            top_p=top_p,
            gamma=1,
            min_kc_ratio=0.05,
        ).contiguous()
        for top_p in args.top_ps
    }
    area = query_sizes[..., :, None].float() * key_sizes[..., None, :].float()
    density = {
        top_p: (
            (dynamic_map.float() * area).sum(dim=(-2, -1)) / query.size(2) ** 2
        ).squeeze(0)
        for top_p, dynamic_map in dynamic_maps.items()
    }
    tail_representative_density = {
        top_p: (
            ((~dynamic_map).float() * query_sizes[..., :, None].float()).sum(
                dim=(-2, -1)
            )
            / query.size(2) ** 2
        ).squeeze(0)
        for top_p, dynamic_map in dynamic_maps.items()
    }

    rows = {top_p: [] for top_p in args.top_ps}
    for local_head, original_head in enumerate(args.heads):
        head_result = evaluate_head(
            query_permuted[0, local_head],
            key_permuted[0, local_head],
            value_permuted[0, local_head],
            query_centroids[0, local_head],
            key_centroids[0, local_head],
            value_centroids[0, local_head],
            query_sizes[0, local_head],
            key_sizes[0, local_head],
            {
                top_p: dynamic_map[0, local_head]
                for top_p, dynamic_map in dynamic_maps.items()
            },
            args.sample_count,
            args.seed + original_head,
            args.ranks,
        )
        for top_p, row in head_result.items():
            row.update(
                {
                    "head": original_head,
                    "exact_density": float(density[top_p][local_head]),
                    "tail_representative_density": float(
                        tail_representative_density[top_p][local_head]
                    ),
                }
            )
            rows[top_p].append(row)

    methods = ["exact_sparse", "ear_mean_tail"]
    methods += [f"positive_kernel_tail_r{rank}" for rank in args.ranks]
    methods += [f"control_variate_tail_r{rank}" for rank in args.ranks]
    token_count = query.size(2)
    sweeps = []
    for top_p in args.top_ps:
        mean_density = float(density[top_p].mean())
        mean_tail_representative_density = float(
            tail_representative_density[top_p].mean()
        )
        summary = {
            method: {
                metric: float(
                    sum(row[method][metric] for row in rows[top_p])
                    / len(rows[top_p])
                )
                for metric in ("mse", "relative_l2", "cosine", "logsumexp_mae")
            }
            for method in methods
        }
        theoretical_density = {
            f"positive_kernel_tail_r{rank}": mean_density
            + 2.0 * rank / token_count
            for rank in args.ranks
        }
        theoretical_density.update(
            {
                f"control_variate_tail_r{rank}": mean_density
                + mean_tail_representative_density
                + 2.0 * rank / token_count
                for rank in args.ranks
            }
        )
        sweeps.append(
            {
                "top_p": top_p,
                "mean_exact_density": mean_density,
                "mean_tail_representative_density": (
                    mean_tail_representative_density
                ),
                "theoretical_density_upper_bound": theoretical_density,
                "summary": summary,
                "per_head": rows[top_p],
            }
        )

    result = {
        "method": "EAR exact blocks + positive-kernel GridMerge tail moment",
        "formula": (
            "tail denominator phi(q)^T sum_j phi(k_j); tail numerator "
            "phi(q)^T sum_j phi(k_j) v_j^T"
        ),
        "qkv": str(args.qkv),
        "heads": args.heads,
        "sample_count": args.sample_count,
        "ranks": args.ranks,
        "softmax_feature_scale": "q,k multiplied by d^(-1/4)",
        "theoretical_feature_overhead": "2r/N of dense QK+AV",
        "sweeps": sweeps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
