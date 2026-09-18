"""Probe a deterministic Nystrom control variate for the GMA tail."""

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
from probe_ear_positive_kernel_tail_real_qkv import (
    clustered_feature_moments,
    exact_sparse_output,
)


def weighted_farthest_landmarks(candidates, weights, rank):
    if rank > candidates.size(0):
        raise ValueError("rank exceeds the number of landmark candidates")
    points = candidates.double()
    normalized_weight = weights.double() / weights.double().mean().clamp_min(1e-12)
    chosen = [int(weights.argmax())]
    minimum_distance = (points - points[chosen[0]]).square().sum(dim=-1)
    for _ in range(1, rank):
        score = minimum_distance * normalized_weight.sqrt()
        score[torch.tensor(chosen, device=score.device)] = -1
        index = int(score.argmax())
        chosen.append(index)
        distance = (points - points[index]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    return candidates.index_select(
        0, torch.tensor(chosen, device=candidates.device)
    ).double()


def softmax_kernel_features(tensor, landmarks):
    scale = tensor.size(-1) ** -0.5
    logits = torch.matmul(tensor.double(), landmarks.T) * scale
    return logits.exp()


def nystrom_control_variate_output(
    query,
    keys,
    values,
    exact_token_mask,
    tail_clusters,
    key_centroids,
    value_centroids,
    key_sizes,
    landmarks,
    cluster_kernel_sum,
    cluster_value_moment,
    representative_kernel_sum,
    representative_value_moment,
    ridge,
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

    residual_kernel_sum = (
        cluster_kernel_sum[tail_clusters]
        - representative_kernel_sum[tail_clusters]
    ).sum(dim=0)
    residual_value_moment = (
        cluster_value_moment[tail_clusters]
        - representative_value_moment[tail_clusters]
    ).sum(dim=0)
    landmark_gram = softmax_kernel_features(landmarks, landmarks)
    regularizer = ridge * landmark_gram.diagonal().mean()
    landmark_gram = landmark_gram + regularizer * torch.eye(
        landmarks.size(0), device=landmarks.device, dtype=torch.float64
    )
    residual_coefficients = torch.linalg.solve(
        landmark_gram,
        torch.cat((residual_kernel_sum[:, None], residual_value_moment), dim=1),
    )
    query_kernel = softmax_kernel_features(query[None], landmarks)[0]
    residual = torch.matmul(query_kernel, residual_coefficients)

    shift = torch.maximum(exact_logits.max(), representative_logits.max())
    exact_weight = (exact_logits - shift).exp()
    representative_weight = (representative_logits - shift).exp()
    residual_scale = (-shift).exp()
    denominator = exact_weight.sum() + representative_weight.sum()
    denominator += residual[0] * residual_scale
    positive = bool(torch.isfinite(denominator) and denominator > 0)
    if not positive:
        denominator = denominator.new_tensor(1e-300)
    numerator = torch.mv(exact_values.T, exact_weight)
    numerator += torch.mv(tail_values.T, representative_weight)
    numerator += residual[1:] * residual_scale
    return numerator / denominator, shift + denominator.log(), positive


def nystrom_denominator_output(
    query,
    keys,
    values,
    exact_token_mask,
    tail_clusters,
    key_centroids,
    value_centroids,
    key_sizes,
    landmarks,
    cluster_kernel_sum,
    representative_kernel_sum,
    ridge,
):
    """Correct only the tail partition function, keeping GMA's value moment."""
    scale = query.size(-1) ** -0.5
    exact_keys = keys[exact_token_mask].double()
    exact_values = values[exact_token_mask].double()
    exact_logits = torch.mv(exact_keys, query.double()) * scale
    tail_keys = key_centroids[tail_clusters].double()
    tail_values = value_centroids[tail_clusters].double()
    tail_sizes = key_sizes[tail_clusters].double()
    representative_logits = torch.mv(tail_keys, query.double()) * scale
    representative_logits += tail_sizes.clamp_min(1).log()

    residual_kernel_sum = (
        cluster_kernel_sum[tail_clusters]
        - representative_kernel_sum[tail_clusters]
    ).sum(dim=0)
    landmark_gram = softmax_kernel_features(landmarks, landmarks)
    regularizer = ridge * landmark_gram.diagonal().mean()
    landmark_gram = landmark_gram + regularizer * torch.eye(
        landmarks.size(0), device=landmarks.device, dtype=torch.float64
    )
    residual_coefficients = torch.linalg.solve(
        landmark_gram, residual_kernel_sum[:, None]
    )[:, 0]
    query_kernel = softmax_kernel_features(query[None], landmarks)[0]
    denominator_residual = torch.dot(query_kernel, residual_coefficients)

    shift = torch.maximum(exact_logits.max(), representative_logits.max())
    exact_weight = (exact_logits - shift).exp()
    representative_weight = (representative_logits - shift).exp()
    denominator = exact_weight.sum() + representative_weight.sum()
    denominator += denominator_residual * (-shift).exp()
    positive = bool(torch.isfinite(denominator) and denominator > 0)
    if not positive:
        denominator = denominator.new_tensor(1e-300)
    numerator = torch.mv(exact_values.T, exact_weight)
    numerator += torch.mv(tail_values.T, representative_weight)
    return numerator / denominator, shift + denominator.log(), positive


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
    ridges,
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

    max_landmarks = weighted_farthest_landmarks(
        query_centroids, query_sizes, max(ranks)
    )
    key_kernel = softmax_kernel_features(key, max_landmarks)
    kernel_sum, value_moment = clustered_feature_moments(
        key_kernel, value, key_labels, key_sizes.numel()
    )
    representative_kernel = softmax_kernel_features(
        key_centroids, max_landmarks
    )
    representative_kernel_sum = representative_kernel * key_sizes.double()[:, None]
    representative_value_moment = (
        representative_kernel_sum[:, :, None]
        * value_centroids.double()[:, None, :]
    )
    log_sizes = key_sizes.float().clamp_min(1).log()
    valid_clusters = torch.ones(
        key_sizes.numel(), device=key.device, dtype=torch.bool
    )
    cluster_ids = torch.arange(key_sizes.numel(), device=key.device)

    result = {}
    for top_p, dynamic_map in dynamic_maps.items():
        names = ["exact_sparse", "ear_mean_tail"]
        names += [
            f"nystrom_cv_r{rank}_ridge{ridge:g}"
            for rank in ranks
            for ridge in ridges
        ]
        names += [
            f"nystrom_z_r{rank}_ridge{ridge:g}"
            for rank in ranks
            for ridge in ridges
        ]
        predictions = {name: [] for name in names}
        lse = {name: [] for name in names}
        valid = {name: 0 for name in names}
        for sample_idx, query_cluster in enumerate(query_clusters):
            exact_clusters = dynamic_map[int(query_cluster)]
            exact_token_mask = exact_clusters.index_select(0, key_labels)
            tail_clusters = ~exact_clusters
            output, output_lse = exact_sparse_output(
                sampled_query[sample_idx], key, value, exact_token_mask
            )
            predictions["exact_sparse"].append(output)
            lse["exact_sparse"].append(output_lse)
            valid["exact_sparse"] += 1

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
            valid["ear_mean_tail"] += 1

            for rank in ranks:
                for ridge in ridges:
                    name = f"nystrom_cv_r{rank}_ridge{ridge:g}"
                    output, output_lse, positive = nystrom_control_variate_output(
                        sampled_query[sample_idx],
                        key,
                        value,
                        exact_token_mask,
                        tail_clusters,
                        key_centroids,
                        value_centroids,
                        key_sizes,
                        max_landmarks[:rank],
                        kernel_sum[:, :rank],
                        value_moment[:, :rank],
                        representative_kernel_sum[:, :rank],
                        representative_value_moment[:, :rank],
                        ridge,
                    )
                    predictions[name].append(output)
                    lse[name].append(output_lse)
                    valid[name] += int(positive)

                    z_name = f"nystrom_z_r{rank}_ridge{ridge:g}"
                    output, output_lse, positive = nystrom_denominator_output(
                        sampled_query[sample_idx],
                        key,
                        value,
                        exact_token_mask,
                        tail_clusters,
                        key_centroids,
                        value_centroids,
                        key_sizes,
                        max_landmarks[:rank],
                        kernel_sum[:, :rank],
                        representative_kernel_sum[:, :rank],
                        ridge,
                    )
                    predictions[z_name].append(output)
                    lse[z_name].append(output_lse)
                    valid[z_name] += int(positive)
        result[top_p] = {}
        for name in names:
            summary = metrics(
                torch.stack(predictions[name]).float(),
                dense_output,
                torch.stack(lse[name]).float(),
                dense_lse,
            )
            summary["positive_denominators"] = valid[name]
            summary["queries"] = sample_count
            result[top_p][name] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=parse_ints, default=[0, 10, 20, 30])
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--ranks", type=parse_ints, default=[16, 32, 64])
    parser.add_argument("--ridges", type=parse_floats, default=[0.001, 0.01, 0.1])
    parser.add_argument("--top-ps", type=parse_floats, default=[0.9, 0.82, 0.75])
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--svg-root", default="/home/denghaoran/workspace/Sparse-VideoGen"
    )
    args = parser.parse_args()
    if any(rank <= 0 for rank in args.ranks):
        parser.error("all Nystrom ranks must be positive")
    if any(ridge <= 0 for ridge in args.ridges):
        parser.error("all ridge values must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["SPARSE_VIDEO_GEN_ROOT"] = args.svg_root
    query, key, value = load_qkv(args.qkv, args.heads, torch.device("cuda"))

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
            args.ridges,
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
    methods += [
        f"nystrom_cv_r{rank}_ridge{ridge:g}"
        for rank in args.ranks
        for ridge in args.ridges
    ]
    methods += [
        f"nystrom_z_r{rank}_ridge{ridge:g}"
        for rank in args.ranks
        for ridge in args.ridges
    ]
    sweeps = []
    for top_p in args.top_ps:
        mean_exact_density = float(density[top_p].mean())
        mean_tail_density = float(tail_representative_density[top_p].mean())
        summary = {}
        for method in methods:
            summary[method] = {
                metric: float(
                    sum(row[method][metric] for row in rows[top_p])
                    / len(rows[top_p])
                )
                for metric in (
                    "mse",
                    "relative_l2",
                    "cosine",
                    "logsumexp_mae",
                    "positive_denominators",
                )
            }
        sweeps.append(
            {
                "top_p": top_p,
                "mean_exact_density": mean_exact_density,
                "mean_tail_representative_density": mean_tail_density,
                "theoretical_density_upper_bound": {
                    f"nystrom_cv_r{rank}": mean_exact_density
                    + mean_tail_density
                    + 2.0 * rank / query.size(2)
                    for rank in args.ranks
                },
                "summary": summary,
                "per_head": rows[top_p],
            }
        )

    result = {
        "method": "EAR exact blocks + Nystrom control-variate GMA tail",
        "qkv": str(args.qkv),
        "heads": args.heads,
        "sample_count": args.sample_count,
        "ranks": args.ranks,
        "ridges": args.ridges,
        "landmarks": "weighted farthest-first query centroids",
        "sweeps": sweeps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
