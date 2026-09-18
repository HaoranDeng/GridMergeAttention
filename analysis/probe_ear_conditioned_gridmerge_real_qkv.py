"""Probe an exact-head / conditionally merged-tail attention measure.

The EAR block selector keeps important key clusters exact.  For every pruned
key cluster, this probe replaces EAR's uniform K/V centroid with a
query-cluster-conditioned GridMerge representative.  The selected exact keys
and merged tail are disjoint and share one softmax normalization, so no key
mass is duplicated.

This is an offline real-QKV screen.  It intentionally samples only a small
number of queries and does not provide a production kernel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.replace(",", " ").split()]


def load_qkv(path: Path, heads: list[int], device: torch.device):
    with np.load(path, allow_pickle=False) as capture:
        tensors = []
        for name in ("q", "k", "v"):
            tensor = torch.from_numpy(np.asarray(capture[name])).to(
                device=device, dtype=torch.float16
            )
            tensors.append(tensor.index_select(1, torch.tensor(heads, device=device)))
    return tuple(tensors)


def cluster_labels(sizes: torch.Tensor):
    count = sizes.numel()
    return torch.repeat_interleave(
        torch.arange(count, device=sizes.device), sizes.long()
    )


def identify_mass_map(
    query_centroids: torch.Tensor,
    key_centroids: torch.Tensor,
    key_sizes: torch.Tensor,
    top_p: float,
):
    """Keep the smallest centroid-ranked prefix reaching attention mass top_p."""

    scale = query_centroids.size(-1) ** -0.5
    logits = torch.einsum(
        "bhqd,bhkd->bhqk", query_centroids, key_centroids
    ) * scale
    logits = logits + key_sizes.float().clamp_min(1).log().unsqueeze(-2)
    probabilities = torch.softmax(logits.float(), dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, dim=-1, descending=True
    )
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    sorted_keep = cumulative <= top_p
    sorted_keep = F.pad(sorted_keep[..., :-1], (1, 0), value=True)
    dynamic_map = torch.zeros_like(sorted_keep)
    dynamic_map.scatter_(-1, sorted_indices, sorted_keep)
    return dynamic_map


def weighted_kcentroid_seeds(
    centroids: torch.Tensor,
    cluster_sizes: torch.Tensor,
    seed_count: int,
    iterations: int = 3,
):
    """Coarsen query centroids with weighted farthest-first Lloyd updates."""

    centroid_count = centroids.size(0)
    if seed_count <= 0 or seed_count >= centroid_count:
        return centroids, torch.arange(centroid_count, device=centroids.device)

    points = centroids.float()
    importance = cluster_sizes.float().clamp_min(1)
    selected = [int(torch.argmax(importance).item())]
    min_distance = torch.full(
        (centroid_count,), torch.inf, device=centroids.device, dtype=torch.float32
    )
    for _ in range(1, seed_count):
        latest = points[selected[-1]]
        distance = (points - latest).square().sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        score = min_distance * importance.sqrt()
        score[selected] = -1
        selected.append(int(torch.argmax(score).item()))
    seeds = points.index_select(
        0, torch.tensor(selected, device=centroids.device)
    )

    labels = None
    for _ in range(iterations):
        distance = torch.cdist(points, seeds).square()
        labels = distance.argmin(dim=-1)
        denominator = torch.zeros(seed_count, device=centroids.device)
        denominator.index_add_(0, labels, importance)
        updated = torch.zeros_like(seeds)
        updated.index_add_(0, labels, importance[:, None] * points)
        nonempty = denominator > 0
        seeds[nonempty] = updated[nonempty] / denominator[nonempty, None]
    return seeds.to(centroids.dtype), labels


@torch.inference_mode()
def conditioned_representatives(
    seeds: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    labels: torch.Tensor,
    cluster_count: int,
    *,
    batch_size: int,
):
    """Return seed-conditioned K/V/entropy for hard key clusters."""

    scale = keys.size(-1) ** -0.5
    result = {}
    for start in range(0, seeds.size(0), batch_size):
        seed_batch = seeds[start : start + batch_size].float()
        count = seed_batch.size(0)
        scores = torch.matmul(seed_batch, keys.float().T) * scale
        offsets = torch.arange(count, device=keys.device)[:, None] * cluster_count
        group = labels[None, :] + offsets
        flat_group = group.reshape(-1)

        maxima = torch.full(
            (count * cluster_count,),
            -torch.inf,
            device=keys.device,
            dtype=torch.float32,
        )
        maxima.scatter_reduce_(
            0, flat_group, scores.reshape(-1), reduce="amax", include_self=True
        )
        local_group = labels[None, :].expand(count, -1)
        shifted = scores - maxima.view(count, cluster_count).gather(
            1, local_group
        )
        exponential = shifted.exp()
        denominator = torch.zeros_like(maxima)
        denominator.scatter_add_(0, flat_group, exponential.reshape(-1))
        weights = exponential / denominator.view(count, cluster_count).gather(
            1, local_group
        )

        merged_keys = torch.zeros(
            count * cluster_count,
            keys.size(-1),
            device=keys.device,
            dtype=torch.float32,
        )
        merged_values = torch.zeros_like(merged_keys)
        merged_keys.index_add_(
            0,
            flat_group,
            (weights[..., None] * keys.float()[None, :, :]).reshape(
                -1, keys.size(-1)
            ),
        )
        merged_values.index_add_(
            0,
            flat_group,
            (weights[..., None] * values.float()[None, :, :]).reshape(
                -1, values.size(-1)
            ),
        )
        expected_score = torch.zeros_like(maxima)
        expected_score.scatter_add_(
            0, flat_group, (weights * scores).reshape(-1)
        )
        entropy = (
            maxima
            + denominator.clamp_min(torch.finfo(torch.float32).tiny).log()
            - expected_score
        )

        merged_keys = merged_keys.view(count, cluster_count, -1)
        merged_values = merged_values.view(count, cluster_count, -1)
        entropy = entropy.view(count, cluster_count)
        for offset in range(count):
            result[start + offset] = (
                merged_keys[offset],
                merged_values[offset],
                entropy[offset],
            )
    return result


def mixed_measure_output(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    key_labels: torch.Tensor,
    exact_clusters: torch.Tensor,
    tail_keys: torch.Tensor,
    tail_values: torch.Tensor,
    tail_bias: torch.Tensor,
):
    exact_key_mask = exact_clusters.index_select(0, key_labels)
    exact_keys = keys[exact_key_mask].float()
    exact_values = values[exact_key_mask].float()
    tail_cluster_mask = ~exact_clusters
    tail_keys = tail_keys[tail_cluster_mask]
    tail_values = tail_values[tail_cluster_mask]
    tail_bias = tail_bias[tail_cluster_mask]
    scale = query.size(-1) ** -0.5
    exact_logits = torch.mv(exact_keys, query.float()) * scale
    tail_logits = torch.mv(tail_keys, query.float()) * scale + tail_bias
    logits = torch.cat((exact_logits, tail_logits))
    merged_values = torch.cat((exact_values, tail_values), dim=0)
    probability = torch.softmax(logits, dim=0)
    return torch.mv(merged_values.T, probability), torch.logsumexp(logits, dim=0)


def metrics(prediction, target, predicted_lse, target_lse):
    error = prediction - target
    return {
        "mse": float(error.square().mean().item()),
        "relative_l2": float(
            (error.norm() / target.norm().clamp_min(1e-12)).item()
        ),
        "cosine": float(F.cosine_similarity(prediction, target, dim=-1).mean().item()),
        "logsumexp_mae": float((predicted_lse - target_lse).abs().mean().item()),
    }


@torch.inference_mode()
def evaluate_head(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_centroids: torch.Tensor,
    key_centroids: torch.Tensor,
    value_centroids: torch.Tensor,
    query_sizes: torch.Tensor,
    key_sizes: torch.Tensor,
    dynamic_maps: dict[float, torch.Tensor],
    tail_seeds: torch.Tensor,
    query_to_tail_seed: torch.Tensor,
    *,
    sample_count: int,
    seed: int,
    merge_batch_size: int,
):
    sequence = query.size(0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(sequence, generator=generator)[:sample_count].sort().values.to(
        query.device
    )
    query_ends = torch.cumsum(query_sizes.long(), dim=0)
    query_clusters = torch.bucketize(indices, query_ends, right=True)
    key_labels = cluster_labels(key_sizes)

    sampled_query = query.index_select(0, indices).float()
    dense_logits = torch.matmul(sampled_query, key.float().T) * query.size(-1) ** -0.5
    dense_probability = torch.softmax(dense_logits, dim=-1)
    dense_output = torch.matmul(dense_probability, value.float())
    dense_lse = torch.logsumexp(dense_logits, dim=-1)

    sample_tail_seeds = query_to_tail_seed.index_select(0, query_clusters)
    unique_clusters, inverse = torch.unique(
        sample_tail_seeds, sorted=True, return_inverse=True
    )
    conditioned = conditioned_representatives(
        tail_seeds.index_select(0, unique_clusters),
        key,
        value,
        key_labels,
        key_sizes.numel(),
        batch_size=merge_batch_size,
    )

    predictions = {
        top_p: {
            "ear_mean_tail": [],
            "ear_mean_tail_lse": [],
            "conditioned_gridmerge_tail": [],
            "conditioned_gridmerge_tail_lse": [],
        }
        for top_p in dynamic_maps
    }
    log_sizes = key_sizes.float().clamp_min(1).log()
    for sample_idx in range(indices.numel()):
        query_cluster = int(query_clusters[sample_idx])
        tail_keys, tail_values, tail_entropy = conditioned[int(inverse[sample_idx])]
        for top_p, dynamic_map in dynamic_maps.items():
            exact_clusters = dynamic_map[query_cluster]
            mean_output, mean_logsumexp = mixed_measure_output(
                sampled_query[sample_idx],
                key,
                value,
                key_labels,
                exact_clusters,
                key_centroids.float(),
                value_centroids.float(),
                log_sizes,
            )
            conditioned_output, conditioned_logsumexp = mixed_measure_output(
                sampled_query[sample_idx],
                key,
                value,
                key_labels,
                exact_clusters,
                tail_keys,
                tail_values,
                tail_entropy,
            )
            row = predictions[top_p]
            row["ear_mean_tail"].append(mean_output)
            row["ear_mean_tail_lse"].append(mean_logsumexp)
            row["conditioned_gridmerge_tail"].append(conditioned_output)
            row["conditioned_gridmerge_tail_lse"].append(
                conditioned_logsumexp
            )

    results = {}
    for top_p, row in predictions.items():
        mean_predictions = torch.stack(row["ear_mean_tail"])
        conditioned_predictions = torch.stack(row["conditioned_gridmerge_tail"])
        mean_lse = torch.stack(row["ear_mean_tail_lse"])
        conditioned_lse = torch.stack(row["conditioned_gridmerge_tail_lse"])
        results[top_p] = {
            "ear_mean_tail": metrics(
                mean_predictions, dense_output, mean_lse, dense_lse
            ),
            "conditioned_gridmerge_tail": metrics(
                conditioned_predictions,
                dense_output,
                conditioned_lse,
                dense_lse,
            ),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=parse_ints, default=[0])
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.6669)
    parser.add_argument("--top-ps", type=parse_floats)
    parser.add_argument("--min-kc-ratio", type=float, default=0.05)
    parser.add_argument(
        "--router", choices=("ear_efficiency", "mass"), default="ear_efficiency"
    )
    parser.add_argument("--tail-seeds", type=int, default=0)
    parser.add_argument("--merge-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--svg-root",
        default="/home/denghaoran/workspace/Sparse-VideoGen",
    )
    args = parser.parse_args()
    top_ps = args.top_ps or [args.top_p]
    if not top_ps:
        raise ValueError("At least one top_p value is required")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["SPARSE_VIDEO_GEN_ROOT"] = args.svg_root
    os.environ["SVG_TOP_P"] = str(max(top_ps))
    device = torch.device("cuda")
    query, key, value = load_qkv(args.qkv, args.heads, device)

    from attention.svg_family_wan import _configure

    modules = _configure("ear", (21, 45, 80), query)
    processor = modules["ear"](layer_idx=0)
    processor.top_p_kmeans = max(top_ps)
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

    from svg.kmeans_utils import identify_dynamic_map_estimated

    if args.router == "mass":
        dynamic_maps = {
            top_p: identify_mass_map(
                query_centroids, key_centroids, key_sizes, top_p
            ).contiguous()
            for top_p in top_ps
        }
    else:
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
                min_kc_ratio=args.min_kc_ratio,
            ).contiguous()
            for top_p in top_ps
        }
    area = query_sizes[..., :, None].float() * key_sizes[..., None, :].float()
    density_per_head = {
        top_p: (
            (dynamic_map.float() * area).sum(dim=(-2, -1))
            / query.size(2) ** 2
        ).squeeze(0)
        for top_p, dynamic_map in dynamic_maps.items()
    }

    rows = {top_p: [] for top_p in top_ps}
    for local_head, original_head in enumerate(args.heads):
        tail_seeds, query_to_tail_seed = weighted_kcentroid_seeds(
            query_centroids[0, local_head],
            query_sizes[0, local_head],
            args.tail_seeds,
        )
        head_results = evaluate_head(
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
            tail_seeds,
            query_to_tail_seed,
            sample_count=args.sample_count,
            seed=args.seed + original_head,
            merge_batch_size=args.merge_batch_size,
        )
        for top_p, row in head_results.items():
            row.update(
                {
                    "head": original_head,
                    "exact_density": float(
                        density_per_head[top_p][local_head].item()
                    ),
                }
            )
            rows[top_p].append(row)

    sweeps = []
    for top_p in top_ps:
        top_p_rows = rows[top_p]

        def average(method, metric):
            return float(
                sum(row[method][metric] for row in top_p_rows)
                / len(top_p_rows)
            )

        summary = {}
        for method in ("ear_mean_tail", "conditioned_gridmerge_tail"):
            summary[method] = {
                metric: average(method, metric)
                for metric in ("mse", "relative_l2", "cosine", "logsumexp_mae")
            }
        summary["conditioned_mse_ratio_vs_ear"] = (
            summary["conditioned_gridmerge_tail"]["mse"]
            / summary["ear_mean_tail"]["mse"]
        )
        sweeps.append(
            {
                "top_p": top_p,
                "mean_exact_density": float(density_per_head[top_p].mean().item()),
                "summary": summary,
                "per_head": top_p_rows,
            }
        )

    result = {
        "method": "EAR exact blocks + query-cluster-conditioned GridMerge tail",
        "qkv": str(args.qkv),
        "heads": args.heads,
        "sample_count": args.sample_count,
        "min_kc_ratio": args.min_kc_ratio,
        "router": args.router,
        "tail_seed_count": (
            args.tail_seeds if 0 < args.tail_seeds < query_centroids.size(2)
            else query_centroids.size(2)
        ),
    }
    if len(sweeps) == 1:
        result.update(sweeps[0])
    else:
        result["sweeps"] = sweeps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
