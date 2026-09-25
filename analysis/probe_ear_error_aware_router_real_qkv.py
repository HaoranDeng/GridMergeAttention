"""Screen error-aware exact-block routing at a matched SVG-EAR budget."""

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
    parse_ints,
)


def _cluster_scalar_mean(values, labels, sizes):
    result = torch.zeros(
        sizes.numel(), device=values.device, dtype=torch.float32
    )
    result.index_add_(0, labels, values.float())
    return result / sizes.float().clamp_min(1)


def cluster_risk_statistics(
    query_centroids,
    key,
    value,
    key_centroids,
    value_centroids,
    key_sizes,
    seed,
):
    labels = cluster_labels(key_sizes)
    key_for_token = key_centroids.index_select(0, labels).float()
    value_for_token = value_centroids.index_select(0, labels).float()
    key_variance = _cluster_scalar_mean(
        (key.float() - key_for_token).square().sum(dim=-1), labels, key_sizes
    )
    value_variance = _cluster_scalar_mean(
        (value.float() - value_for_token).square().sum(dim=-1), labels, key_sizes
    )

    scale = key.size(-1) ** -0.5
    centroid_logits = torch.matmul(
        query_centroids.float(), key_centroids.float().T
    ) * scale
    mass = torch.softmax(
        centroid_logits + key_sizes.float().clamp_min(1).log()[None, :], dim=-1
    )

    token_logits = torch.matmul(query_centroids.float(), key.float().T) * scale
    centered_logits = token_logits - centroid_logits.index_select(1, labels)
    directional_variance = torch.zeros(
        query_centroids.size(0),
        key_sizes.numel(),
        device=key.device,
        dtype=torch.float32,
    )
    directional_variance.scatter_add_(
        1,
        labels[None, :].expand(query_centroids.size(0), -1),
        centered_logits.square(),
    )
    directional_variance /= key_sizes.float().clamp_min(1)[None, :]

    tangent_bound = value_variance[None, :] * directional_variance
    key_size = key_sizes.float().clamp_min(1)[None, :]
    scores = {
        "centroid_mass": mass,
        "mass_logit_variance": mass * directional_variance,
        "mass_tangent_bound": mass * tangent_bound.sqrt(),
        "squared_mass_tangent_bound": mass.square() * tangent_bound,
        "squared_mass_tangent_bound_per_token": (
            mass.square() * tangent_bound / key_size
        ),
        "mass_tangent_bound_per_token": mass * tangent_bound.sqrt() / key_size,
    }

    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        value.size(-1), 8, generator=generator, dtype=torch.float32
    )
    projection = torch.linalg.qr(projection, mode="reduced").Q.to(value.device)
    value_sketch = torch.matmul(value.float() - value_for_token, projection)
    cross_response = torch.zeros(
        query_centroids.size(0),
        key_sizes.numel(),
        projection.size(1),
        device=value.device,
        dtype=torch.float32,
    )
    cross_response.scatter_add_(
        1,
        labels[None, :, None].expand(
            query_centroids.size(0), -1, projection.size(1)
        ),
        centered_logits[:, :, None] * value_sketch[None, :, :],
    )
    cross_response /= key_sizes.float().clamp_min(1)[None, :, None]
    for rank in (2, 4, 8):
        response_energy = cross_response[..., :rank].square().sum(dim=-1)
        scores[f"signed_cross_r{rank}"] = mass.square() * response_energy
        scores[f"signed_cross_r{rank}_per_token"] = (
            mass.square() * response_energy / key_size
        )
    return scores, {
        "key_variance": key_variance,
        "value_variance": value_variance,
        "directional_variance": directional_variance,
    }


def matched_cardinality_map(score, reference_map):
    selected = torch.zeros_like(reference_map)
    counts = reference_map.sum(dim=-1).long()
    for query_cluster, count in enumerate(counts.tolist()):
        if count:
            indices = score[query_cluster].topk(count).indices
            selected[query_cluster, indices] = True
    return selected


def swap_refined_map(score, reference_map, swaps):
    selected = reference_map.clone()
    for query_cluster in range(reference_map.size(0)):
        exact_indices = reference_map[query_cluster].nonzero().flatten()
        tail_indices = (~reference_map[query_cluster]).nonzero().flatten()
        count = min(swaps, exact_indices.numel(), tail_indices.numel())
        if not count:
            continue
        remove = exact_indices[
            score[query_cluster, exact_indices].topk(count, largest=False).indices
        ]
        add = tail_indices[
            score[query_cluster, tail_indices].topk(count).indices
        ]
        selected[query_cluster, remove] = False
        selected[query_cluster, add] = True
    return selected


def map_density(dynamic_map, query_sizes, key_sizes, sequence):
    area = query_sizes[:, None].float() * key_sizes[None, :].float()
    exact = (dynamic_map.float() * area).sum() / sequence**2
    tail_representatives = (
        (~dynamic_map).float() * query_sizes[:, None].float()
    ).sum() / sequence**2
    return float(exact), float(tail_representatives)


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
    reference_maps,
    budget_top_p,
    sample_count,
    seed,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(query.size(0), generator=generator)[:sample_count]
    indices = indices.sort().values.to(query.device)
    query_clusters = torch.bucketize(
        indices, torch.cumsum(query_sizes.long(), dim=0), right=True
    )
    sampled_query = query.index_select(0, indices).float()
    scale = query.size(-1) ** -0.5
    dense_logits = torch.matmul(sampled_query, key.float().T) * scale
    dense_output = torch.matmul(torch.softmax(dense_logits, dim=-1), value.float())
    dense_lse = torch.logsumexp(dense_logits, dim=-1)

    score, statistics = cluster_risk_statistics(
        query_centroids,
        key,
        value,
        key_centroids,
        value_centroids,
        key_sizes,
        seed,
    )
    maps = {
        "ear_p09": reference_maps[0.9],
        "ear_budget": reference_maps[budget_top_p],
    }
    maps.update(
        {
            name: matched_cardinality_map(
                candidate_score, reference_maps[budget_top_p]
            )
            for name, candidate_score in score.items()
        }
    )
    for score_name in ("signed_cross_r4_per_token", "signed_cross_r8_per_token"):
        for swaps in (1, 2, 4):
            maps[f"ear_swap{swaps}_{score_name}"] = swap_refined_map(
                score[score_name], reference_maps[budget_top_p], swaps
            )

    labels = cluster_labels(key_sizes)
    log_sizes = key_sizes.float().clamp_min(1).log()
    cluster_ids = torch.arange(key_sizes.numel(), device=key.device)
    valid_clusters = torch.ones_like(cluster_ids, dtype=torch.bool)
    result = {}
    for name, dynamic_map in maps.items():
        predictions = []
        prediction_lse = []
        for sample_idx, query_cluster in enumerate(query_clusters):
            output, output_lse = mixed_measure_output(
                sampled_query[sample_idx],
                key,
                value,
                labels,
                dynamic_map[int(query_cluster)],
                key_centroids.float(),
                value_centroids.float(),
                log_sizes,
                cluster_ids,
                valid_clusters,
            )
            predictions.append(output)
            prediction_lse.append(output_lse)
        summary = metrics(
            torch.stack(predictions),
            dense_output,
            torch.stack(prediction_lse),
            dense_lse,
        )
        exact_density, tail_density = map_density(
            dynamic_map, query_sizes, key_sizes, query.size(0)
        )
        summary.update(
            {
                "exact_density": exact_density,
                "tail_representative_density": tail_density,
                "theoretical_density": exact_density + tail_density,
                "mean_selected_clusters": float(dynamic_map.float().sum(-1).mean()),
                "overlap_with_ear_budget": float(
                    (dynamic_map & reference_maps[budget_top_p]).sum()
                    / dynamic_map.sum().clamp_min(1)
                ),
            }
        )
        result[name] = summary
    result["statistics"] = {
        "mean_key_variance": float(statistics["key_variance"].mean()),
        "mean_value_variance": float(statistics["value_variance"].mean()),
        "mean_directional_variance": float(
            statistics["directional_variance"].mean()
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=parse_ints, default=[0, 10, 20, 30])
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--budget-top-p", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--svg-root", default="/home/denghaoran/workspace/Sparse-VideoGen"
    )
    args = parser.parse_args()
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
    reference_maps = {
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
        for top_p in (0.9, args.budget_top_p)
    }

    per_head = []
    for local_head, original_head in enumerate(args.heads):
        row = evaluate_head(
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
                for top_p, dynamic_map in reference_maps.items()
            },
            args.budget_top_p,
            args.sample_count,
            args.seed + original_head,
        )
        row["head"] = original_head
        per_head.append(row)

    methods = [
        name for name in per_head[0] if name not in {"statistics", "head"}
    ]
    summary = {
        method: {
            metric: float(
                sum(row[method][metric] for row in per_head) / len(per_head)
            )
            for metric in (
                "mse",
                "relative_l2",
                "cosine",
                "logsumexp_mae",
                "exact_density",
                "tail_representative_density",
                "theoretical_density",
                "mean_selected_clusters",
                "overlap_with_ear_budget",
            )
        }
        for method in methods
    }
    result = {
        "method": "matched-cardinality error-aware routing for exact blocks",
        "qkv": str(args.qkv),
        "heads": args.heads,
        "sample_count": args.sample_count,
        "budget_reference": (
            f"SVG-EAR top_p={args.budget_top_p:g} selected cluster count per "
            "query cluster"
        ),
        "summary": summary,
        "per_head": per_head,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
