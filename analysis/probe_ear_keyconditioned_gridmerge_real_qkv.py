"""Probe one consumer-conditioned GridMerge representative per key cluster."""

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
    selected = torch.tensor(heads, device=device)
    with np.load(path, allow_pickle=False) as capture:
        return tuple(
            torch.from_numpy(np.asarray(capture[name]))
            .to(device=device, dtype=torch.float16)
            .index_select(1, selected)
            for name in ("q", "k", "v")
        )


def cluster_labels(sizes: torch.Tensor):
    return torch.repeat_interleave(
        torch.arange(sizes.numel(), device=sizes.device), sizes.long()
    )


@torch.inference_mode()
def key_consumer_representatives(
    query_centroids: torch.Tensor,
    key_centroids: torch.Tensor,
    query_sizes: torch.Tensor,
    key_sizes: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    dynamic_map: torch.Tensor,
    tail_rank: int,
):
    """Build one or two disjoint consumer-conditioned reps per key cluster."""

    scale = keys.size(-1) ** -0.5
    logits = torch.matmul(
        query_centroids.float(), key_centroids.float().T
    ) * scale
    logits += key_sizes.float().clamp_min(1).log()[None, :]
    estimated_mass = torch.softmax(logits, dim=-1)
    consumer_weight = (
        query_sizes.float()[:, None]
        * estimated_mass
        * (~dynamic_map).float()
    )
    denominator = consumer_weight.sum(dim=0)
    seeds = torch.matmul(consumer_weight.T, query_centroids.float())
    fallback = (
        query_sizes.float()[:, None] * query_centroids.float()
    ).sum(dim=0) / query_sizes.float().sum().clamp_min(1)
    mean_seeds = torch.where(
        (denominator > 0)[:, None],
        seeds / denominator.clamp_min(1e-20)[:, None],
        fallback[None, :],
    )

    labels = cluster_labels(key_sizes)
    cluster_count = key_sizes.numel()
    if tail_rank == 1:
        token_seed = mean_seeds.index_select(0, labels)
        score = (token_seed * keys.float()).sum(dim=-1) * scale
        group = labels
        representative_count = cluster_count
    elif tail_rank in (2, 4):
        query_score = torch.matmul(
            query_centroids.float(), key_centroids.float().T
        ) * scale
        threshold = (consumer_weight * query_score).sum(dim=0)
        threshold = threshold / denominator.clamp_min(1e-20)
        if tail_rank == 2:
            mode = (query_score >= threshold[None, :]).long()
        else:
            variance = (
                consumer_weight
                * (query_score - threshold[None, :]).square()
            ).sum(dim=0) / denominator.clamp_min(1e-20)
            spread = variance.clamp_min(0).sqrt()
            quartile = 0.67448975
            boundaries = torch.stack(
                (
                    threshold - quartile * spread,
                    threshold,
                    threshold + quartile * spread,
                ),
                dim=0,
            )
            mode = (query_score[:, None, :] >= boundaries[None, :, :]).sum(dim=1)
        mode_id = torch.arange(tail_rank, device=keys.device)
        mode_weight = consumer_weight[..., None] * (
            mode[..., None] == mode_id[None, None, :]
        )
        mode_denominator = mode_weight.sum(dim=0)
        mode_seeds = torch.einsum(
            "qkr,qd->krd", mode_weight, query_centroids.float()
        )
        mode_seeds = torch.where(
            (mode_denominator > 0)[..., None],
            mode_seeds / mode_denominator.clamp_min(1e-20)[..., None],
            mean_seeds[:, None, :],
        )
        token_seed = mode_seeds.index_select(0, labels)
        seed_score = torch.einsum(
            "nrd,nd->nr", token_seed, keys.float()
        ) * scale
        seed_index = torch.arange(tail_rank, device=keys.device)[None, :]
        candidate_group = labels[:, None] * tail_rank + seed_index
        candidate_count = cluster_count * tail_rank
        candidate_max = torch.full(
            (candidate_count,),
            -torch.inf,
            device=keys.device,
            dtype=torch.float32,
        )
        candidate_max.scatter_reduce_(
            0,
            candidate_group.reshape(-1),
            seed_score.reshape(-1),
            reduce="amax",
            include_self=True,
        )
        candidate_exp = (
            seed_score
            - candidate_max.index_select(0, candidate_group.reshape(-1)).view_as(
                seed_score
            )
        ).exp()
        candidate_sum = torch.zeros_like(candidate_max)
        candidate_sum.scatter_add_(
            0, candidate_group.reshape(-1), candidate_exp.reshape(-1)
        )
        log_probability = (
            seed_score
            - candidate_max.index_select(0, candidate_group.reshape(-1)).view_as(
                seed_score
            )
            - candidate_sum
            .index_select(0, candidate_group.reshape(-1))
            .view_as(seed_score)
            .clamp_min(1e-20)
            .log()
        )
        owner = log_probability.argmax(dim=-1)
        group = labels * 2 + owner
        score = seed_score.gather(1, owner[:, None]).squeeze(1)
        representative_count = candidate_count
    else:
        raise ValueError("tail_rank must be one, two, or four")

    maxima = torch.full(
        (representative_count,),
        -torch.inf,
        device=keys.device,
        dtype=torch.float32,
    )
    maxima.scatter_reduce_(0, group, score, reduce="amax", include_self=True)
    exponential = (score - maxima.index_select(0, group)).exp()
    normalizer = torch.zeros_like(maxima)
    normalizer.scatter_add_(0, group, exponential)
    weight = exponential / normalizer.index_select(0, group).clamp_min(1e-20)

    merged_keys = torch.zeros(
        representative_count,
        keys.size(-1),
        device=keys.device,
        dtype=torch.float32,
    )
    merged_values = torch.zeros_like(merged_keys)
    merged_keys.index_add_(0, group, weight[:, None] * keys.float())
    merged_values.index_add_(0, group, weight[:, None] * values.float())
    expected_score = torch.zeros_like(maxima)
    expected_score.scatter_add_(0, group, weight * score)
    entropy = maxima + normalizer.clamp_min(1e-20).log() - expected_score
    representative_cluster = torch.arange(
        representative_count, device=keys.device
    ) // tail_rank
    valid = normalizer > 0
    return merged_keys, merged_values, entropy, representative_cluster, valid


def mixed_measure_output(
    query,
    keys,
    values,
    key_labels,
    exact_clusters,
    tail_keys,
    tail_values,
    tail_bias,
    tail_cluster,
    tail_valid,
):
    exact_token_mask = exact_clusters.index_select(0, key_labels)
    exact_keys = keys[exact_token_mask].float()
    exact_values = values[exact_token_mask].float()
    tail_mask = (~exact_clusters.index_select(0, tail_cluster)) & tail_valid
    scale = query.size(-1) ** -0.5
    logits = torch.cat(
        (
            torch.mv(exact_keys, query.float()) * scale,
            torch.mv(tail_keys[tail_mask], query.float()) * scale
            + tail_bias[tail_mask],
        )
    )
    representatives = torch.cat((exact_values, tail_values[tail_mask]), dim=0)
    probability = torch.softmax(logits, dim=0)
    return torch.mv(representatives.T, probability), torch.logsumexp(logits, dim=0)


def metrics(prediction, target, prediction_lse, target_lse):
    error = prediction - target
    return {
        "mse": float(error.square().mean().item()),
        "relative_l2": float(
            (error.norm() / target.norm().clamp_min(1e-12)).item()
        ),
        "cosine": float(F.cosine_similarity(prediction, target, dim=-1).mean()),
        "logsumexp_mae": float((prediction_lse - target_lse).abs().mean()),
    }


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
    tail_rank,
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
    log_sizes = key_sizes.float().clamp_min(1).log()

    result = {}
    for top_p, dynamic_map in dynamic_maps.items():
        (
            tail_keys,
            tail_values,
            tail_entropy,
            tail_cluster,
            tail_valid,
        ) = key_consumer_representatives(
            query_centroids,
            key_centroids,
            query_sizes,
            key_sizes,
            key,
            value,
            dynamic_map,
            tail_rank,
        )
        conditioned_name = f"key_conditioned_gma_tail_r{tail_rank}"
        predictions = {"ear_mean_tail": [], conditioned_name: []}
        lse = {"ear_mean_tail": [], conditioned_name: []}
        for sample_idx, query_cluster in enumerate(query_clusters):
            exact_clusters = dynamic_map[int(query_cluster)]
            for (
                method,
                method_keys,
                method_values,
                method_bias,
                method_cluster,
                method_valid,
            ) in (
                (
                    "ear_mean_tail",
                    key_centroids.float(),
                    value_centroids.float(),
                    log_sizes,
                    torch.arange(key_sizes.numel(), device=key.device),
                    torch.ones(key_sizes.numel(), device=key.device, dtype=torch.bool),
                ),
                (
                    conditioned_name,
                    tail_keys,
                    tail_values,
                    tail_entropy,
                    tail_cluster,
                    tail_valid,
                ),
            ):
                output, output_lse = mixed_measure_output(
                    sampled_query[sample_idx],
                    key,
                    value,
                    key_labels,
                    exact_clusters,
                    method_keys,
                    method_values,
                    method_bias,
                    method_cluster,
                    method_valid,
                )
                predictions[method].append(output)
                lse[method].append(output_lse)
        result[top_p] = {
            method: metrics(
                torch.stack(predictions[method]),
                dense_output,
                torch.stack(lse[method]),
                dense_lse,
            )
            for method in predictions
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=parse_ints, default=[0, 10, 20, 30])
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--tail-rank", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--top-ps", type=parse_floats, default=[0.9, 0.75, 0.5])
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--svg-root", default="/home/denghaoran/workspace/Sparse-VideoGen"
    )
    args = parser.parse_args()

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
            args.tail_rank,
        )
        for top_p, row in head_result.items():
            row.update(
                {
                    "head": original_head,
                    "exact_density": float(density[top_p][local_head]),
                }
            )
            rows[top_p].append(row)

    sweeps = []
    for top_p in args.top_ps:
        summary = {}
        for method in (
            "ear_mean_tail",
            f"key_conditioned_gma_tail_r{args.tail_rank}",
        ):
            summary[method] = {
                metric: float(
                    sum(row[method][metric] for row in rows[top_p])
                    / len(rows[top_p])
                )
                for metric in ("mse", "relative_l2", "cosine", "logsumexp_mae")
            }
        sweeps.append(
            {
                "top_p": top_p,
                "mean_exact_density": float(density[top_p].mean()),
                "summary": summary,
                "per_head": rows[top_p],
            }
        )

    result = {
        "method": "EAR exact blocks + key-consumer-conditioned GMA tail",
        "qkv": str(args.qkv),
        "heads": args.heads,
        "sample_count": args.sample_count,
        "tail_rank": args.tail_rank,
        "sweeps": sweeps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
