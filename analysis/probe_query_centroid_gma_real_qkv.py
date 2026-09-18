"""Screen query-centroid seed construction for Hard-C4 on native Wan QKV.

For a query ``q`` represented by seed ``u_c``, the leading cluster-mass error
of entropy-corrected GMA is proportional to

    (q - u_c)^T Cov_w(k) (q - u_c).

Euclidean K-means minimizes an isotropic surrogate of this term.  This probe
compares local Lloyd centroids, spherical centroids, and global farthest-point
centers without changing the dense Wan trajectory or the downstream Hard-C4
merge equations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from attention.hard_route_pytorch import (
    biased_sdpa,
    gather_seed_queries,
    hard_route_merge_kv_pytorch,
    wide_local_c4_seed_rows,
)
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


CLUSTERS = 4
LOCAL_CANDIDATES = (
    ("local_euclid_p7_s1", 7, 1, "euclidean", False),
    ("local_euclid_p9_s1", 9, 1, "euclidean", False),
    ("local_euclid_p11_s1", 11, 1, "euclidean", False),
    ("local_euclid_p11_s2", 11, 2, "euclidean", False),
    ("local_euclid_p13_s1", 13, 1, "euclidean", False),
    ("local_euclid_p15_s1", 15, 1, "euclidean", False),
    ("local_euclid_p15_s2", 15, 2, "euclidean", False),
    ("local_euclid_p15_s4", 15, 4, "euclidean", False),
    ("local_euclid_p21_s2", 21, 2, "euclidean", False),
    ("local_norm_p15_s1", 15, 1, "euclidean", True),
    ("local_norm_p15_s2", 15, 2, "euclidean", True),
    ("local_spherical_p15_s2", 15, 2, "cosine", True),
)
LOCAL_KCENTER_CANDIDATES = (
    ("local_kcenter_medoid_p7", 7),
    ("local_kcenter_medoid_p11", 11),
    ("local_kcenter_medoid_p15", 15),
    ("local_kcenter_medoid_p21", 21),
)
LOCAL_C5_CANDIDATES = (
    ("local_c5_uniform_medoids_p11", 11, 0, "euclidean", False),
    ("local_c5_euclid_p11_s1", 11, 1, "euclidean", False),
    ("local_c5_norm_p11_s1", 11, 1, "euclidean", True),
    ("local_c5_euclid_p15_s1", 15, 1, "euclidean", False),
)
GLOBAL_CANDIDATES = (
    ("global_uniform_medoids", "uniform", 0, "euclidean", False),
    ("global_uniform_euclid_s1", "uniform", 1, "euclidean", False),
    ("global_uniform_euclid_s2", "uniform", 2, "euclidean", False),
    ("global_uniform_euclid_s4", "uniform", 4, "euclidean", False),
    ("global_farthest_medoids", "farthest", 0, "euclidean", False),
    ("global_farthest_euclid_s1", "farthest", 1, "euclidean", False),
    ("global_farthest_euclid_s2", "farthest", 2, "euclidean", False),
    ("global_farthest_euclid_s4", "farthest", 4, "euclidean", False),
    ("global_farthest_norm_s2", "farthest", 2, "euclidean", True),
    ("global_farthest_spherical_s2", "farthest", 2, "cosine", True),
)


class ErrorStats:
    def __init__(self):
        self.calls = 0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0

    def add(self, output, reference):
        difference = output.float() - reference.float()
        self.calls += 1
        self.sqerr += float(difference.square().sum())
        self.ref_sq += float(reference.float().square().sum())
        self.numel += difference.numel()

    def summary(self):
        return {
            "calls": self.calls,
            "numel": self.numel,
            "mse_vs_dense": self.sqerr / self.numel,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
        }


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").split()]


def parse_names(value):
    return {item for item in value.replace(",", " ").split() if item}


def _point_distance(points, centers, mode):
    if mode == "euclidean":
        return (points.unsqueeze(-2).float() - centers.unsqueeze(-3).float()).square().mean(-1)
    if mode == "cosine":
        point_direction = F.normalize(points.float(), dim=-1)
        center_direction = F.normalize(centers.float(), dim=-1)
        return 1.0 - torch.einsum("...pd,...cd->...pc", point_direction, center_direction)
    raise ValueError(f"unknown distance mode {mode!r}")


def lloyd_centroids(points, initial_centers, steps, mode="euclidean", normalize=False):
    """Run deterministic Lloyd updates over the penultimate point dimension."""

    centers = initial_centers.float()
    for _ in range(steps):
        cluster_id = _point_distance(points, centers, mode).argmin(dim=-1)
        membership = F.one_hot(cluster_id, centers.size(-2)).float()
        count = membership.sum(dim=-2)
        if mode == "cosine":
            source = F.normalize(points.float(), dim=-1)
        else:
            source = points.float()
        centroid = torch.einsum("...pc,...pd->...cd", membership, source)
        centroid = centroid / count.clamp_min(1).unsqueeze(-1)
        if normalize or mode == "cosine":
            point_norm = points.float().norm(dim=-1)
            target_norm = torch.einsum("...pc,...p->...c", membership, point_norm)
            target_norm = target_norm / count.clamp_min(1)
            centroid = F.normalize(centroid, dim=-1) * target_norm.unsqueeze(-1)
        centers = torch.where((count > 0).unsqueeze(-1), centroid, centers)
    return centers.to(points.dtype)


def farthest_point_medoids(points, clusters=CLUSTERS):
    """Choose deterministic K-center medoids, starting nearest the point mean."""

    points_f32 = points.float()
    mean = points_f32.mean(dim=-2, keepdim=True)
    first = (points_f32 - mean).square().mean(dim=-1).argmin(dim=-1)

    def gather(index):
        return points.gather(
            -2,
            index[..., None, None].expand(*index.shape, 1, points.size(-1)),
        ).squeeze(-2)

    selected = [gather(first)]
    minimum_distance = (points_f32 - selected[0].float().unsqueeze(-2)).square().mean(-1)
    for _ in range(1, clusters):
        index = minimum_distance.argmax(dim=-1)
        center = gather(index)
        selected.append(center)
        distance = (points_f32 - center.float().unsqueeze(-2)).square().mean(-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    return torch.stack(selected, dim=-2)


def global_centroid_seed_queries(query, key_frames, key_rows, init, steps, mode, normalize):
    """Build four centers shared by all key blocks of one query column/head."""

    points = rearrange(query, "b a i j h d -> b a j h i d")
    if init == "uniform":
        index = torch.linspace(
            0, points.size(-2) - 1, CLUSTERS, device=query.device
        ).round().long()
        centers = points.index_select(-2, index)
    elif init == "farthest":
        centers = farthest_point_medoids(points)
    else:
        raise ValueError(f"unknown initialization {init!r}")
    centers = lloyd_centroids(points, centers, steps, mode=mode, normalize=normalize)
    centers = rearrange(centers, "b a j h c d -> b a j c h d")
    return centers[:, :, None, None].expand(
        -1, -1, key_frames, key_rows, -1, -1, -1, -1
    )


def local_centroid_seed_queries(
    query,
    key_frames,
    key_rows,
    candidate_count,
    steps,
    mode,
    normalize,
    clusters=CLUSTERS,
):
    """Replace each wide-local seed set by centroids of its nearby query rows."""

    query_rows = query.size(2)
    key_row = torch.arange(key_rows, device=query.device)
    start = (key_row - candidate_count // 2).clamp(0, query_rows - candidate_count)
    candidate_rows = start[:, None] + torch.arange(candidate_count, device=query.device)
    candidates = query[:, :, candidate_rows]
    points = rearrange(candidates, "b a k p j h d -> b a k j h p d")

    if clusters == CLUSTERS:
        seed_rows = wide_local_c4_seed_rows(
            torch.arange(query.size(1), device=query.device),
            torch.arange(1, device=query.device),
            key_rows,
            query.size(3),
            query_rows,
        )
        seeds = gather_seed_queries(query, seed_rows)[:, :, 0]
        seeds = rearrange(seeds, "b a k j c h d -> b a k j h c d")
    else:
        seed_positions = torch.linspace(
            0, candidate_count - 1, clusters, device=query.device
        ).round().long()
        seeds = candidates.index_select(3, seed_positions)
        seeds = rearrange(seeds, "b a k c j h d -> b a k j h c d")
    centers = lloyd_centroids(points, seeds, steps, mode=mode, normalize=normalize)
    centers = rearrange(centers, "b a k j h c d -> b a k j c h d")
    return centers[:, :, None].expand(-1, -1, key_frames, -1, -1, -1, -1, -1)


def local_kcenter_seed_queries(
    query,
    key_frames,
    key_rows,
    candidate_count,
    clusters=CLUSTERS,
):
    """Choose actual query tokens by greedy Euclidean k-center coverage."""

    query_rows = query.size(2)
    key_row = torch.arange(key_rows, device=query.device)
    start = (key_row - candidate_count // 2).clamp(
        0, query_rows - candidate_count
    )
    candidate_rows = start[:, None] + torch.arange(
        candidate_count, device=query.device
    )
    candidates = query[:, :, candidate_rows]
    points = rearrange(candidates, "b a k p j h d -> b a k j h p d")
    centers = farthest_point_medoids(points, clusters=clusters)
    centers = rearrange(centers, "b a k j h c d -> b a k j c h d")
    return centers[:, :, None].expand(
        -1, -1, key_frames, -1, -1, -1, -1, -1
    )


def fixed_seed_queries(query, key_frames, key_rows):
    seed_rows = wide_local_c4_seed_rows(
        torch.arange(query.size(1), device=query.device),
        torch.arange(key_frames, device=query.device),
        key_rows,
        query.size(3),
        query.size(2),
    )
    return gather_seed_queries(query, seed_rows)


@torch.inference_mode()
def responsibility_centroids_from_merged(
    query,
    merged_key,
    routing_bias,
    initial_seeds,
    key_frames,
    key_rows,
    scale,
    responsibility_power=1,
):
    """Return the closed-form query centroids for compressed responsibilities."""

    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    key_h = rearrange(
        merged_key, "b a f j k c h d -> b h a j (f k c) d"
    ).float()
    bias_h = rearrange(
        routing_bias, "b h a f k j c -> b h a j (f k c)"
    ).float()
    scores = torch.einsum("bhajid,bhajtd->bhajit", query_h, key_h)
    responsibility = torch.softmax(scores * float(scale) - bias_h.unsqueeze(-2), dim=-1)
    centroid_weight = responsibility.pow(int(responsibility_power))

    normalizer = centroid_weight.sum(dim=-2)
    centroid = torch.einsum(
        "bhajit,bhajid->bhajtd", centroid_weight, query_h
    ) / normalizer.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    fallback = rearrange(
        initial_seeds,
        "b a f k j c h d -> b h a j (f k c) d",
    ).float()
    centroid = torch.where((normalizer > 0).unsqueeze(-1), centroid, fallback)
    centroids = rearrange(
        centroid,
        "b h a j (f k c) d -> b a f k j c h d",
        f=key_frames,
        k=key_rows,
        c=initial_seeds.size(5),
    ).to(query.dtype)
    mass = rearrange(
        normalizer,
        "b h a j (f k c) -> b a f k j c h",
        f=key_frames,
        k=key_rows,
        c=initial_seeds.size(5),
    )
    return centroids, mass


@torch.inference_mode()
def responsibility_centroid_seed_queries(query, key, value, initial_seeds, scale):
    """Take one attention-responsibility centroid update from fixed seeds."""

    merged_key, _, routing_bias = hard_route_merge_kv_pytorch(
        initial_seeds, key, value, scale
    )
    centroids, _ = responsibility_centroids_from_merged(
        query,
        merged_key,
        routing_bias,
        initial_seeds,
        key.size(1),
        key.size(2),
        scale,
    )
    return centroids


def hard_seed_assignment(seed_queries, key, scale):
    logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seed_queries, key
    ).float()
    assignment_scores = torch.log_softmax(logits * float(scale), dim=-1)
    best_seed = assignment_scores.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(assignment_scores, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    return assignment


def fixed_assignment_merge_kv(seed_queries, key, value, assignment, scale):
    logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seed_queries, key
    ).float()
    routed_logits = (logits * float(scale)).masked_fill(
        ~assignment, torch.finfo(torch.float32).min
    )
    log_weights = torch.log_softmax(routed_logits, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    bias = (weights * log_weights).sum(dim=-1)
    bias = bias.masked_fill(~assignment.any(dim=-1), float("inf"))
    merge_weights = weights.to(key.dtype)
    merged_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", merge_weights, key
    )
    merged_value = torch.einsum(
        "bhafkjcl,bfklhe->bafjkche", merge_weights, value
    )
    return merged_key, merged_value, bias


def compressed_c4_output(query, merged_key, merged_value, bias, scale):
    query_flat = rearrange(query, "b a i j h d -> (b h a j) i d")
    key_flat = rearrange(merged_key, "b a f j k c h d -> (b h a j) (f k c) d")
    value_flat = rearrange(
        merged_value, "b a f j k c h d -> (b h a j) (f k c) d"
    )
    bias_flat = rearrange(bias, "b h a f k j c -> (b h a j) (f k c)")
    output = biased_sdpa(query_flat, key_flat, value_flat, bias_flat, scale=scale)
    return rearrange(
        output,
        "(b h a j) i d -> b a i j h d",
        b=query.size(0),
        h=query.size(4),
        a=query.size(1),
        j=query.size(3),
    )


def hard_c4_output(query, key, value, seed_queries, scale):
    merged_key, merged_value, bias = hard_route_merge_kv_pytorch(
        seed_queries, key, value, scale
    )
    return compressed_c4_output(query, merged_key, merged_value, bias, scale)


def fixed_route_responsibility_centroid_output(
    query, key, value, initial_seeds, scale
):
    """Update expansion points while retaining the original hard key partition."""

    assignment = hard_seed_assignment(initial_seeds, key, scale)
    initial_key, initial_value, initial_bias = hard_route_merge_kv_pytorch(
        initial_seeds, key, value, scale
    )
    centroids, responsibility_mass = responsibility_centroids_from_merged(
        query,
        initial_key,
        initial_bias,
        initial_seeds,
        key.size(1),
        key.size(2),
        scale,
    )
    merged_key, merged_value, bias = fixed_assignment_merge_kv(
        centroids, key, value, assignment, scale
    )
    outputs = {
        "responsibility_centroid_value_only": compressed_c4_output(
            query, initial_key, merged_value, initial_bias, scale
        ),
        "responsibility_centroid_bias_only": compressed_c4_output(
            query, initial_key, initial_value, bias, scale
        ),
        "responsibility_centroid_bias_value": compressed_c4_output(
            query, initial_key, merged_value, bias, scale
        ),
        "responsibility_centroid_key_only": compressed_c4_output(
            query, merged_key, initial_value, initial_bias, scale
        ),
        "responsibility_centroid_mass_only": compressed_c4_output(
            query, merged_key, initial_value, bias, scale
        ),
        "responsibility_centroid_fixed_route": compressed_c4_output(
            query, merged_key, merged_value, bias, scale
        ),
    }
    update_gain = responsibility_mass * (
        centroids.float() - initial_seeds.float()
    ).square().sum(dim=-1)
    for updates in (1, 2):
        selected = update_gain.topk(updates, dim=5).indices
        seed_mask = torch.zeros_like(update_gain, dtype=torch.bool)
        seed_mask.scatter_(dim=5, index=selected, value=True)
        merged_mask = rearrange(seed_mask, "b a f k j c h -> b a f j k c h")
        bias_mask = rearrange(seed_mask, "b a f k j c h -> b h a f k j c")
        selected_key = torch.where(
            merged_mask.unsqueeze(-1), merged_key, initial_key
        )
        selected_value = torch.where(
            merged_mask.unsqueeze(-1), merged_value, initial_value
        )
        selected_bias = torch.where(bias_mask, bias, initial_bias)
        outputs[f"responsibility_centroid_top{updates}_gain"] = compressed_c4_output(
            query, selected_key, selected_value, selected_bias, scale
        )
    return outputs, centroids


def fixed_route_influence_centroid_output(
    query, key, value, initial_seeds, scale
):
    """Use squared responsibilities from the diagonal output-error objective."""

    assignment = hard_seed_assignment(initial_seeds, key, scale)
    initial_key, initial_value, initial_bias = hard_route_merge_kv_pytorch(
        initial_seeds, key, value, scale
    )
    centroids, influence_mass = responsibility_centroids_from_merged(
        query,
        initial_key,
        initial_bias,
        initial_seeds,
        key.size(1),
        key.size(2),
        scale,
        responsibility_power=2,
    )
    merged_key, merged_value, bias = fixed_assignment_merge_kv(
        centroids, key, value, assignment, scale
    )
    update_gain = influence_mass * (
        centroids.float() - initial_seeds.float()
    ).square().sum(dim=-1)
    outputs = {}
    for updates in (1, 2):
        selected = update_gain.topk(updates, dim=5).indices
        seed_mask = torch.zeros_like(update_gain, dtype=torch.bool)
        seed_mask.scatter_(dim=5, index=selected, value=True)
        merged_mask = rearrange(seed_mask, "b a f k j c h -> b a f j k c h")
        bias_mask = rearrange(seed_mask, "b a f k j c h -> b h a f k j c")
        selected_key = torch.where(
            merged_mask.unsqueeze(-1), merged_key, initial_key
        )
        selected_value = torch.where(
            merged_mask.unsqueeze(-1), merged_value, initial_value
        )
        selected_bias = torch.where(bias_mask, bias, initial_bias)
        outputs[f"influence_centroid_top{updates}_gain"] = compressed_c4_output(
            query, selected_key, selected_value, selected_bias, scale
        )
    return outputs, centroids


def _overwrite_selected_query_rows(approximate, reference, score, rows):
    selected = score.topk(int(rows), dim=-1).indices
    selected_mask = torch.zeros_like(score, dtype=torch.bool)
    selected_mask.scatter_(dim=-1, index=selected, value=True)
    selected_mask = rearrange(
        selected_mask, "b h a j i -> b a i j h"
    ).unsqueeze(-1)
    return torch.where(selected_mask, reference, approximate)


def budgeted_exact_query_outputs(
    query, key, value, initial_seeds, reference, scale, exact_rows=3
):
    """Probe theorem-driven exact-query allocation at a fixed FLOP budget."""

    merged_key, merged_value, routing_bias = hard_route_merge_kv_pytorch(
        initial_seeds, key, value, scale
    )
    approximate = compressed_c4_output(
        query, merged_key, merged_value, routing_bias, scale
    )
    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    key_h = rearrange(
        merged_key, "b a f j k c h d -> b h a j (f k c) d"
    ).float()
    bias_h = rearrange(
        routing_bias, "b h a f k j c -> b h a j (f k c)"
    ).float()
    prototypes_h = rearrange(
        initial_seeds, "b a f k j c h d -> b h a j (f k c) d"
    ).float()
    logits = torch.einsum("bhajid,bhajtd->bhajit", query_h, key_h)
    responsibility = torch.softmax(
        logits * float(scale) - bias_h.unsqueeze(-2), dim=-1
    )
    squared_displacement = (
        query_h.unsqueeze(-2) - prototypes_h.unsqueeze(-3)
    ).square().sum(dim=-1)
    distortion = (responsibility * squared_displacement).sum(dim=-1)

    assignment = hard_seed_assignment(initial_seeds, key, scale)
    seed_key_logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", initial_seeds, key
    ).float()
    routed_logits = (seed_key_logits * float(scale)).masked_fill(
        ~assignment, torch.finfo(torch.float32).min
    )
    merge_weights = torch.softmax(routed_logits, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    key_second_moment = torch.einsum(
        "bhafkjcl,bfklh->bhafkjc",
        merge_weights,
        key.float().square().sum(dim=-1),
    )
    value_second_moment = torch.einsum(
        "bhafkjcl,bfklh->bhafkjc",
        merge_weights,
        value.float().square().sum(dim=-1),
    )
    merged_key_norm = rearrange(
        merged_key.float().square().sum(dim=-1),
        "b a f j k c h -> b h a f k j c",
    )
    merged_value_norm = rearrange(
        merged_value.float().square().sum(dim=-1),
        "b a f j k c h -> b h a f k j c",
    )
    key_variance = (key_second_moment - merged_key_norm).clamp_min(0.0)
    value_variance = (value_second_moment - merged_value_norm).clamp_min(0.0)
    key_variance = rearrange(
        key_variance,
        "b h a f k j c -> b h a j (f k c)",
    )
    value_variance = rearrange(
        value_variance,
        "b h a f k j c -> b h a j (f k c)",
    )
    mass_curvature_bound = (
        responsibility * key_variance.unsqueeze(-2) * squared_displacement
    ).sum(dim=-1)
    value_uncertainty = (
        responsibility * value_variance.unsqueeze(-2)
    ).sum(dim=-1)
    crosscovariance_bound = (
        responsibility
        * key_variance.unsqueeze(-2)
        * value_variance.unsqueeze(-2)
        * squared_displacement
    ).sum(dim=-1)
    attention_entropy = -(
        responsibility
        * responsibility.clamp_min(torch.finfo(torch.float32).tiny).log()
    ).sum(dim=-1)
    actual_error = rearrange(
        (approximate.float() - reference.float()).square().mean(dim=-1),
        "b a i j h -> b h a j i",
    )

    uniform_rows = torch.linspace(
        0, query.size(2) - 1, int(exact_rows), device=query.device
    ).round().long()
    uniform_mask = torch.zeros_like(actual_error, dtype=torch.bool)
    uniform_mask[..., uniform_rows] = True
    uniform_mask = rearrange(
        uniform_mask, "b h a j i -> b a i j h"
    ).unsqueeze(-1)
    return {
        "exact_query_uniform3": torch.where(
            uniform_mask, reference, approximate
        ),
        "exact_query_top3_distortion": _overwrite_selected_query_rows(
            approximate, reference, distortion, exact_rows
        ),
        "exact_query_top3_mass_bound": _overwrite_selected_query_rows(
            approximate, reference, mass_curvature_bound, exact_rows
        ),
        "exact_query_top3_value_uncertainty": _overwrite_selected_query_rows(
            approximate, reference, value_uncertainty, exact_rows
        ),
        "exact_query_top3_crosscov_bound": _overwrite_selected_query_rows(
            approximate, reference, crosscovariance_bound, exact_rows
        ),
        "exact_query_top3_entropy": _overwrite_selected_query_rows(
            approximate, reference, attention_entropy, exact_rows
        ),
        "exact_query_oracle3": _overwrite_selected_query_rows(
            approximate, reference, actual_error, exact_rows
        ),
    }


def seed_fit_mse(query, seed_queries):
    """Average nearest-seed distance, averaged across repeated key blocks."""

    query_points = rearrange(query, "b a i j h d -> b a 1 1 j h i d")
    seeds = rearrange(seed_queries, "b a f k j c h d -> b a f k j h c d")
    return float(_point_distance(query_points, seeds, "euclidean").amin(-1).mean())


def install_probe(
    stats, per_call_stats, per_head_stats, per_call_head_stats, fit_stats, args
):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    selected_candidates = parse_names(args.candidates) if args.candidates else None

    def enabled(name):
        return name == "fixed_wide" or selected_candidates is None or name in selected_candidates

    state = {"calls_total": 0, "grid_size": None, "evaluated": []}

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len, heads, dim = *x.shape[:2], self.num_heads, self.head_dim
        q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(
            batch, seq_len, heads, dim
        )
        k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(
            batch, seq_len, heads, dim
        )
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        dense = wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

        state["calls_total"] += 1
        call = state["calls_total"]
        frames, rows, width = [int(item) for item in grid_size]
        state["grid_size"] = [frames, rows, width]
        if call in record_calls:
            head_index = torch.tensor(selected_heads, device=q.device)
            q_blocks = _hard_c4_video_to_blocks(
                q.index_select(2, head_index).contiguous(), rows, width
            )
            k_blocks = _hard_c4_video_to_blocks(
                k.index_select(2, head_index).contiguous(), rows, width
            )
            v_blocks = _hard_c4_video_to_blocks(
                v.index_select(2, head_index).contiguous(), rows, width
            )
            dense_blocks = _hard_c4_video_to_blocks(
                dense.index_select(2, head_index).contiguous(), rows, width
            )
            step = (call - 1) // 40
            layer = (call - 1) % 40
            query_frame = (step * 7 + layer * 3) % frames
            query = q_blocks[:, query_frame : query_frame + 1]
            reference = dense_blocks[:, query_frame : query_frame + 1]
            scale = dim**-0.5

            fixed_seeds = fixed_seed_queries(query, frames, rows)
            candidates = [("fixed_wide", fixed_seeds)]

            def record_candidate(name, output, seed_queries):
                fit_stats[name].append(seed_fit_mse(query, seed_queries))
                stats[name].add(output, reference)
                per_call_stats[call][name].add(output, reference)
                for position, head in enumerate(selected_heads):
                    head_output = output[..., position : position + 1, :]
                    head_reference = reference[..., position : position + 1, :]
                    per_head_stats[head][name].add(head_output, head_reference)
                    per_call_head_stats[call][head][name].add(
                        head_output, head_reference
                    )

            centroid_names = {
                "responsibility_centroid_value_only",
                "responsibility_centroid_bias_only",
                "responsibility_centroid_bias_value",
                "responsibility_centroid_key_only",
                "responsibility_centroid_mass_only",
                "responsibility_centroid_fixed_route",
                "responsibility_centroid_top1_gain",
                "responsibility_centroid_top2_gain",
            }
            if any(enabled(name) for name in centroid_names):
                centroid_outputs, centroid_seeds = (
                    fixed_route_responsibility_centroid_output(
                        query, k_blocks, v_blocks, fixed_seeds, scale
                    )
                )
                for name, output in centroid_outputs.items():
                    if enabled(name):
                        record_candidate(name, output, centroid_seeds)
                del centroid_outputs, centroid_seeds
            influence_names = {
                "influence_centroid_top1_gain",
                "influence_centroid_top2_gain",
            }
            if any(enabled(name) for name in influence_names):
                influence_outputs, influence_seeds = (
                    fixed_route_influence_centroid_output(
                        query, k_blocks, v_blocks, fixed_seeds, scale
                    )
                )
                for name, output in influence_outputs.items():
                    if enabled(name):
                        record_candidate(name, output, influence_seeds)
                del influence_outputs, influence_seeds
            exact_query_names = {
                "exact_query_uniform3",
                "exact_query_top3_distortion",
                "exact_query_top3_mass_bound",
                "exact_query_top3_value_uncertainty",
                "exact_query_top3_crosscov_bound",
                "exact_query_top3_entropy",
                "exact_query_oracle3",
            }
            if any(enabled(name) for name in exact_query_names):
                exact_query_outputs = budgeted_exact_query_outputs(
                    query,
                    k_blocks,
                    v_blocks,
                    fixed_seeds,
                    reference,
                    scale,
                    exact_rows=3,
                )
                for name, output in exact_query_outputs.items():
                    if enabled(name):
                        record_candidate(name, output, fixed_seeds)
                del exact_query_outputs
            if enabled("responsibility_centroid_em1"):
                candidates.append(
                    (
                        "responsibility_centroid_em1",
                        responsibility_centroid_seed_queries(
                            query, k_blocks, v_blocks, fixed_seeds, scale
                        ),
                    )
                )
            for name, count, steps, mode, normalize in LOCAL_CANDIDATES:
                if not enabled(name):
                    continue
                candidates.append(
                    (
                        name,
                        local_centroid_seed_queries(
                            query, frames, rows, count, steps, mode, normalize
                        ),
                    )
                )
            for name, count in LOCAL_KCENTER_CANDIDATES:
                if not enabled(name):
                    continue
                candidates.append(
                    (
                        name,
                        local_kcenter_seed_queries(
                            query, frames, rows, count
                        ),
                    )
                )
            for name, count, steps, mode, normalize in LOCAL_C5_CANDIDATES:
                if not enabled(name):
                    continue
                candidates.append(
                    (
                        name,
                        local_centroid_seed_queries(
                            query,
                            frames,
                            rows,
                            count,
                            steps,
                            mode,
                            normalize,
                            clusters=5,
                        ),
                    )
                )
            for name, init, steps, mode, normalize in GLOBAL_CANDIDATES:
                if not enabled(name):
                    continue
                candidates.append(
                    (
                        name,
                        global_centroid_seed_queries(
                            query, frames, rows, init, steps, mode, normalize
                        ),
                    )
                )

            for name, seed_queries in candidates:
                output = hard_c4_output(
                    query, k_blocks, v_blocks, seed_queries, scale
                )
                record_candidate(name, output, seed_queries)
                del output, seed_queries

            state["evaluated"].append(
                {"call": call, "step": step, "layer": layer, "query_frame": query_frame}
            )
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def theoretical_flops(svg_density, local_candidate_count=15, lloyd_steps=2):
    frames, rows, width, clusters = 21, 45, 80, CLUSTERS
    hard_c4 = (3 * clusters * width + 2 * rows * clusters) / (2 * rows * width)
    hard_c5 = (3 * 5 * width + 2 * rows * 5) / (2 * rows * width)
    responsibility_bias_value = (
        4 * clusters * width + 3 * rows * clusters
    ) / (2 * rows * width)
    responsibility_full = (
        5 * clusters * width + 4 * rows * clusters
    ) / (2 * rows * width)
    responsibility_top_r = {
        updates: (
            2 * clusters * width
            + 4 * rows * clusters
            + 3 * updates * width
        )
        / (2 * rows * width)
        for updates in (1, 2)
    }
    exact_query_rows = 3
    exact_query_top3 = hard_c4 + exact_query_rows / rows
    # Conservatively count distance, assignment, and centroid accumulation as
    # four multiply-add equivalents per point/center/channel per Lloyd step.
    global_overhead = 4 * lloyd_steps * rows * clusters / (
        4 * rows * width * frames * rows * width
    )
    local_overhead = 4 * lloyd_steps * rows * local_candidate_count * clusters / (
        4 * rows * width * frames * rows * width
    )
    return {
        "hard_c4_ratio_vs_dense": hard_c4,
        "hard_c5_ratio_vs_dense": hard_c5,
        "responsibility_bias_value_ratio_vs_dense": responsibility_bias_value,
        "responsibility_bias_value_ratio_vs_svg_ear": (
            responsibility_bias_value / svg_density
        ),
        "responsibility_bias_value_passes_limit": (
            responsibility_bias_value <= 0.70 * svg_density
        ),
        "responsibility_full_ratio_vs_dense": responsibility_full,
        "responsibility_full_ratio_vs_svg_ear": responsibility_full / svg_density,
        "responsibility_full_passes_limit": responsibility_full <= 0.70 * svg_density,
        "responsibility_top_r": {
            str(updates): {
                "ratio_vs_dense": ratio,
                "ratio_vs_svg_ear": ratio / svg_density,
                "passes_limit": ratio <= 0.70 * svg_density,
            }
            for updates, ratio in responsibility_top_r.items()
        },
        "exact_query_top3_ratio_vs_dense": exact_query_top3,
        "exact_query_top3_ratio_vs_svg_ear": exact_query_top3 / svg_density,
        "exact_query_top3_passes_limit": exact_query_top3 <= 0.70 * svg_density,
        "global_centroid_ratio_vs_dense": hard_c4 + global_overhead,
        "local_centroid_ratio_vs_dense": hard_c4 + local_overhead,
        "limit_ratio_vs_dense": 0.70 * svg_density,
        "global_passes_limit": hard_c4 + global_overhead <= 0.70 * svg_density,
        "local_passes_limit": hard_c4 + local_overhead <= 0.70 * svg_density,
        "local_c5_passes_limit": hard_c5 + local_overhead <= 0.70 * svg_density,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--heads", default="0,20")
    parser.add_argument("--record_calls", default="1")
    parser.add_argument(
        "--candidates",
        default="",
        help="Optional comma-separated candidate names; fixed_wide is always included.",
    )
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    stats = defaultdict(ErrorStats)
    per_call_stats = defaultdict(lambda: defaultdict(ErrorStats))
    per_head_stats = defaultdict(lambda: defaultdict(ErrorStats))
    per_call_head_stats = defaultdict(
        lambda: defaultdict(lambda: defaultdict(ErrorStats))
    )
    fit_stats = defaultdict(list)
    state = install_probe(
        stats,
        per_call_stats,
        per_head_stats,
        per_call_head_stats,
        fit_stats,
        args,
    )
    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: value.summary() for name, value in sorted(stats.items())}
    baseline = errors["fixed_wide"]["mse_vs_dense"]
    for value in errors.values():
        value["mse_ratio_vs_fixed_wide"] = value["mse_vs_dense"] / baseline
    per_call_errors = {}
    for call, call_values in sorted(per_call_stats.items()):
        summaries = {
            name: value.summary() for name, value in sorted(call_values.items())
        }
        call_baseline = summaries["fixed_wide"]["mse_vs_dense"]
        for value in summaries.values():
            value["mse_ratio_vs_fixed_wide"] = (
                value["mse_vs_dense"] / call_baseline
            )
        per_call_errors[str(call)] = summaries
    per_head_errors = {}
    for head, head_values in sorted(per_head_stats.items()):
        summaries = {
            name: value.summary() for name, value in sorted(head_values.items())
        }
        head_baseline = summaries["fixed_wide"]["mse_vs_dense"]
        for value in summaries.values():
            value["mse_ratio_vs_fixed_wide"] = (
                value["mse_vs_dense"] / head_baseline
            )
        per_head_errors[str(head)] = summaries
    per_call_head_errors = {}
    for call, heads in sorted(per_call_head_stats.items()):
        per_call_head_errors[str(call)] = {}
        for head, head_values in sorted(heads.items()):
            summaries = {
                name: value.summary()
                for name, value in sorted(head_values.items())
            }
            head_baseline = summaries["fixed_wide"]["mse_vs_dense"]
            for value in summaries.values():
                value["mse_ratio_vs_fixed_wide"] = (
                    value["mse_vs_dense"] / head_baseline
                )
            per_call_head_errors[str(call)][str(head)] = summaries
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "grid_size": state["grid_size"],
        "calls_total": state["calls_total"],
        "record_calls": parse_ints(args.record_calls),
        "heads": parse_ints(args.heads),
        "wall_seconds": time.perf_counter() - started,
        "errors_vs_dense": errors,
        "per_call_errors_vs_dense": per_call_errors,
        "per_head_errors_vs_dense": per_head_errors,
        "per_call_head_errors_vs_dense": per_call_head_errors,
        "seed_fit_mse": {
            name: sum(values) / len(values) for name, values in sorted(fit_stats.items())
        },
        "flops": theoretical_flops(args.svg_ear_density),
        "derivation": {
            "local_gap": "0.5 * beta^2 * (q-u_c)^T Cov_w(k) (q-u_c) + O(||q-u_c||^3)",
            "isotropic_surrogate": "min_{u_1,...,u_C} sum_i min_c ||q_i-u_c||_2^2",
            "trajectory": "dense",
            "intended_schedule": "all 40 layers at every denoise step",
        },
        "per_call": state["evaluated"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
