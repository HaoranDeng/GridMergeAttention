"""Test query-coreset calibration of GMA cluster masses on native Wan QKV.

The hard key partition and merged values remain unchanged.  Exact
log-partition residuals are evaluated at three query anchors and transferred
to the remaining queries.  The model itself follows the dense trajectory, so
this probe isolates the attention approximation before any production kernel
work or end-to-end generation experiment.
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
from analysis.probe_query_centroid_gma_real_qkv import (
    ErrorStats,
    farthest_point_medoids,
    fixed_seed_queries,
    hard_seed_assignment,
    lloyd_centroids,
    parse_ints,
)
from attention.hard_route_pytorch import hard_route_merge_kv_pytorch
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


class MeanStats:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value):
        value = value.float()
        self.total += float(value.sum())
        self.count += value.numel()

    def mean(self):
        return self.total / max(self.count, 1)


def cluster_log_partitions(query, key, assignment, scale, row_chunk=5):
    """Return exact log-sum-exp for every fixed hard key cluster."""

    outputs = []
    for start in range(0, query.size(2), int(row_chunk)):
        query_chunk = query[:, :, start : start + int(row_chunk)]
        scores = torch.einsum(
            "baijhd,bfklhd->bhafkjil", query_chunk, key
        ).float()
        scores.mul_(float(scale))
        clusters = []
        for cluster in range(assignment.size(-2)):
            cluster_mask = assignment[..., cluster, :].unsqueeze(-2)
            clusters.append(
                scores.masked_fill(~cluster_mask, float("-inf")).logsumexp(-1)
            )
        outputs.append(torch.stack(clusters, dim=-1))
    return torch.cat(outputs, dim=-2)


def compressed_cluster_logits(query, merged_key, routing_bias, scale):
    logits = torch.einsum(
        "baijhd,bafjkchd->bhafkjic", query, merged_key
    ).float()
    return logits * float(scale) - routing_bias.float().unsqueeze(-2)


def flatten_logits(logits):
    return rearrange(logits, "b h a f k j i c -> b h a j i (f k c)")


def compressed_output(logits, merged_value):
    probability = torch.softmax(flatten_logits(logits), dim=-1)
    values = rearrange(
        merged_value, "b a f j k c h d -> b h a j (f k c) d"
    ).float()
    output = torch.einsum("bhajit,bhajtd->bhajid", probability, values)
    return rearrange(output, "b h a j i d -> b a i j h d")


def coupled_cumulant_outputs(
    query,
    key,
    value,
    seeds,
    assignment,
    merged_key,
    merged_value,
    approximate_logits,
    exact_logits,
    scale,
    row_chunk=3,
):
    """Evaluate the second-order mass and first-order radial-value expansion."""

    seed_scores = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seeds, key
    ).float()
    routed_scores = (seed_scores * float(scale)).masked_fill(
        ~assignment, float("-inf")
    )
    log_weights = torch.log_softmax(routed_scores, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    weights = log_weights.exp().masked_fill(~assignment, 0.0)

    value_h = rearrange(value, "b f k l h d -> b h f k l d").float()
    merged_value_h = rearrange(
        merged_value, "b a f j k c h d -> b h a f k j c d"
    ).float()
    merged_value_norm = merged_value_h.square().sum(dim=-1)
    value_projections = [
        torch.einsum(
            "bhfkld,bhafkjd->bhafkjl",
            value_h,
            merged_value_h[..., cluster, :],
        )
        for cluster in range(assignment.size(-2))
    ]

    merged_value_flat = rearrange(
        merged_value_h, "b h a f k j c d -> b h a j (f k c) d"
    )
    outputs = defaultdict(list)
    for start in range(0, query.size(2), int(row_chunk)):
        query_chunk = query[:, :, start : start + int(row_chunk)]
        query_scores = torch.einsum(
            "baijhd,bfklhd->bhafkjil", query_chunk, key
        ).float()

        mass_terms = []
        radial_terms = []
        for cluster in range(assignment.size(-2)):
            cluster_weights = weights[..., cluster, :]
            displacement_score = (
                query_scores - seed_scores[..., cluster, :].unsqueeze(-2)
            )
            mean_displacement = (
                cluster_weights.unsqueeze(-2) * displacement_score
            ).sum(dim=-1)
            variance = (
                cluster_weights.unsqueeze(-2) * displacement_score.square()
            ).sum(dim=-1) - mean_displacement.square()
            mass_terms.append(
                0.5 * float(scale) ** 2 * variance.clamp_min(0.0)
            )

            projected_cross_moment = (
                cluster_weights.unsqueeze(-2)
                * value_projections[cluster].unsqueeze(-2)
                * displacement_score
            ).sum(dim=-1)
            value_norm = merged_value_norm[..., cluster].unsqueeze(-1)
            projected_covariance = (
                projected_cross_moment - value_norm * mean_displacement
            )
            radial_terms.append(
                float(scale)
                * projected_covariance
                / value_norm.clamp_min(1e-20)
            )

        mass_correction = torch.stack(mass_terms, dim=-1)
        radial_correction = torch.stack(radial_terms, dim=-1)
        linear_scale = 1.0 + radial_correction
        positive_scale = linear_scale.clamp_min(0.0)

        approximate_chunk = approximate_logits[
            ..., start : start + int(row_chunk), :
        ]
        exact_chunk = exact_logits[..., start : start + int(row_chunk), :]

        def evaluate(logits, value_scale=None):
            probability = torch.softmax(flatten_logits(logits), dim=-1)
            if value_scale is None:
                return torch.einsum(
                    "bhajit,bhajtd->bhajid", probability, merged_value_flat
                )
            scale_flat = flatten_logits(value_scale)
            return torch.einsum(
                "bhajit,bhajit,bhajtd->bhajid",
                probability,
                scale_flat,
                merged_value_flat,
            )

        corrected_logits = approximate_chunk + mass_correction
        outputs["cumulant_mass_second_order"].append(
            evaluate(corrected_logits)
        )
        outputs["cumulant_radial_value_first_order"].append(
            evaluate(approximate_chunk, linear_scale)
        )
        outputs["coupled_cumulant_linear"].append(
            evaluate(corrected_logits, linear_scale)
        )
        outputs["coupled_cumulant_positive"].append(
            evaluate(corrected_logits, positive_scale)
        )
        outputs["oracle_mass_cumulant_radial"].append(
            evaluate(exact_chunk, linear_scale)
        )

    return {
        name: rearrange(
            torch.cat(chunks, dim=-2), "b h a j i d -> b a i j h d"
        ).to(query.dtype)
        for name, chunks in outputs.items()
    }


def oracle_cluster_value_scale_outputs(
    query,
    key,
    value,
    assignment,
    merged_value,
    approximate_logits,
    scale,
    row_chunk=3,
):
    """Return exact-mass scalar-value oracles and an exact reconstruction."""

    value_h = rearrange(value, "b f k l h d -> b h f k l d").float()
    approximate_value = rearrange(
        merged_value, "b a f j k c h d -> b h a f k j c d"
    ).float()
    outputs = defaultdict(list)
    for start in range(0, query.size(2), int(row_chunk)):
        query_chunk = query[:, :, start : start + int(row_chunk)]
        scores = torch.einsum(
            "baijhd,bfklhd->bhafkjil", query_chunk, key
        ).float()
        scores.mul_(float(scale))
        exact_logits = []
        exact_values = []
        for cluster in range(assignment.size(-2)):
            cluster_mask = assignment[..., cluster, :].unsqueeze(-2)
            masked_scores = scores.masked_fill(~cluster_mask, float("-inf"))
            log_partition = masked_scores.logsumexp(-1)
            log_weight = masked_scores - log_partition.unsqueeze(-1)
            weight = log_weight.exp().masked_fill(~cluster_mask, 0.0)
            exact_value = torch.einsum(
                "bhafkjil,bhfkld->bhafkjid", weight, value_h
            )
            exact_logits.append(log_partition)
            exact_values.append(exact_value)
        exact_logits = torch.stack(exact_logits, dim=-1)
        exact_value = torch.stack(exact_values, dim=-2)
        approximate_value_chunk = approximate_value.unsqueeze(-3).expand(
            *exact_value.shape
        )
        numerator = (exact_value * approximate_value_chunk).sum(dim=-1)
        denominator = approximate_value_chunk.square().sum(dim=-1)
        signed_scale = numerator / denominator.clamp_min(1e-20)
        positive_scale = signed_scale.clamp_min(0.0)

        exact_probability = torch.softmax(flatten_logits(exact_logits), dim=-1)
        approximate_probability = torch.softmax(
            flatten_logits(approximate_logits[..., start : start + int(row_chunk), :]),
            dim=-1,
        )
        exact_value_flat = rearrange(
            exact_value,
            "b h a f k j i c d -> b h a j i (f k c) d",
        )
        approximate_value_flat = rearrange(
            approximate_value_chunk,
            "b h a f k j i c d -> b h a j i (f k c) d",
        )
        signed_scale = rearrange(
            signed_scale, "b h a f k j i c -> b h a j i (f k c)"
        )
        positive_scale = rearrange(
            positive_scale, "b h a f k j i c -> b h a j i (f k c)"
        )

        def weighted_output(probability, values):
            return torch.einsum("bhajit,bhajitd->bhajid", probability, values)

        outputs["oracle_exact_cluster_reconstruction"].append(
            weighted_output(exact_probability, exact_value_flat)
        )
        outputs["oracle_exact_mass_signed_value_scale"].append(
            weighted_output(
                exact_probability,
                approximate_value_flat * signed_scale.unsqueeze(-1),
            )
        )
        outputs["oracle_exact_mass_positive_value_scale"].append(
            weighted_output(
                exact_probability,
                approximate_value_flat * positive_scale.unsqueeze(-1),
            )
        )
        outputs["oracle_approx_mass_signed_value_scale"].append(
            weighted_output(
                approximate_probability,
                approximate_value_flat * signed_scale.unsqueeze(-1),
            )
        )

    return {
        name: rearrange(
            torch.cat(chunks, dim=-2), "b h a j i d -> b a i j h d"
        )
        for name, chunks in outputs.items()
    }


def mass_kl(exact_logits, candidate_logits):
    exact_log_probability = torch.log_softmax(flatten_logits(exact_logits), dim=-1)
    candidate_log_probability = torch.log_softmax(
        flatten_logits(candidate_logits), dim=-1
    )
    probability = exact_log_probability.exp()
    term = probability * (exact_log_probability - candidate_log_probability)
    return torch.where(probability > 0, term, 0.0).sum(dim=-1)


def nearest_residual(points, centers, anchor_residual):
    distance = (
        points.unsqueeze(-2).float() - centers.unsqueeze(-3).float()
    ).square().mean(dim=-1)
    nearest = distance.argmin(dim=-1)
    index = nearest.unsqueeze(-1).expand(*nearest.shape, anchor_residual.size(-1))
    predicted = anchor_residual.gather(dim=-2, index=index)
    return predicted, distance.amin(dim=-1)


def isotropic_quadratic_residual(points, centers, anchor_residual, seed_queries):
    """Fit the nonnegative second-order mass gap around each routing seed."""

    seeds = rearrange(
        seed_queries, "b a f k j c h d -> b h a j (f k c) d"
    ).float()
    anchor_distance = (
        centers.unsqueeze(-2).float() - seeds.unsqueeze(-3)
    ).square().mean(dim=-1)
    query_distance = (
        points.unsqueeze(-2).float() - seeds.unsqueeze(-3)
    ).square().mean(dim=-1)
    numerator = (anchor_distance * anchor_residual).sum(dim=-2)
    denominator = anchor_distance.square().sum(dim=-2)
    coefficient = (numerator / denominator.clamp_min(1e-20)).clamp_min(0.0)
    return query_distance * coefficient.unsqueeze(-2)


def anchor_residuals(centers, key, assignment, merged_key, routing_bias, scale):
    anchor_query = rearrange(centers, "b h a j r d -> b a r j h d")
    exact = cluster_log_partitions(anchor_query, key, assignment, scale)
    approximate = compressed_cluster_logits(
        anchor_query, merged_key, routing_bias, scale
    )
    active = assignment.any(dim=-1).unsqueeze(-2)
    residual = torch.where(active, exact - approximate, 0.0)
    return rearrange(
        residual, "b h a f k j r c -> b h a j r (f k c)"
    )


def corrected_logits(approximate, predicted_residual):
    flattened = flatten_logits(approximate) + predicted_residual
    return rearrange(
        flattened,
        "b h a j i (f k c) -> b h a f k j i c",
        f=approximate.size(3),
        k=approximate.size(4),
        c=approximate.size(-1),
    )


def install_probe(
    stats,
    math_stats,
    kl_stats,
    cover_stats,
    per_call_stats,
    per_call_math_stats,
    args,
):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
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

            seeds = fixed_seed_queries(query, frames, rows)
            assignment = hard_seed_assignment(seeds, k_blocks, scale)
            merged_key, merged_value, routing_bias = hard_route_merge_kv_pytorch(
                seeds, k_blocks, v_blocks, scale
            )
            approximate_logits = compressed_cluster_logits(
                query, merged_key, routing_bias, scale
            )
            exact_logits = cluster_log_partitions(
                query, k_blocks, assignment, scale, row_chunk=args.row_chunk
            )

            candidates = {
                "fixed_wide": approximate_logits,
                "oracle_cluster_mass": exact_logits,
            }
            points = rearrange(query, "b a i j h d -> b h a j i d")
            uniform_index = torch.linspace(
                0, rows - 1, args.anchors, device=query.device
            ).round().long()
            uniform_centers = points.index_select(-2, uniform_index)
            kcenter_centers = farthest_point_medoids(points, clusters=args.anchors)
            kmeans_centers = lloyd_centroids(
                points, kcenter_centers, steps=2, mode="euclidean", normalize=False
            )

            anchor_values = {}
            for name, centers in (
                ("uniform3", uniform_centers),
                ("kcenter3", kcenter_centers),
                ("kmeans3", kmeans_centers),
            ):
                residual = anchor_residuals(
                    centers,
                    k_blocks,
                    assignment,
                    merged_key,
                    routing_bias,
                    scale,
                )
                anchor_values[name] = residual
                prediction, nearest_distance = nearest_residual(
                    points, centers, residual
                )
                candidates[f"{name}_query_nn"] = corrected_logits(
                    approximate_logits, prediction
                )
                quadratic_prediction = isotropic_quadratic_residual(
                    points, centers, residual, seeds
                )
                candidates[f"{name}_isotropic_quadratic"] = corrected_logits(
                    approximate_logits, quadratic_prediction
                )
                cover_stats[f"{name}_query_nn"].add(nearest_distance.sqrt())

            uniform_residual = anchor_values["uniform3"]
            flat = uniform_residual.flatten(0, -3).transpose(1, 2)
            interpolated = F.interpolate(
                flat, size=rows, mode="linear", align_corners=True
            ).transpose(1, 2)
            interpolated = interpolated.reshape(
                *uniform_residual.shape[:-2], rows, uniform_residual.size(-1)
            )
            candidates["uniform3_row_linear"] = corrected_logits(
                approximate_logits, interpolated
            )

            candidate_outputs = {}
            for name, logits in candidates.items():
                output = compressed_output(logits, merged_value)
                candidate_outputs[name] = output
                stats[name].add(output, reference)
                per_call_stats[call][name].add(output, reference)
                kl_stats[name].add(mass_kl(exact_logits, logits))
            if args.include_cumulant_probe:
                cumulant_outputs = coupled_cumulant_outputs(
                    query,
                    k_blocks,
                    v_blocks,
                    seeds,
                    assignment,
                    merged_key,
                    merged_value,
                    approximate_logits,
                    exact_logits,
                    scale,
                    row_chunk=args.row_chunk,
                )
                for name, output in cumulant_outputs.items():
                    candidate_outputs[name] = output
                    stats[name].add(output, reference)
                    per_call_stats[call][name].add(output, reference)
            if args.include_value_oracle:
                oracle_outputs = oracle_cluster_value_scale_outputs(
                    query,
                    k_blocks,
                    v_blocks,
                    assignment,
                    merged_value,
                    approximate_logits,
                    scale,
                    row_chunk=args.row_chunk,
                )
                for name, output in oracle_outputs.items():
                    candidate_outputs[name] = output
                    stats[name].add(output, reference)
                    per_call_stats[call][name].add(output, reference)
                math_reference = oracle_outputs[
                    "oracle_exact_cluster_reconstruction"
                ]
                for name, output in candidate_outputs.items():
                    math_stats[name].add(output, math_reference)
                    per_call_math_stats[call][name].add(output, math_reference)

            state["evaluated"].append(
                {"call": call, "step": step, "layer": layer, "query_frame": query_frame}
            )
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def summarize_errors(stats):
    summaries = {name: value.summary() for name, value in sorted(stats.items())}
    baseline = summaries["fixed_wide"]["mse_vs_dense"]
    for value in summaries.values():
        value["mse_ratio_vs_fixed_wide"] = value["mse_vs_dense"] / baseline
    return summaries


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
    parser.add_argument(
        "--record_calls",
        default="1,20,40,41,60,80,81,100,120,121,140,160",
    )
    parser.add_argument("--anchors", type=int, default=3)
    parser.add_argument("--row_chunk", type=int, default=5)
    parser.add_argument("--include_value_oracle", action="store_true")
    parser.add_argument("--include_cumulant_probe", action="store_true")
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.include_cumulant_probe and not args.include_value_oracle:
        raise ValueError("--include_cumulant_probe requires --include_value_oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    stats = defaultdict(ErrorStats)
    math_stats = defaultdict(ErrorStats)
    kl_stats = defaultdict(MeanStats)
    cover_stats = defaultdict(MeanStats)
    per_call_stats = defaultdict(lambda: defaultdict(ErrorStats))
    per_call_math_stats = defaultdict(lambda: defaultdict(ErrorStats))
    state = install_probe(
        stats,
        math_stats,
        kl_stats,
        cover_stats,
        per_call_stats,
        per_call_math_stats,
        args,
    )

    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    hard_c4_ratio = (3 * 4 * 80 + 2 * 45 * 4) / (2 * 45 * 80)
    conservative_ratio = hard_c4_ratio + args.anchors / 45
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
        "anchors_per_query_strip": args.anchors,
        "wall_seconds": time.perf_counter() - started,
        "errors_vs_dense": summarize_errors(stats),
        "errors_vs_exact_dense_math": (
            summarize_errors(math_stats) if args.include_value_oracle else None
        ),
        "cluster_mass_kl": {
            name: value.mean() for name, value in sorted(kl_stats.items())
        },
        "mean_query_cover_radius": {
            name: value.mean() for name, value in sorted(cover_stats.items())
        },
        "per_call_errors_vs_dense": {
            str(call): summarize_errors(values)
            for call, values in sorted(per_call_stats.items())
        },
        "per_call_errors_vs_exact_dense_math": (
            {
                str(call): summarize_errors(values)
                for call, values in sorted(per_call_math_stats.items())
            }
            if args.include_value_oracle
            else None
        ),
        "flops": {
            "hard_c4_ratio_vs_dense": hard_c4_ratio,
            "conservative_calibrated_ratio_vs_dense": conservative_ratio,
            "ratio_vs_svg_ear": conservative_ratio / args.svg_ear_density,
            "limit_ratio_vs_dense": 0.70 * args.svg_ear_density,
            "passes_limit": conservative_ratio <= 0.70 * args.svg_ear_density,
            "note": "Counts each exact log-partition anchor as a full dense query.",
        },
        "derivation": {
            "residual": "r_c(q)=logsumexp_{j in S_c}(beta q^T k_j)-hat_z_c(q)",
            "kcenter": "min_{|A|=3} max_i min_{a in A} ||q_i-q_a||_2",
            "mass_bound": "KL(p_i || p_tilde_i) <= 0.25 ||P(e_i)||_2^2",
            "trajectory": "dense",
        },
        "per_call": state["evaluated"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
