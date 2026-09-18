"""Probe a first-order value-moment correction for Hard-C4 GMA.

For one routed key cluster with seed query ``u`` and normalized merge weights
``w``, the exact conditional value has the local expansion

    E_w[V exp(beta (q-u)^T K)] / E_w[exp(beta (q-u)^T K)]
      = mean_w(V) + beta Cov_w(V, K) (q-u) + O(||q-u||^2).

The maintained merge keeps only ``mean_w(V)``.  This probe approximates the
missing covariance action along one deterministic grid-tangent shared by all
clusters in a block.  The tangent is the normalized secant between the first
and last local seed query.  This gives a paper-level first-order correction,
not a learned or schedule-specific heuristic.
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
    fixed_seed_queries,
    local_centroid_seed_queries,
)
from attention.hard_route_pytorch import FLOAT_NEG_INF
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


DEFAULT_RADII = (20,)
DEFAULT_SCALES = (0.25, 0.5, 0.75, 1.0, 1.25)


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").split()]


def parse_floats(value):
    return [float(item) for item in value.replace(",", " ").split()]


def shared_secant_directions(seeds):
    """Return one grid tangent shared by all clusters in a merge block."""

    secant = seeds[..., -1, :, :] - seeds[..., 0, :, :]
    direction = F.normalize(secant.float(), dim=-1).to(seeds.dtype)
    return direction.unsqueeze(-3).expand_as(seeds)


def shared_affine_rank2_basis(seeds):
    """Return an orthonormal basis for the affine hull of three centroids."""

    if seeds.size(-3) != 3:
        raise ValueError("the affine rank-2 basis requires exactly three seeds")
    first, second, third = seeds.float().unbind(dim=-3)
    basis_1 = F.normalize(second - first, dim=-1)
    residual = third - first
    residual = residual - (residual * basis_1).sum(dim=-1, keepdim=True) * basis_1
    basis_2 = F.normalize(residual, dim=-1)
    return torch.stack((basis_1, basis_2), dim=-3)


def one_sweep_rank1_factors(left, transpose_action):
    """Optimize one shared right factor and scalar for fixed left factors."""

    aggregate = transpose_action.float().sum(dim=-3)
    right = F.normalize(aggregate, dim=-1)
    left_energy = left.float().square().sum(dim=-1).sum(dim=-2)
    scale = torch.where(
        left_energy > 0,
        aggregate.norm(dim=-1)
        / left_energy.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(left_energy),
    )
    scaled_left = left.float() * scale[:, :, :, None, :, None]
    return scaled_left, right


@torch.inference_mode()
def tangent_moment_outputs(
    query,
    key,
    value,
    seeds,
    scale,
    query_frame,
    radii,
    correction_scales,
    include_full_oracle=False,
    include_affine_rank2=False,
    include_als_rank1=False,
):
    """Stream hard-routed merge plus low-rank moments over key blocks."""

    batch, query_outer, query_rows, query_cols, heads, dim = query.shape
    key_outer, key_rows = key.size(1), key.size(2)
    query_h = rearrange(query, "b a i j h d -> b h a i j d").float()

    accumulator = torch.zeros(
        batch,
        heads,
        query_outer,
        query_rows,
        query_cols,
        dim,
        device=query.device,
        dtype=torch.float32,
    )
    tangent_accumulators = {
        radius: torch.zeros_like(accumulator) for radius in radii
    }
    oracle_accumulator = (
        torch.zeros_like(accumulator) if include_full_oracle else None
    )
    affine_accumulator = (
        torch.zeros_like(accumulator) if include_affine_rank2 else None
    )
    als_accumulator = torch.zeros_like(accumulator) if include_als_rank1 else None
    row_max = torch.full(
        (batch, heads, query_outer, query_rows, query_cols),
        FLOAT_NEG_INF,
        device=query.device,
        dtype=torch.float32,
    )
    row_sum = torch.zeros_like(row_max)

    for key_outer_index in range(key_outer):
        temporal_distance = abs(int(key_outer_index) - int(query_frame))
        active_radii = [radius for radius in radii if temporal_distance <= radius]
        for key_row_index in range(key_rows):
            seed = seeds[:, :, key_outer_index, key_row_index]
            key_row = key[:, key_outer_index, key_row_index]
            value_row = value[:, key_outer_index, key_row_index]

            seed_logits = torch.einsum(
                "bajchd,blhd->bhajcl", seed, key_row
            ).float()
            seed_logits *= float(scale)
            assignment = torch.zeros_like(seed_logits, dtype=torch.bool)
            best_seed = torch.log_softmax(seed_logits, dim=-1).argmax(
                dim=-2, keepdim=True
            )
            assignment.scatter_(dim=-2, index=best_seed, value=True)
            routed_logits = seed_logits.masked_fill(~assignment, FLOAT_NEG_INF)
            log_weights = torch.log_softmax(routed_logits, dim=-1).masked_fill(
                ~assignment, 0.0
            )
            weights = log_weights.exp().masked_fill(~assignment, 0.0)
            bias = (weights * log_weights).sum(dim=-1)
            bias = bias.masked_fill(~assignment.any(dim=-1), float("inf"))

            merge_weights = weights.to(key.dtype)
            merged_key = torch.einsum(
                "bhajcl,blhd->bajchd", merge_weights, key_row
            )
            merged_value = torch.einsum(
                "bhajcl,blhe->bajche", merge_weights, value_row
            )

            tangent = shared_secant_directions(seed)
            key_projection = torch.einsum(
                "bajchd,blhd->bhajcl", tangent, key_row
            ).float()
            mean_projection = torch.einsum(
                "bajchd,bajchd->bajch", tangent.float(), merged_key.float()
            )
            cross_moment = torch.einsum(
                "bhajcl,blhe->bajche",
                weights * key_projection,
                value_row.float(),
            )
            tangent_value = float(scale) * (
                cross_moment
                - mean_projection[..., None] * merged_value.float()
            )

            tangent_h = rearrange(tangent.float(), "b a j c h d -> b h a j c d")
            seed_projection = rearrange(
                torch.einsum(
                    "bajchd,bajchd->bajch", seed.float(), tangent.float()
                ),
                "b a j c h -> b h a j c",
            )
            query_projection = torch.einsum(
                "bhaijd,bhajcd->bhaijc", query_h, tangent_h
            )
            displacement = query_projection - seed_projection[:, :, :, None]

            scores = torch.einsum(
                "bhaijd,bajchd->bhaijc", query_h, merged_key.float()
            )
            scores = scores * float(scale) - bias[:, :, :, None]
            next_max = torch.maximum(row_max, scores.max(dim=-1).values)
            probabilities = torch.exp(scores - next_max[..., None])
            old_scale = torch.exp(row_max - next_max)

            accumulator = accumulator * old_scale[..., None] + torch.einsum(
                "bhaijc,bajchd->bhaijd",
                probabilities,
                merged_value.float(),
            )
            tangent_value_h = rearrange(
                tangent_value, "b a j c h d -> b h a j c d"
            )
            tangent_contribution = torch.einsum(
                "bhaijc,bhaijc,bhajcd->bhaijd",
                probabilities,
                displacement,
                tangent_value_h,
            )
            if als_accumulator is not None:
                value_projection = torch.einsum(
                    "blhe,bajche->bhajcl",
                    value_row.float(),
                    tangent_value.float(),
                )
                transpose_cross_moment = torch.einsum(
                    "bhajcl,bhajcl,blhd->bajchd",
                    weights,
                    value_projection,
                    key_row.float(),
                )
                mean_value_projection = torch.einsum(
                    "bajche,bajche->bajch",
                    merged_value.float(),
                    tangent_value.float(),
                )
                transpose_action = float(scale) * (
                    transpose_cross_moment
                    - mean_value_projection[..., None] * merged_key.float()
                )
                als_left, als_right = one_sweep_rank1_factors(
                    tangent_value, transpose_action
                )
                als_query_projection = torch.einsum(
                    "bhaijd,bajhd->bhaij", query_h, als_right
                )
                als_seed_projection = torch.einsum(
                    "bajchd,bajhd->bhajc", seed.float(), als_right
                )
                als_displacement = (
                    als_query_projection[..., None]
                    - als_seed_projection.unsqueeze(3)
                )
                als_left_h = rearrange(
                    als_left, "b a j c h d -> b h a j c d"
                )
                als_accumulator.mul_(old_scale[..., None]).add_(
                    torch.einsum(
                        "bhaijc,bhaijc,bhajcd->bhaijd",
                        probabilities,
                        als_displacement,
                        als_left_h,
                    )
                )
            if affine_accumulator is not None:
                basis = shared_affine_rank2_basis(seed)
                basis_key_projection = torch.einsum(
                    "bajrhd,blhd->bhajrl", basis, key_row.float()
                )
                affine_cross_moment = torch.einsum(
                    "bhajcl,bhajrl,blhe->bajcrhe",
                    weights,
                    basis_key_projection,
                    value_row.float(),
                )
                affine_mean_projection = torch.einsum(
                    "bajchd,bajrhd->bajcrh", merged_key.float(), basis.float()
                )
                affine_value = float(scale) * (
                    affine_cross_moment
                    - affine_mean_projection[..., None]
                    * merged_value.float().unsqueeze(-3)
                )
                basis_h = rearrange(basis.float(), "b a j r h d -> b h a j r d")
                affine_query_projection = torch.einsum(
                    "bhaijd,bhajrd->bhaijr", query_h, basis_h
                )
                affine_seed_projection = torch.einsum(
                    "bajchd,bajrhd->bhajcr", seed.float(), basis.float()
                )
                affine_displacement = (
                    affine_query_projection.unsqueeze(-2)
                    - affine_seed_projection.unsqueeze(3)
                )
                affine_value_h = rearrange(
                    affine_value, "b a j c r h e -> b h a j c r e"
                )
                affine_accumulator.mul_(old_scale[..., None]).add_(
                    torch.einsum(
                        "bhaijc,bhaijcr,bhajcre->bhaije",
                        probabilities,
                        affine_displacement,
                        affine_value_h,
                    )
                )
            if oracle_accumulator is not None:
                seed_h = rearrange(seed.float(), "b a j c h d -> b h a j c d")
                delta = query_h.unsqueeze(-2) - seed_h.unsqueeze(3)
                key_delta_projection = torch.einsum(
                    "bhaijcd,blhd->bhaijcl", delta, key_row.float()
                )
                cross_moment_delta = torch.einsum(
                    "bhajcl,bhaijcl,blhe->bhaijce",
                    weights,
                    key_delta_projection,
                    value_row.float(),
                )
                merged_key_h = rearrange(
                    merged_key.float(), "b a j c h d -> b h a j c d"
                )
                merged_value_h = rearrange(
                    merged_value.float(), "b a j c h d -> b h a j c d"
                )
                mean_key_projection = torch.einsum(
                    "bhaijcd,bhajcd->bhaijc", delta, merged_key_h
                )
                full_first_order_value = float(scale) * (
                    cross_moment_delta
                    - mean_key_projection[..., None]
                    * merged_value_h.unsqueeze(3)
                )
                oracle_accumulator.mul_(old_scale[..., None]).add_(
                    torch.einsum(
                        "bhaijc,bhaijcd->bhaijd",
                        probabilities,
                        full_first_order_value,
                    )
                )
            for radius, tangent_accumulator in tangent_accumulators.items():
                tangent_accumulator.mul_(old_scale[..., None])
                if radius in active_radii:
                    tangent_accumulator.add_(tangent_contribution)
            row_sum = row_sum * old_scale + probabilities.sum(dim=-1)
            row_max = next_max

    base = accumulator / row_sum[..., None]
    outputs = {"fixed_wide_streaming": base}
    for radius in radii:
        correction = tangent_accumulators[radius] / row_sum[..., None]
        for correction_scale in correction_scales:
            name = f"tangent_r{radius}_s{correction_scale:g}"
            outputs[name] = base + correction * float(correction_scale)
    if oracle_accumulator is not None:
        outputs["full_first_order_oracle"] = (
            base + oracle_accumulator / row_sum[..., None]
        )
    if affine_accumulator is not None:
        outputs["affine_rank2"] = base + affine_accumulator / row_sum[..., None]
    if als_accumulator is not None:
        outputs["als_rank1"] = base + als_accumulator / row_sum[..., None]
    return {
        name: rearrange(output, "b h a i j d -> b a i j h d").to(query.dtype)
        for name, output in outputs.items()
    }


def temporal_pair_fraction(frames, radius):
    radius = min(int(radius), frames - 1)
    return (frames * (2 * radius + 1) - radius * (radius + 1)) / frames**2


def theoretical_flops(svg_density, radii):
    frames, rows, width, clusters = 21, 45, 80, 4
    base = (3 * clusters * width + 2 * rows * clusters) / (2 * rows * width)
    # One shared tangent lets the key projection be reused across clusters.
    # Hard ownership also makes the cross-moment accumulation a segmented
    # reduction over W keys rather than a dense C-by-W contraction.
    full_tangent_extra = (2 * width + rows + rows * clusters) / (
        2 * rows * width
    )
    direction_overhead = (3 + clusters) / (2 * frames * rows * width)
    limit = 0.70 * float(svg_density)
    candidates = {}
    for radius in radii:
        fraction = temporal_pair_fraction(frames, radius)
        ratio = base + fraction * full_tangent_extra + direction_overhead
        candidates[str(radius)] = {
            "temporal_pair_fraction": fraction,
            "ratio_vs_dense": ratio,
            "ratio_vs_svg_ear": ratio / float(svg_density),
            "passes_limit": ratio <= limit,
        }
    return {
        "hard_c4_ratio_vs_dense": base,
        "full_tangent_extra_ratio_vs_dense": full_tangent_extra,
        "direction_overhead_ratio_vs_dense": direction_overhead,
        "limit_ratio_vs_dense": limit,
        "radii": candidates,
    }


def cluster_rank_flops(svg_density, clusters, rank):
    frames, rows, width = 21, 45, 80
    denominator = 2 * rows * width
    base = (3 * clusters * width + 2 * rows * clusters) / denominator
    per_rank = (2 * width + rows + rows * clusters) / denominator
    basis_overhead = rank * (3 + clusters) / (2 * frames * rows * width)
    ratio = base + rank * per_rank + basis_overhead
    return {
        "clusters": clusters,
        "rank": rank,
        "base_ratio_vs_dense": base,
        "per_rank_ratio_vs_dense": per_rank,
        "basis_overhead_ratio_vs_dense": basis_overhead,
        "ratio_vs_dense": ratio,
        "ratio_vs_svg_ear": ratio / float(svg_density),
        "limit_ratio_vs_dense": 0.70 * float(svg_density),
        "passes_limit": ratio <= 0.70 * float(svg_density),
    }


def als_rank1_flops(svg_density, clusters=4):
    rows, width = 45, 80
    denominator = 2 * rows * width
    base = (3 * clusters * width + 2 * rows * clusters) / denominator
    correction = (4 * width + rows + rows * clusters) / denominator
    ratio = base + correction
    return {
        "clusters": clusters,
        "rank": 1,
        "base_ratio_vs_dense": base,
        "correction_ratio_vs_dense": correction,
        "ratio_vs_dense": ratio,
        "ratio_vs_svg_ear": ratio / float(svg_density),
        "limit_ratio_vs_dense": 0.70 * float(svg_density),
        "passes_limit": ratio <= 0.70 * float(svg_density),
    }


def install_probe(stats, per_call_stats, per_head_stats, per_call_head_stats, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    radii = parse_ints(args.radii)
    correction_scales = parse_floats(args.correction_scales)
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
            if args.seed_strategy == "fixed":
                seeds = fixed_seed_queries(query, frames, rows)
            else:
                clusters = (
                    3 if args.seed_strategy == "local_c3_norm_p11_s1" else 4
                )
                seeds = local_centroid_seed_queries(
                    query,
                    frames,
                    rows,
                    candidate_count=11,
                    steps=1,
                    mode="euclidean",
                    normalize=True,
                    clusters=clusters,
                )
            outputs = tangent_moment_outputs(
                query,
                k_blocks,
                v_blocks,
                seeds,
                dim**-0.5,
                query_frame,
                radii,
                correction_scales,
                args.include_full_oracle,
                args.include_affine_rank2,
                args.include_als_rank1,
            )
            for name, output in outputs.items():
                stats[name].add(output, reference)
                per_call_stats[call][name].add(output, reference)
                for position, head in enumerate(selected_heads):
                    head_output = output[..., position : position + 1, :]
                    head_reference = reference[..., position : position + 1, :]
                    per_head_stats[head][name].add(head_output, head_reference)
                    per_call_head_stats[call][head][name].add(
                        head_output, head_reference
                    )
            state["evaluated"].append(
                {"call": call, "step": step, "layer": layer, "query_frame": query_frame}
            )
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


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
    parser.add_argument("--heads", default="0")
    parser.add_argument("--record_calls", default="1")
    parser.add_argument("--radii", default=",".join(map(str, DEFAULT_RADII)))
    parser.add_argument(
        "--correction_scales", default=",".join(map(str, DEFAULT_SCALES))
    )
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument("--include_full_oracle", action="store_true")
    parser.add_argument("--include_affine_rank2", action="store_true")
    parser.add_argument("--include_als_rank1", action="store_true")
    parser.add_argument(
        "--seed_strategy",
        choices=("fixed", "local_norm_p11_s1", "local_c3_norm_p11_s1"),
        default="fixed",
    )
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
    state = install_probe(
        stats, per_call_stats, per_head_stats, per_call_head_stats, args
    )
    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: value.summary() for name, value in sorted(stats.items())}
    baseline = errors["fixed_wide_streaming"]["mse_vs_dense"]
    for value in errors.values():
        value["mse_ratio_vs_fixed_wide"] = value["mse_vs_dense"] / baseline
    per_call_errors = {}
    for call, values in sorted(per_call_stats.items()):
        summaries = {
            name: value.summary() for name, value in sorted(values.items())
        }
        call_baseline = summaries["fixed_wide_streaming"]["mse_vs_dense"]
        for value in summaries.values():
            value["mse_ratio_vs_fixed_wide"] = (
                value["mse_vs_dense"] / call_baseline
            )
        per_call_errors[str(call)] = summaries

    per_head_errors = {}
    for head, values in sorted(per_head_stats.items()):
        summaries = {
            name: value.summary() for name, value in sorted(values.items())
        }
        head_baseline = summaries["fixed_wide_streaming"]["mse_vs_dense"]
        for value in summaries.values():
            value["mse_ratio_vs_fixed_wide"] = (
                value["mse_vs_dense"] / head_baseline
            )
        per_head_errors[str(head)] = summaries

    per_call_head_errors = {}
    for call, heads in sorted(per_call_head_stats.items()):
        per_call_head_errors[str(call)] = {}
        for head, values in sorted(heads.items()):
            summaries = {
                name: value.summary() for name, value in sorted(values.items())
            }
            baseline = summaries["fixed_wide_streaming"]["mse_vs_dense"]
            for value in summaries.values():
                value["mse_ratio_vs_fixed_wide"] = (
                    value["mse_vs_dense"] / baseline
                )
            per_call_head_errors[str(call)][str(head)] = summaries

    radii = parse_ints(args.radii)
    candidate_clusters = (
        3 if args.seed_strategy == "local_c3_norm_p11_s1" else 4
    )
    candidate_rank = 2 if args.include_affine_rank2 else 1
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
        "radii": radii,
        "correction_scales": parse_floats(args.correction_scales),
        "include_full_oracle": args.include_full_oracle,
        "include_affine_rank2": args.include_affine_rank2,
        "include_als_rank1": args.include_als_rank1,
        "seed_strategy": args.seed_strategy,
        "wall_seconds": time.perf_counter() - started,
        "errors_vs_dense": errors,
        "per_call_errors_vs_dense": per_call_errors,
        "per_head_errors_vs_dense": per_head_errors,
        "per_call_head_errors_vs_dense": per_call_head_errors,
        "flops": theoretical_flops(args.svg_ear_density, radii),
        "cluster_rank_flops": cluster_rank_flops(
            args.svg_ear_density, candidate_clusters, candidate_rank
        ),
        "als_rank1_flops": als_rank1_flops(
            args.svg_ear_density, candidate_clusters
        ),
        "derivation": {
            "conditional_value": "mu_v + beta Cov_w(V,K)(q-u) + O(||q-u||^2)",
            "rank1": "Cov_w(V,K)(q-u) ~= Cov_w(V,K)b * b^T(q-u)",
            "rank2": "Cov_w(V,K)(q-u) ~= Cov_w(V,K)B B^T(q-u)",
            "direction": "one normalized endpoint secant shared by all clusters in a block",
            "affine_basis": "orthonormal basis of the three-centroid affine hull",
            "als_rank1": "one closed-form right-factor and scalar update",
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
