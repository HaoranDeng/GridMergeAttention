"""Screen moment-corrected Hard-C4 on native StepDistill Wan QKV.

For a hard cluster with seed distribution ``w`` and seed query ``u``, write
``delta = q - u``.  The exact cluster log mass differs from GMA's
entropy-corrected first-order logit by the centered cumulant

    log E_w exp(scale * delta^T (k - E_w[k])).

This probe tests its second-order approximation while keeping the dense Wan
trajectory unchanged.  Key covariance is approximated as isotropic inside
small channel groups, including the temporal/height/width RoPE subspaces.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from attention.hard_route_pytorch import FLOAT_NEG_INF, gather_seed_queries, wide_local_c4_seed_rows
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


MASS_SCALES = (0.25, 0.5, 1.0)
GROUP_SCHEMES = ("global", "rope3", "g8", "g16")
VALUE_SCHEMES = ("rope3", "g8")
VALUE_SCALES = (0.25, 0.5)


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


def scale_label(value):
    return str(value).replace(".", "p")


def channel_groups(dim, scheme):
    if scheme == "global":
        return ((0, dim),)
    if scheme == "rope3":
        unit = dim // 6
        temporal = dim - 4 * unit
        return ((0, temporal), (temporal, temporal + 2 * unit), (temporal + 2 * unit, dim))
    if scheme.startswith("g"):
        count = int(scheme[1:])
        if dim % count:
            raise ValueError(f"head dimension {dim} is not divisible by {count}")
        width = dim // count
        return tuple((index * width, (index + 1) * width) for index in range(count))
    raise ValueError(f"unknown channel grouping {scheme!r}")


def hard_route_merge_with_moments(seed_queries, key, value, scale):
    seed_key_logits = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seed_queries, key
    ).float()
    seed_key_logits.mul_(float(scale))
    assignment_scores = torch.log_softmax(seed_key_logits, dim=-1)
    best_seed = assignment_scores.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)

    routed_logits = seed_key_logits.masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(routed_logits, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(
        ~assignment.any(dim=-1), float("inf")
    )

    merge_weights = weights.to(key.dtype)
    merged_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", merge_weights, key
    )
    merged_value = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", merge_weights, value
    )
    mean_key_square = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", weights, key.float().square()
    )
    mean_value_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd",
        weights,
        value.float() * key.float(),
    )
    diagonal_key_variance = (
        mean_key_square - merged_key.float().square()
    ).clamp_min_(0.0)
    diagonal_value_key_covariance = (
        mean_value_key - merged_value.float() * merged_key.float()
    )
    return (
        merged_key,
        merged_value,
        routing_bias,
        diagonal_key_variance,
        diagonal_value_key_covariance,
    )


def flatten_route_tensors(
    query,
    seed_queries,
    merged_key,
    merged_value,
    routing_bias,
    diagonal_key_variance,
    diagonal_value_key_covariance,
):
    return {
        "query": rearrange(query, "b a i j h d -> b h a j i d").float(),
        "seed": rearrange(
            seed_queries, "b a f k j c h d -> b h a j (f k c) d"
        ).float(),
        "key": rearrange(
            merged_key, "b a f j k c h d -> b h a j (f k c) d"
        ).float(),
        "value": rearrange(
            merged_value, "b a f j k c h d -> b h a j (f k c) d"
        ).float(),
        "bias": rearrange(
            routing_bias, "b h a f k j c -> b h a j (f k c)"
        ).float(),
        "key_variance": rearrange(
            diagonal_key_variance,
            "b a f j k c h d -> b h a j (f k c) d",
        ),
        "value_key_covariance": rearrange(
            diagonal_value_key_covariance,
            "b a f j k c h d -> b h a j (f k c) d",
        ),
    }


def grouped_cumulant_correction(route, groups, scale):
    query = route["query"]
    seed = route["seed"]
    variance = route["key_variance"]
    correction = torch.zeros(
        (*query.shape[:-1], seed.size(-2)),
        device=query.device,
        dtype=torch.float32,
    )
    for start, end in groups:
        query_group = query[..., start:end]
        seed_group = seed[..., start:end]
        variance_group = variance[..., start:end].mean(dim=-1)
        distance_square = (
            query_group.square().sum(dim=-1, keepdim=True)
            + seed_group.square().sum(dim=-1).unsqueeze(-2)
            - 2.0
            * torch.einsum(
                "bhajid,bhajtd->bhajit", query_group, seed_group
            )
        ).clamp_min_(0.0)
        correction.add_(distance_square * variance_group.unsqueeze(-2))
    return correction * (0.5 * float(scale) ** 2)


def compressed_output(route, logits, groups=None, value_scale=0.0, scale=1.0):
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.einsum(
        "bhajit,bhajtd->bhajid", probabilities, route["value"]
    )
    if value_scale:
        query = route["query"]
        seed = route["seed"]
        covariance = route["value_key_covariance"]
        for start, end in groups:
            alpha = covariance[..., start:end].mean(dim=-1)
            weighted_probability = probabilities * alpha.unsqueeze(-2)
            coefficient = weighted_probability.sum(dim=-1, keepdim=True)
            weighted_seed = torch.einsum(
                "bhajit,bhajtd->bhajid",
                weighted_probability,
                seed[..., start:end],
            )
            output[..., start:end].add_(
                coefficient * query[..., start:end] - weighted_seed,
                alpha=float(value_scale) * float(scale),
            )
    return rearrange(output, "b h a j i d -> b a i j h d")


def evaluate_candidates(route, reference, scale, stats):
    base_logits = (
        torch.einsum(
            "bhajid,bhajtd->bhajit", route["query"], route["key"]
        )
        * float(scale)
        - route["bias"].unsqueeze(-2)
    )
    stats["hard_c4"].add(compressed_output(route, base_logits), reference)
    corrections = {}
    for scheme in GROUP_SCHEMES:
        groups = channel_groups(route["query"].size(-1), scheme)
        correction = grouped_cumulant_correction(route, groups, scale)
        corrections[scheme] = correction
        for mass_scale in MASS_SCALES:
            name = f"{scheme}_mass_{scale_label(mass_scale)}"
            stats[name].add(
                compressed_output(
                    route, base_logits + float(mass_scale) * correction
                ),
                reference,
            )

    for scheme in VALUE_SCHEMES:
        groups = channel_groups(route["query"].size(-1), scheme)
        for value_scale in VALUE_SCALES:
            name = f"{scheme}_mass_0p5_value_{scale_label(value_scale)}"
            stats[name].add(
                compressed_output(
                    route,
                    base_logits + 0.5 * corrections[scheme],
                    groups=groups,
                    value_scale=value_scale,
                    scale=scale,
                ),
                reference,
            )


def install_probe(stats, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    state = {"calls_total": 0, "grid_size": None, "evaluated": []}

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len, heads, dim = *x.shape[:2], self.num_heads, self.head_dim
        q = wan_model.rmsnorm(
            self.q(x), self.norm_q.weight, eps=self.norm_q.eps
        ).view(batch, seq_len, heads, dim)
        k = wan_model.rmsnorm(
            self.k(x), self.norm_k.weight, eps=self.norm_k.eps
        ).view(batch, seq_len, heads, dim)
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        dense = wan_model.flash_attention(
            q=q, k=k, v=v, window_size=self.window_size
        )

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
            seed_rows = wide_local_c4_seed_rows(
                torch.arange(query_frame, query_frame + 1, device=q.device),
                torch.arange(frames, device=q.device),
                rows,
                width,
                rows,
            )
            seed_queries = gather_seed_queries(query, seed_rows)
            scale = dim**-0.5
            moments = hard_route_merge_with_moments(
                seed_queries, k_blocks, v_blocks, scale
            )
            route = flatten_route_tensors(
                query, seed_queries, *moments
            )
            evaluate_candidates(route, reference, scale, stats)
            state["evaluated"].append(
                {
                    "call": call,
                    "step": step,
                    "layer": layer,
                    "query_frame": query_frame,
                }
            )

        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def flops(svg_density, groups, include_value):
    frames, rows, width, clusters, dim = 21, 45, 80, 4, 128
    base = (3 * clusters * width + 2 * rows * clusters) / (2 * rows * width)
    route_moment = clusters / (2 * width)
    scalar_merge = clusters * groups / (2 * rows * dim)
    value = route_moment + scalar_merge if include_value else 0.0
    total = base + route_moment + scalar_merge + value
    return {
        "ratio_vs_dense": total,
        "ratio_vs_svg_ear": total / svg_density,
        "passes_70_percent_svg_ear_limit": total <= 0.70 * svg_density,
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
    state = install_probe(stats, args)
    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: value.summary() for name, value in sorted(stats.items())}
    baseline = errors["hard_c4"]["mse_vs_dense"]
    for value in errors.values():
        value["mse_ratio_vs_hard_c4"] = value["mse_vs_dense"] / baseline
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
        "flops": {
            scheme: {
                "mass_only": flops(
                    args.svg_ear_density,
                    len(channel_groups(128, scheme)),
                    False,
                ),
                "mass_and_value": flops(
                    args.svg_ear_density,
                    len(channel_groups(128, scheme)),
                    True,
                ),
            }
            for scheme in GROUP_SCHEMES
        },
        "derivation": {
            "exact_gap": "log E_w exp(beta * delta^T (k - E_w[k]))",
            "second_order": "0.5 * beta^2 * delta^T Cov_w(k) delta",
            "value_first_order": "E_w[v] + beta * Cov_w(v,k) delta",
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
