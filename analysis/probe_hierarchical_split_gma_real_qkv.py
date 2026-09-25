"""Probe monotone binary refinement of Hard-C4 clusters on native Wan QKV."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_query_centroid_gma_real_qkv import (
    ErrorStats,
    fixed_seed_queries,
    parse_ints,
)
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


def _conditional_child(weights, log_weights, mask, key, value):
    mass = (weights * mask).sum(dim=-1)
    conditional = weights * mask / mass.unsqueeze(-1).clamp_min(1e-20)
    conditional_log_weight = torch.where(
        mask,
        log_weights - mass.clamp_min(1e-20).log().unsqueeze(-1),
        0.0,
    )
    bias = (conditional * conditional_log_weight).sum(dim=-1)
    bias = bias.masked_fill(mass <= 0, float("inf"))
    merged_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", conditional.to(key.dtype), key
    )
    merged_value = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", conditional.to(value.dtype), value
    )
    return merged_key, merged_value, bias, mass, conditional


def hard_c4_binary_children(seeds, key, value, scale, split_mode="median"):
    """Return parent moments and a balanced tangent split of every cluster."""

    seed_scores = torch.einsum(
        "bafkjchd,bfklhd->bhafkjcl", seeds, key
    ).float() * float(scale)
    assignment = torch.zeros_like(seed_scores, dtype=torch.bool)
    best_seed = torch.log_softmax(seed_scores, dim=-1).argmax(
        dim=-2, keepdim=True
    )
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    routed_scores = seed_scores.masked_fill(~assignment, float("-inf"))
    log_weights = torch.log_softmax(routed_scores, dim=-1).masked_fill(
        ~assignment, 0.0
    )
    weights = log_weights.exp().masked_fill(~assignment, 0.0)

    parent_key, parent_value, parent_bias, _, _ = _conditional_child(
        weights,
        log_weights,
        assignment,
        key,
        value,
    )

    clusters = seed_scores.size(-2)
    cluster_index = torch.arange(clusters, device=seed_scores.device)
    left_index = (cluster_index - 1).clamp_min(0)
    right_index = (cluster_index + 1).clamp_max(clusters - 1)
    tangent_score = seed_scores.index_select(-2, right_index) - seed_scores.index_select(
        -2, left_index
    )

    if split_mode == "median":
        order = tangent_score.masked_fill(~assignment, float("inf")).argsort(dim=-1)
        sorted_weight = weights.gather(dim=-1, index=order)
        cut = (sorted_weight.cumsum(dim=-1) < 0.5).sum(dim=-1)
        cut = cut.clamp_max(seed_scores.size(-1) - 1)
        rank = torch.empty_like(order)
        positions = torch.arange(seed_scores.size(-1), device=seed_scores.device)
        rank.scatter_(dim=-1, index=order, src=positions.expand_as(order))
        low_mask = assignment & (rank <= cut.unsqueeze(-1))
    elif split_mode == "mean":
        threshold = (weights * tangent_score).sum(dim=-1)
        low_mask = assignment & (tangent_score <= threshold.unsqueeze(-1))
    else:
        raise ValueError(f"unknown split mode {split_mode!r}")
    high_mask = assignment & ~low_mask

    low_key, low_value, low_bias, low_mass, low_weight = _conditional_child(
        weights, log_weights, low_mask, key, value
    )
    high_key, high_value, high_bias, high_mass, high_weight = _conditional_child(
        weights, log_weights, high_mask, key, value
    )
    low_mean = (low_weight * tangent_score).sum(dim=-1)
    high_mean = (high_weight * tangent_score).sum(dim=-1)
    split_gain = low_mass * high_mass * (high_mean - low_mean).square()
    split_gain = split_gain.masked_fill(
        (low_mass <= 0) | (high_mass <= 0), float("-inf")
    )
    return {
        "parent_key": parent_key,
        "parent_value": parent_value,
        "parent_bias": parent_bias,
        "low_key": low_key,
        "low_value": low_value,
        "low_bias": low_bias,
        "high_key": high_key,
        "high_value": high_value,
        "high_bias": high_bias,
        "split_gain": split_gain,
    }


def refined_representatives(children, refinements):
    """Replace the selected parent token by its two conditional children."""

    parent_key = rearrange(
        children["parent_key"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    parent_value = rearrange(
        children["parent_value"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    low_key = rearrange(
        children["low_key"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    low_value = rearrange(
        children["low_value"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    high_key = rearrange(
        children["high_key"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    high_value = rearrange(
        children["high_value"], "b a f j k c h d -> b h a f k j c d"
    ).float()

    selected_index = children["split_gain"].topk(int(refinements), dim=-1).indices
    selected = torch.zeros_like(children["split_gain"], dtype=torch.bool)
    selected.scatter_(dim=-1, index=selected_index, value=True)
    base_key = torch.where(selected.unsqueeze(-1), low_key, parent_key)
    base_value = torch.where(selected.unsqueeze(-1), low_value, parent_value)
    base_bias = torch.where(
        selected, children["low_bias"], children["parent_bias"]
    )

    gather_index = selected_index.unsqueeze(-1).expand(
        *selected_index.shape, high_key.size(-1)
    )
    extra_key = high_key.gather(dim=-2, index=gather_index)
    extra_value = high_value.gather(dim=-2, index=gather_index)
    extra_bias = children["high_bias"].gather(dim=-1, index=selected_index)
    return (
        torch.cat((base_key, extra_key), dim=-2),
        torch.cat((base_value, extra_value), dim=-2),
        torch.cat((base_bias, extra_bias), dim=-1),
    )


def output_from_representatives(query, key, value, bias, scale):
    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    key_flat = rearrange(key, "b h a f k j c d -> b h a j (f k c) d")
    value_flat = rearrange(value, "b h a f k j c d -> b h a j (f k c) d")
    bias_flat = rearrange(bias, "b h a f k j c -> b h a j (f k c)")
    logits = torch.einsum("bhajid,bhajtd->bhajit", query_h, key_flat)
    probability = torch.softmax(logits * float(scale) - bias_flat.unsqueeze(-2), dim=-1)
    output = torch.einsum("bhajit,bhajtd->bhajid", probability, value_flat)
    return rearrange(output, "b h a j i d -> b a i j h d").to(query.dtype)


def hierarchical_split_outputs(query, key, value, seeds, scale, refinements=(1, 2, 4)):
    children = hard_c4_binary_children(seeds, key, value, scale, split_mode="median")
    parent_key = rearrange(
        children["parent_key"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    parent_value = rearrange(
        children["parent_value"], "b a f j k c h d -> b h a f k j c d"
    ).float()
    outputs = {
        "fixed_wide": output_from_representatives(
            query, parent_key, parent_value, children["parent_bias"], scale
        )
    }
    for count in refinements:
        representatives = refined_representatives(children, count)
        outputs[f"tangent_median_split_r{count}"] = output_from_representatives(
            query, *representatives, scale
        )
    mean_children = hard_c4_binary_children(
        seeds, key, value, scale, split_mode="mean"
    )
    for count in refinements:
        representatives = refined_representatives(mean_children, count)
        outputs[f"tangent_mean_split_r{count}"] = output_from_representatives(
            query, *representatives, scale
        )
    return outputs


def install_probe(stats, per_call_stats, per_head_stats, per_call_head_stats, args):
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
            candidates = hierarchical_split_outputs(
                query, k_blocks, v_blocks, seeds, scale
            )
            for name, output in candidates.items():
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


def summarize(values):
    result = {name: value.summary() for name, value in sorted(values.items())}
    baseline = result["fixed_wide"]["mse_vs_dense"]
    for value in result.values():
        value["mse_ratio_vs_fixed_wide"] = value["mse_vs_dense"] / baseline
    return result


def theoretical_flops(svg_density):
    rows, width, clusters = 45, 80, 4
    denominator = 2 * rows * width
    methods = {}
    for refinements in (1, 2, 4):
        representatives = clusters + refinements
        ratio = (
            clusters * width
            + 2 * representatives * width
            + 2 * rows * representatives
        ) / denominator
        methods[f"tangent_median_split_r{refinements}"] = {
            "representatives": representatives,
            "ratio_vs_dense": ratio,
            "ratio_vs_svg_ear": ratio / svg_density,
            "passes_limit": ratio <= 0.70 * svg_density,
        }
        methods[f"tangent_mean_split_r{refinements}"] = {
            "representatives": representatives,
            "ratio_vs_dense": ratio,
            "ratio_vs_svg_ear": ratio / svg_density,
            "passes_limit": ratio <= 0.70 * svg_density,
        }
    return {
        "svg_ear_density": svg_density,
        "limit_ratio_vs_dense": 0.70 * svg_density,
        "methods": methods,
        "note": "Scalar weighted-median selection work is lower order in head dimension.",
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
    parser.add_argument(
        "--record_calls",
        default="1,20,40,41,60,80,81,100,120,121,140,160",
    )
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
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
        "errors_vs_dense": summarize(stats),
        "per_call_errors_vs_dense": {
            str(call): summarize(values)
            for call, values in sorted(per_call_stats.items())
        },
        "per_head_errors_vs_dense": {
            str(head): summarize(values)
            for head, values in sorted(per_head_stats.items())
        },
        "per_call_head_errors_vs_dense": {
            str(call): {
                str(head): summarize(values)
                for head, values in sorted(heads.items())
            }
            for call, heads in sorted(per_call_head_stats.items())
        },
        "flops": theoretical_flops(args.svg_ear_density),
        "derivation": {
            "mass_order": "Z_parent_hat(q) <= Z_refined_hat(q) <= Z_exact(q)",
            "value_identity": "Cov(V,K)=E_s Cov(V,K|s)+Cov_s(E[V|s],E[K|s])",
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
