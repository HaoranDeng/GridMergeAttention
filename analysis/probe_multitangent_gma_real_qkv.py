"""Probe a fixed-partition, multi-tangent GMA approximation on native Wan QKV.

For each hard-routed key cluster, ordinary GMA stores the tangent of its
log-partition at the routing seed.  This probe adds one tangent anchored at a
different member of the existing C4 seed coreset while preserving key
ownership.  The pointwise maximum of the two tangents is a monotonically
tighter lower bound to the convex cluster log-partition.  The value
representative is selected from the same winning anchor.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_hierarchical_split_gma_real_qkv import summarize
from analysis.probe_query_centroid_gma_real_qkv import (
    ErrorStats,
    fixed_assignment_merge_kv,
    fixed_seed_queries,
    hard_seed_assignment,
    parse_ints,
)
from attention.hard_c4_wan21_14b_720p_triton import (
    hard_c4_720p_attention_triton_blocks,
)
from attention.token_merging import (
    _hard_c4_blocks_to_video,
    _hard_c4_video_to_blocks,
)
from generate import load_prompts


CLUSTERS = 4


def all_seed_anchor_representatives(seeds, key, value, assignment, scale):
    """Return fixed-partition representatives at every existing C4 seed."""

    keys = []
    values = []
    biases = []
    for anchor in range(CLUSTERS):
        expansion_seed = seeds[..., anchor : anchor + 1, :, :].expand_as(seeds)
        merged_key, merged_value, bias = fixed_assignment_merge_kv(
            expansion_seed, key, value, assignment, scale
        )
        keys.append(merged_key)
        values.append(merged_value)
        biases.append(bias)
    return (
        torch.stack(keys, dim=6),
        torch.stack(values, dim=6),
        torch.stack(biases, dim=7),
    )


def diagonal_representatives(key_all, value_all, bias_all):
    """Gather the ordinary representative anchored at its own routing seed."""

    key_h = rearrange(
        key_all, "b a f j k c s h d -> b h a f k j c s d"
    ).float()
    value_h = rearrange(
        value_all, "b a f j k c s h d -> b h a f k j c s d"
    ).float()
    primary_key = torch.stack(
        [key_h[..., cluster, cluster, :] for cluster in range(CLUSTERS)],
        dim=-2,
    )
    primary_value = torch.stack(
        [value_h[..., cluster, cluster, :] for cluster in range(CLUSTERS)],
        dim=-2,
    )
    primary_bias = torch.stack(
        [bias_all[..., cluster, cluster] for cluster in range(CLUSTERS)],
        dim=-1,
    ).float()
    return key_h, value_h, primary_key, primary_value, primary_bias


def gather_anchor(tensor, anchor_index):
    """Gather a head- and block-specific anchor from the penultimate axis."""

    gather_index = anchor_index.unsqueeze(-1).unsqueeze(-1).expand(
        *anchor_index.shape, 1, tensor.size(-1)
    )
    return tensor.gather(dim=-2, index=gather_index).squeeze(-2)


def synthetic_centroid_anchors(seeds):
    """Return cheap complementary centroids without changing key ownership."""

    global_mean = seeds.float().mean(dim=5, keepdim=True).to(seeds.dtype)
    leave_one_out = (
        (seeds.float().sum(dim=5, keepdim=True) - seeds.float())
        / float(CLUSTERS - 1)
    ).to(seeds.dtype)
    lower = seeds[..., :2, :, :].float().mean(dim=5, keepdim=True)
    upper = seeds[..., 2:, :, :].float().mean(dim=5, keepdim=True)
    opposite_pair = torch.cat(
        (
            upper.expand(*upper.shape[:5], 2, *upper.shape[6:]),
            lower.expand(*lower.shape[:5], 2, *lower.shape[6:]),
        ),
        dim=5,
    ).to(seeds.dtype)
    return {
        "global_seed_centroid": global_mean.expand_as(seeds),
        "leave_one_out_seed_centroid": leave_one_out,
        "opposite_pair_seed_centroid": opposite_pair,
    }


def selected_value_output(logits, primary_value, alternate_value, use_alternate):
    probability = torch.softmax(
        rearrange(logits, "b h a f k j i c -> b h a j i (f k c)"),
        dim=-1,
    )
    primary = primary_value.unsqueeze(-3).expand(
        *primary_value.shape[:-2], logits.size(-2), *primary_value.shape[-2:]
    )
    alternate = alternate_value.unsqueeze(-3).expand_as(primary)
    selected = torch.where(use_alternate.unsqueeze(-1), alternate, primary)
    selected = rearrange(
        selected,
        "b h a f k j i c d -> b h a j i (f k c) d",
    )
    output = torch.einsum("bhajit,bhajitd->bhajid", probability, selected)
    return rearrange(output, "b h a j i d -> b a i j h d")


def multitangent_outputs(query, key, value, seeds, scale):
    """Construct the primary and residual-selected tangent envelopes."""

    assignment = hard_seed_assignment(seeds, key, scale)
    key_all, value_all, bias_all = all_seed_anchor_representatives(
        seeds, key, value, assignment, scale
    )
    (
        key_all_h,
        value_all_h,
        primary_key,
        primary_value,
        primary_bias,
    ) = diagonal_representatives(key_all, value_all, bias_all)

    seeds_h = rearrange(
        seeds, "b a f k j s h d -> b h a f k j s d"
    ).float()
    exact_at_seed = torch.einsum(
        "bhafkjsd,bhafkjcsd->bhafkjcs", seeds_h, key_all_h
    ) * float(scale) - bias_all.float()
    primary_at_seed = torch.einsum(
        "bhafkjsd,bhafkjcd->bhafkjcs", seeds_h, primary_key
    ) * float(scale) - primary_bias.unsqueeze(-1)
    tangent_gap = exact_at_seed - primary_at_seed
    diagonal = torch.eye(
        CLUSTERS, dtype=torch.bool, device=query.device
    ).view(*([1] * (tangent_gap.ndim - 2)), CLUSTERS, CLUSTERS)
    active = torch.isfinite(primary_bias)
    finite_gap = torch.where(
        torch.isfinite(tangent_gap), tangent_gap, float("-inf")
    )
    tangent_gap = torch.where(
        active.unsqueeze(-1) & ~diagonal,
        finite_gap,
        float("-inf"),
    )
    gain, alternate_anchor = tangent_gap.max(dim=-1)
    alternate_key = gather_anchor(key_all_h, alternate_anchor)
    alternate_value = gather_anchor(value_all_h, alternate_anchor)
    alternate_bias = bias_all.float().gather(
        dim=-1, index=alternate_anchor.unsqueeze(-1)
    ).squeeze(-1)

    query_h = rearrange(query, "b a i j h d -> b h a j i d").float()
    primary_logits = torch.einsum(
        "bhajid,bhafkjcd->bhafkjic", query_h, primary_key
    ) * float(scale) - primary_bias.unsqueeze(-2)
    alternate_logits = torch.einsum(
        "bhajid,bhafkjcd->bhafkjic", query_h, alternate_key
    ) * float(scale) - alternate_bias.unsqueeze(-2)

    outputs = {}
    alternate_fractions = {}
    never = torch.zeros_like(primary_logits, dtype=torch.bool)
    outputs["fixed_wide"] = selected_value_output(
        primary_logits, primary_value, primary_value, never
    )
    r2_refine = None
    for refinements in (1, 2, 4):
        selected_cluster = gain.topk(refinements, dim=-1).indices
        refine = torch.zeros_like(gain, dtype=torch.bool)
        refine.scatter_(dim=-1, index=selected_cluster, value=True)
        refine &= active
        if refinements == 2:
            r2_refine = refine
        use_alternate = refine.unsqueeze(-2) & (
            alternate_logits > primary_logits
        )
        alternate_fractions[str(refinements)] = float(
            use_alternate.float().mean().item()
        )
        envelope_logits = torch.where(
            use_alternate, alternate_logits, primary_logits
        )
        outputs[f"seed_tangent_envelope_r{refinements}"] = (
            selected_value_output(
                envelope_logits,
                primary_value,
                alternate_value,
                use_alternate,
            )
        )
        if refinements == 2:
            outputs["seed_tangent_mass_only_r2"] = selected_value_output(
                envelope_logits, primary_value, primary_value, never
            )
            outputs["seed_tangent_value_only_r2"] = selected_value_output(
                primary_logits,
                primary_value,
                alternate_value,
                use_alternate,
            )

    for name, expansion_anchor in synthetic_centroid_anchors(seeds).items():
        centroid_key, centroid_value, centroid_bias = fixed_assignment_merge_kv(
            expansion_anchor, key, value, assignment, scale
        )
        centroid_key = rearrange(
            centroid_key, "b a f j k c h d -> b h a f k j c d"
        ).float()
        centroid_value = rearrange(
            centroid_value, "b a f j k c h d -> b h a f k j c d"
        ).float()
        centroid_logits = torch.einsum(
            "bhajid,bhafkjcd->bhafkjic", query_h, centroid_key
        ) * float(scale) - centroid_bias.float().unsqueeze(-2)
        use_centroid = r2_refine.unsqueeze(-2) & (
            centroid_logits > primary_logits
        )
        centroid_envelope = torch.where(
            use_centroid, centroid_logits, primary_logits
        )
        output_name = f"{name}_envelope_r2"
        outputs[output_name] = selected_value_output(
            centroid_envelope,
            primary_value,
            centroid_value,
            use_centroid,
        )
        alternate_fractions[name] = float(
            use_centroid.float().mean().item()
        )
    diagnostics = {
        "mean_selected_tangent_gap": float(
            gain[torch.isfinite(gain)].mean().item()
        ),
        "alternate_fraction_by_refinement_count": alternate_fractions,
    }
    return outputs, diagnostics


def install_probe(stats, per_call_stats, per_head_stats, diagnostics, args):
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
            seeds = fixed_seed_queries(query, frames, rows)
            candidates, call_diagnostics = multitangent_outputs(
                query, k_blocks, v_blocks, seeds, dim**-0.5
            )
            diagnostics[str(call)] = call_diagnostics
            for name, output in candidates.items():
                stats[name].add(output, reference)
                per_call_stats[call][name].add(output, reference)
                for position, head in enumerate(selected_heads):
                    per_head_stats[head][name].add(
                        output[..., position : position + 1, :],
                        reference[..., position : position + 1, :],
                    )
            state["evaluated"].append(
                {"call": call, "step": step, "layer": layer, "query_frame": query_frame}
            )

        if args.trajectory == "dense":
            trajectory_output = dense
        elif args.trajectory == "c4":
            q_blocks = _hard_c4_video_to_blocks(q, rows, width)
            k_blocks = _hard_c4_video_to_blocks(k, rows, width)
            v_blocks = _hard_c4_video_to_blocks(v, rows, width)
            c4_blocks = torch.empty_like(q_blocks)
            for head_start in range(heads):
                hard_c4_720p_attention_triton_blocks(
                    q_blocks,
                    k_blocks,
                    v_blocks,
                    dim**-0.5,
                    output=c4_blocks,
                    head_start=head_start,
                    head_count=1,
                )
            trajectory_output = _hard_c4_blocks_to_video(c4_blocks)
        else:
            raise ValueError(f"unknown trajectory {args.trajectory!r}")
        return self.o(trajectory_output.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def theoretical_flops(svg_density):
    rows, width, clusters = 45, 80, 4
    denominator = 2 * rows * width
    methods = {}
    for refinements in (1, 2, 4):
        ratio = (
            3 * clusters * width
            + 2 * refinements * width
            + 2 * rows * clusters
            + rows * refinements
        ) / denominator
        methods[f"seed_tangent_envelope_r{refinements}"] = {
            "ratio_vs_dense": ratio,
            "ratio_vs_svg_ear": ratio / svg_density,
            "limit_ratio_vs_dense": 0.70 * svg_density,
            "passes_limit": ratio <= 0.70 * svg_density,
        }
    synthetic_ratio = (
        3 * clusters * width
        + 3 * 2 * width
        + 2 * rows * clusters
        + rows * 2
        + clusters
    ) / denominator
    return {
        "methods": methods,
        "synthetic_centroid_r2": {
            "ratio_vs_dense": synthetic_ratio,
            "ratio_vs_svg_ear": synthetic_ratio / svg_density,
            "limit_ratio_vs_dense": 0.70 * svg_density,
            "passes_limit": synthetic_ratio <= 0.70 * svg_density,
        },
        "leading_cost": "3CW + 2RW + 2HC + HR",
        "note": (
            "Secondary anchors reuse existing C4 seed-key logits; scalar "
            "candidate selection is lower order in head dimension. Synthetic "
            "centroids add RW anchor-key products and at most C vector sums."
        ),
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
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument(
        "--record_calls",
        default="1,20,40,41,60,80,81,100,120,121,140,160",
    )
    parser.add_argument("--trajectory", choices=("dense", "c4"), default="dense")
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
    diagnostics = {}
    state = install_probe(stats, per_call_stats, per_head_stats, diagnostics, args)

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
        "trajectory": args.trajectory,
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
        "diagnostics": diagnostics,
        "flops": theoretical_flops(args.svg_ear_density),
        "derivation": {
            "cluster_tangent": "T_cs(q)=phi_c(r_s)+grad phi_c(r_s)^T(q-r_s)",
            "envelope": "max(T_cc(q), T_cs(q)) <= phi_c(q)",
            "anchor_score": "D_phi_c(r_s,r_c)=phi_c(r_s)-T_cc(r_s)",
            "partition": "fixed by the original C4 routing seeds",
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
