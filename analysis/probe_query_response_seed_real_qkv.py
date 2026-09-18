"""Evaluate low-cost response-coverage seeds on native Wan StepDistill QKV.

The model always advances on the dense trajectory.  For declared attention
calls, the probe replaces the four fixed C4 query anchors only inside the
measurement branch: a small, uniformly sampled key sketch selects four query
rows whose response profiles span the local candidate set.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_query_centroid_gma_real_qkv import ErrorStats, parse_ints
from analysis.sweep_wan14b_720p_hard_c4_algorithm import VARIANTS, hard_c4_sample
from attention.token_merging import _hard_c4_video_to_blocks
from generate import load_prompts


DEFAULT_VARIANTS = (
    "baseline",
    "qblock_response15_s4_span_prob",
    "qblock_response15_s8_span_prob",
    "qblock_response15_s8_span_prob_anchor_left",
    "qblock_response21_s8_span_prob",
    "qblock_response15_s8_farthest_prob",
)


def _register_cheap_calibrations():
    """Declare scalar-only calibrations around the regular 15x8 selector."""

    base = dict(VARIANTS["qblock_response15_s8_span_prob_anchor_left"])
    for value in (0.9, 0.975, 1.0, 1.075, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_qresp_bias_{label}"] = {**base, "bias_scale": value}
    for value in (0.25, 0.5, 0.75):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_qresp_entropy_{label}"] = {
            **base,
            "entropy_bias_mix": value,
        }
    for value in (0.15, 0.35, 0.5):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_qresp_peak_{label}"] = {
            **base,
            "peak_weight_mix": value,
        }
    for value in (0.1, 0.25, 0.5, 0.75, 1.0):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_qresp_anchor_{label}"] = {
            **base,
            "adaptive_anchor_blend": value,
            "adaptive_anchor_mode": "logit",
            "adaptive_anchor_mass": "raw",
        }
    for value in (0.75, 1.25, 1.5):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_qresp_merge_temp_{label}"] = {
            **base,
            "merge_temperature": value,
        }
    VARIANTS["paper_qresp_combo"] = {
        **base,
        "bias_scale": 1.075,
        "entropy_bias_mix": 0.5,
        "peak_weight_mix": 0.35,
        "adaptive_anchor_blend": 0.25,
        "adaptive_anchor_mode": "logit",
        "adaptive_anchor_mass": "raw",
    }
    VARIANTS["paper_qresp_combo_no_peak"] = {
        **base,
        "bias_scale": 1.075,
        "entropy_bias_mix": 0.5,
        "adaptive_anchor_blend": 0.25,
        "adaptive_anchor_mode": "logit",
        "adaptive_anchor_mass": "raw",
    }
    VARIANTS["paper_qresp_overlap"] = {**base, "overlap_clusters": True}
    temporal_base = {**base, "block_proxy_temporal_mean": True}
    for value in (1.075, 1.1, 1.2):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_qresp_tmean_bias_{label}"] = {
            **temporal_base,
            "bias_scale": value,
        }
    VARIANTS["paper_qresp_tmean_combo"] = {
        **temporal_base,
        "bias_scale": 1.075,
        "adaptive_anchor_blend": 0.25,
        "adaptive_anchor_mode": "logit",
        "adaptive_anchor_mass": "raw",
    }


def _register_fixed_c4_calibrations():
    """Declare all-layer corrections around production's fixed wide C4 seeds."""

    base = dict(VARIANTS["seed_wide"])
    for value in (1.025, 1.05, 1.075, 1.1, 1.2):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_c4_wide_bias_{label}"] = {**base, "bias_scale": value}
    for value in (0.1, 0.25, 0.5):
        label = str(value).replace(".", "p")
        VARIANTS[f"paper_c4_wide_anchor_{label}"] = {
            **base,
            "adaptive_anchor_blend": value,
            "adaptive_anchor_mode": "logit",
            "adaptive_anchor_mass": "raw",
        }
    for bias_scale in (1.05, 1.075, 1.1):
        bias_label = str(bias_scale).replace(".", "p")
        for anchor_blend in (0.1, 0.25, 0.5):
            anchor_label = str(anchor_blend).replace(".", "p")
            VARIANTS[f"paper_c4_wide_combo_b{bias_label}_a{anchor_label}"] = {
                **base,
                "bias_scale": bias_scale,
                "adaptive_anchor_blend": anchor_blend,
                "adaptive_anchor_mode": "logit",
                "adaptive_anchor_mass": "raw",
            }


_register_cheap_calibrations()
_register_fixed_c4_calibrations()


def _measure_cuda_ms(function):
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = function()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def _summarize(stats):
    result = {name: value.summary() for name, value in sorted(stats.items())}
    baseline = result["baseline"]["mse_vs_dense"]
    for value in result.values():
        value["mse_ratio_vs_c4"] = value["mse_vs_dense"] / baseline
    return result


def _flops(variants, svg_density):
    rows, width, clusters, head_dim = 45, 80, 4, 128
    dense = 2 * rows * width
    c4 = 3 * clusters * width + 2 * rows * clusters
    result = {}
    for name in variants:
        config = VARIANTS[name]
        candidates = int(config.get("seed_candidates", 0))
        samples = int(config.get("block_proxy_samples", 0))
        proxy_qk = candidates * samples
        if config.get("block_proxy_temporal_mean", False):
            proxy_qk /= 21
        # Pivoting works on short response vectors, so retain it explicitly
        # rather than hiding it in the matrix-only estimate.
        pivot_scalar = 0
        if config.get("seed") == "q_block_response":
            selected_after_anchor = clusters - (
                1 + len(config.get("seed_anchor_offsets", ()))
                if config.get("force_center", True)
                else len(config.get("seed_anchor_offsets", ()))
            )
            selected_after_anchor = max(selected_after_anchor, 0)
            pivot_scalar = 3 * candidates * samples * selected_after_anchor
        # The adaptive CxC anchor correction is scalar work over the W keys;
        # convert its approximate logsumexp/blend cost to head-dimension units.
        anchor_scalar = 0
        if "adaptive_anchor_blend" in config:
            anchor_scalar = 3 * clusters * clusters * width
        units = c4 + proxy_qk + (pivot_scalar + anchor_scalar) / head_dim
        result[name] = {
            "dense_units_per_head_dim": dense,
            "c4_units_per_head_dim": c4,
            "proxy_qk_units_per_head_dim": proxy_qk,
            "pivot_scalar_flops_as_head_dim_units": pivot_scalar / head_dim,
            "anchor_scalar_flops_as_head_dim_units": anchor_scalar / head_dim,
            "total_units_per_head_dim": units,
            "ratio_vs_dense": units / dense,
            "ratio_vs_svg_ear": (units / dense) / svg_density,
            "passes_70pct_svg_limit": units / dense <= 0.70 * svg_density,
        }
    return result


def install_probe(state, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    head_index = None
    record_calls = set(parse_ints(args.record_calls))

    def probed_self_attention(self, x, grid_size, rope_cache):
        nonlocal head_index
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

        state["call"] += 1
        call = state["call"]
        if call in record_calls:
            frames, rows, width = [int(item) for item in grid_size]
            if head_index is None:
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

            for name in state["variants"]:
                output, elapsed_ms = _measure_cuda_ms(
                    lambda name=name: hard_c4_sample(
                        query,
                        k_blocks,
                        v_blocks,
                        dim**-0.5,
                        VARIANTS[name],
                    )
                )
                approximation, cluster_counts = output
                state["stats"][name].add(approximation, reference)
                state["per_call_stats"][call][name].add(approximation, reference)
                state["timings_ms"][name].append(elapsed_ms)
                state["cluster_count_cv"][name].append(
                    float(
                        cluster_counts.float().std(correction=0)
                        / cluster_counts.float().mean().clamp_min(1e-12)
                    )
                )
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--heads", default="10,30")
    parser.add_argument(
        "--record_calls",
        default="1,20,40,41,60,80,81,100,120,121,140,160",
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    variants = args.variants.replace(",", " ").split()
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if "baseline" not in variants:
        raise ValueError("The baseline C4 variant is required")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + 1)
    state = {
        "call": 0,
        "variants": variants,
        "stats": defaultdict(ErrorStats),
        "per_call_stats": defaultdict(lambda: defaultdict(ErrorStats)),
        "timings_ms": defaultdict(list),
        "cluster_count_cv": defaultdict(list),
        "evaluated": [],
    }
    install_probe(state, args)

    sync()
    started = time.perf_counter()
    run_denoise_only(
        pipeline, config, prompts[args.prompt_index], args.seed, dtype
    )
    sync()

    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "heads": parse_ints(args.heads),
        "record_calls": parse_ints(args.record_calls),
        "wall_seconds": time.perf_counter() - started,
        "errors_vs_dense": _summarize(state["stats"]),
        "per_call_errors_vs_dense": {
            str(call): _summarize(values)
            for call, values in sorted(state["per_call_stats"].items())
        },
        "sampled_head_frame_timing_ms": {
            name: {
                "calls": len(values),
                "mean": sum(values) / len(values),
                "values": values,
            }
            for name, values in sorted(state["timings_ms"].items())
        },
        "cluster_count_cv": {
            name: sum(values) / len(values)
            for name, values in sorted(state["cluster_count_cv"].items())
        },
        "flops": _flops(variants, args.svg_ear_density),
        "variant_configs": {name: VARIANTS[name] for name in variants},
        "evaluated": state["evaluated"],
        "trajectory": "dense; candidates are measured off-path on native StepDistill QKV",
        "schedule": "pure all-layer candidate, not a Dense/C4 layer hybrid",
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
