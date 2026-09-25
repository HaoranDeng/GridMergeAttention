"""Probe anisotropic 3-D residual anchors on native Wan2.1-14B QKV.

The dense denoising trajectory is preserved.  On selected calls and heads,
the probe measures the Hard-C4 residual field, estimates its axis-wise
curvature, and compares equal-budget tensor-product anchor lattices.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.compare_hard_c4_backends_dense import load_pipeline, run_denoise_only, sync
from analysis.probe_anchor_residual_real_qkv import (
    ErrorStats,
    ResidualFitStats,
    parse_floats,
    parse_ints,
)
from attention.token_merging import (
    _hard_c4_attention,
    _hard_c4_video_to_blocks,
    _linear_interpolation_map,
    _uniform_anchor_indices,
)
from generate import load_prompts


AXES = {"time": 1, "row": 2, "width": 3}


class CurvatureStats:
    def __init__(self):
        self.calls = 0
        self.sq = 0.0
        self.abs_max = 0.0
        self.numel = 0

    def add(self, residual, axis):
        length = residual.size(axis)
        if length < 3:
            raise ValueError("second differences require at least three samples")
        center = residual.narrow(axis, 1, length - 2).float()
        left = residual.narrow(axis, 0, length - 2).float()
        right = residual.narrow(axis, 2, length - 2).float()
        second = right - 2.0 * center + left
        self.calls += 1
        self.sq += float(second.square().sum())
        self.abs_max = max(self.abs_max, float(second.abs().max()))
        self.numel += second.numel()

    def summary(self, length):
        rms = math.sqrt(self.sq / max(self.numel, 1))
        return {
            "calls": self.calls,
            "numel": self.numel,
            "second_difference_rms": rms,
            "second_difference_abs_max": self.abs_max,
            "rms_allocation_coefficient": (length - 1) ** 2 * rms,
            "max_allocation_coefficient": (length - 1) ** 2 * self.abs_max,
        }


def parse_layouts(value):
    layouts = []
    for item in value.replace(",", " ").split():
        counts = tuple(int(part) for part in item.lower().split("x"))
        if len(counts) != 3:
            raise ValueError(f"expected TxHxW layout, got {item!r}")
        layouts.append(counts)
    return layouts


def _interpolate_axis(values, anchors, length, axis):
    left, right, alpha = _linear_interpolation_map(length, anchors)
    value_left = values.index_select(axis, left)
    value_right = values.index_select(axis, right)
    shape = [1] * values.ndim
    shape[axis] = length
    alpha = alpha.to(values.dtype).view(shape)
    return torch.lerp(value_left, value_right, alpha)


def interpolate_anchor_residual_3d(
    anchor_residual,
    time_anchors,
    row_anchors,
    width_anchors,
    *,
    frames,
    rows,
    width,
):
    """Trilinearly reconstruct [B,T,H,W,heads,dim] anchor residuals."""

    residual = _interpolate_axis(anchor_residual, width_anchors, width, axis=3)
    residual = _interpolate_axis(residual, row_anchors, rows, axis=2)
    return _interpolate_axis(residual, time_anchors, frames, axis=1)


def continuous_axis_allocation(coefficients, budget):
    product = math.prod(coefficients.values())
    denominator = product ** (1.0 / 6.0)
    scale = budget ** (1.0 / 3.0)
    return {
        axis: scale * math.sqrt(value) / denominator
        for axis, value in coefficients.items()
    }


def install_probe(error_stats, fit_stats, curvature_stats, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    layouts = parse_layouts(args.layouts)
    scales = parse_floats(args.scales)
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
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        dense = wan_model.flash_attention(
            q=q, k=k, v=v, window_size=self.window_size
        )

        state["calls_total"] += 1
        call = state["calls_total"]
        frames, rows, width = [int(item) for item in grid_size]
        state["grid_size"] = [frames, rows, width]
        if call in record_calls:
            for head in selected_heads:
                if not 0 <= head < heads:
                    raise ValueError(f"head index {head} is outside [0, {heads})")
            head_index = torch.tensor(selected_heads, device=q.device)
            q_selected = q.index_select(2, head_index).contiguous()
            k_selected = k.index_select(2, head_index).contiguous()
            v_selected = v.index_select(2, head_index).contiguous()
            dense_selected = dense.index_select(2, head_index).contiguous()
            hard = _hard_c4_attention(
                q_selected,
                k_selected,
                v_selected,
                h=rows,
                w=width,
                h_reduce=1,
                w_reduce=1,
                sm_scale=dim**-0.5,
                query_outer_chunk=frames,
                head_chunk=len(selected_heads),
                hard_route_backend="hard_c4",
                exact_intra_frame_attention=False,
                exact_neighbor_window_columns=0,
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
                seed_mode="local_with_ith",
            )
            dense_blocks = _hard_c4_video_to_blocks(dense_selected, rows, width)
            hard_blocks = _hard_c4_video_to_blocks(hard, rows, width)
            residual = dense_blocks - hard_blocks
            error_stats["hard_c4"].add(hard_blocks, dense_blocks)
            for axis_name, axis in AXES.items():
                curvature_stats[axis_name].add(residual, axis)

            call_result = {
                "call": call,
                "step": (call - 1) // 40,
                "layer": (call - 1) % 40,
                "layouts": {},
            }
            for time_count, row_count, width_count in layouts:
                counts = (time_count, row_count, width_count)
                limits = (frames, rows, width)
                if any(count > limit for count, limit in zip(counts, limits)):
                    raise ValueError(f"layout {counts} exceeds grid {limits}")
                label = f"t{time_count}_h{row_count}_w{width_count}"
                time_anchors = _uniform_anchor_indices(
                    frames, time_count, residual.device
                )
                row_anchors = _uniform_anchor_indices(rows, row_count, residual.device)
                width_anchors = _uniform_anchor_indices(
                    width, width_count, residual.device
                )
                anchor_residual = (
                    residual.index_select(1, time_anchors)
                    .index_select(2, row_anchors)
                    .index_select(3, width_anchors)
                )
                interpolation = interpolate_anchor_residual_3d(
                    anchor_residual,
                    time_anchors,
                    row_anchors,
                    width_anchors,
                    frames=frames,
                    rows=rows,
                    width=width,
                )
                fit_stats[label].add(residual, interpolation)
                call_result["layouts"][label] = {
                    "anchor_queries": math.prod(counts),
                    "anchor_query_fraction": math.prod(counts)
                    / (frames * rows * width),
                }
                for scale in scales:
                    scale_label = str(scale).replace("-", "m").replace(".", "p")
                    corrected = hard_blocks + scale * interpolation
                    error_stats[f"{label}_scale_{scale_label}"].add(
                        corrected, dense_blocks
                    )
                del anchor_residual, interpolation

            state["evaluated"].append(call_result)
            del q_selected, k_selected, v_selected, dense_selected
            del hard, dense_blocks, hard_blocks, residual

        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe anisotropic residual-anchor lattices on Wan QKV."
    )
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument(
        "--record_calls",
        default="1,20,40,41,60,80,81,100,120,121,140,160",
    )
    parser.add_argument(
        "--layouts",
        default=(
            "5x9x80,7x13x40,8x15x30,9x16x25,10x18x20,"
            "7x15x57,8x15x50,10x15x40,10x20x30,12x20x25"
        ),
    )
    parser.add_argument("--scales", default="0.75,1.0")
    parser.add_argument("--svg_ear_density", type=float, default=0.37906453310046345)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    error_stats = defaultdict(ErrorStats)
    fit_stats = defaultdict(ResidualFitStats)
    curvature_stats = defaultdict(CurvatureStats)
    state = install_probe(error_stats, fit_stats, curvature_stats, args)

    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: stats.summary() for name, stats in sorted(error_stats.items())}
    hard_mse = errors["hard_c4"]["mse_vs_dense"]
    for result in errors.values():
        result["mse_ratio_vs_hard_c4"] = result["mse_vs_dense"] / hard_mse
    frames, rows, width = state["grid_size"]
    lengths = {"time": frames, "row": rows, "width": width}
    curvature = {
        axis: curvature_stats[axis].summary(lengths[axis]) for axis in AXES
    }
    coefficients = {
        axis: summary["rms_allocation_coefficient"]
        for axis, summary in curvature.items()
    }
    budgets = sorted({math.prod(layout) for layout in parse_layouts(args.layouts)})
    base_c4_ratio = 3 * 4 / (2 * rows) + 4 / width
    total_queries = frames * rows * width
    layout_flops = {}
    for layout in parse_layouts(args.layouts):
        label = f"t{layout[0]}_h{layout[1]}_w{layout[2]}"
        density = base_c4_ratio + math.prod(layout) / total_queries
        layout_flops[label] = {
            "anchor_queries": math.prod(layout),
            "total_ratio_vs_dense": density,
            "ratio_vs_svg_ear": density / args.svg_ear_density,
            "passes_70_percent_svg_ear_limit": (
                density <= 0.7 * args.svg_ear_density
            ),
        }

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
        "layouts": [list(layout) for layout in parse_layouts(args.layouts)],
        "scales": parse_floats(args.scales),
        "wall_seconds": time.perf_counter() - started,
        "axis_curvature": curvature,
        "continuous_axis_allocations": {
            str(budget): continuous_axis_allocation(coefficients, budget)
            for budget in budgets
        },
        "errors_vs_dense": errors,
        "residual_fits": {
            name: stats.summary() for name, stats in sorted(fit_stats.items())
        },
        "layout_flops": layout_flops,
        "per_call": state["evaluated"],
        "invariants": {
            "dense_trajectory_preserved": True,
            "all_layers_and_denoising_steps_remain_dense_for_probe": True,
            "candidate_inference_schedule": "all-layer Hard-C4 plus sparse exact-query residual anchors",
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
