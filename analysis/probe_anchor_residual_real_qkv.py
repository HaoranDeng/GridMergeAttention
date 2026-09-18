"""Measure sparse residual-anchor correction on native StepDistill Wan QKV.

The denoising trajectory always uses dense attention.  On selected calls and
heads, this probe evaluates the production Hard-C4 kernel, samples its exact
dense-output residual on several time/row anchor grids, and interpolates that
residual back to every query.  This separates two questions before an
expensive video sweep:

* which anchor-grid aspect ratio best reconstructs the real residual field;
* whether a globally shrunken or amplified correction is preferable.
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
from attention.token_merging import (
    _hard_c4_attention,
    _hard_c4_video_to_blocks,
    _interpolate_anchor_grid_residual,
    _uniform_anchor_indices,
)
from generate import load_prompts


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
            "mse_vs_dense": self.sqerr / self.numel if self.numel else 0.0,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
        }


class ResidualFitStats:
    def __init__(self):
        self.calls = 0
        self.residual_sq = 0.0
        self.interpolation_sq = 0.0
        self.cross = 0.0
        self.numel = 0

    def add(self, residual, interpolation):
        residual_f = residual.float().reshape(-1)
        interpolation_f = interpolation.float().reshape(-1)
        self.calls += 1
        self.residual_sq += float(torch.dot(residual_f, residual_f))
        self.interpolation_sq += float(torch.dot(interpolation_f, interpolation_f))
        self.cross += float(torch.dot(residual_f, interpolation_f))
        self.numel += residual_f.numel()

    def summary(self):
        alpha = self.cross / max(self.interpolation_sq, 1e-20)
        corrected_sq = (
            self.residual_sq
            - 2.0 * alpha * self.cross
            + alpha * alpha * self.interpolation_sq
        )
        cosine = self.cross / math.sqrt(
            max(self.residual_sq * self.interpolation_sq, 1e-20)
        )
        return {
            "calls": self.calls,
            "numel": self.numel,
            "optimal_global_scale": alpha,
            "residual_interpolation_cosine": cosine,
            "oracle_scaled_mse_ratio_vs_hard_c4": corrected_sq
            / max(self.residual_sq, 1e-20),
            "unit_scale_mse_ratio_vs_hard_c4": (
                self.residual_sq - 2.0 * self.cross + self.interpolation_sq
            )
            / max(self.residual_sq, 1e-20),
        }


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").replace(":", " ").split()]


def parse_floats(value):
    return [
        float(item) for item in value.replace(",", " ").replace(":", " ").split()
    ]


def parse_layouts(value):
    layouts = []
    for item in value.replace(",", " ").split():
        time_count, row_count = item.lower().split("x", maxsplit=1)
        layouts.append((int(time_count), int(row_count)))
    return layouts


def install_probe(error_stats, fit_stats, args):
    import wan.modules.model as wan_model

    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    layouts = parse_layouts(args.layouts)
    scales = parse_floats(args.scales)
    state = {
        "calls_total": 0,
        "grid_size": None,
        "evaluated": [],
    }

    def dense_attention(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

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
        dense = dense_attention(q, k, v, self)

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
                first_frame_full_attention=False,
                full_attention_anchor_time_count=0,
                full_attention_anchor_row_count=0,
                full_attention_anchor_scale=1.0,
            )
            dense_blocks = _hard_c4_video_to_blocks(dense_selected, rows, width)
            hard_blocks = _hard_c4_video_to_blocks(hard, rows, width)
            residual = dense_blocks - hard_blocks
            error_stats["hard_c4"].add(hard_blocks, dense_blocks)

            call_result = {
                "call": call,
                "step": (call - 1) // 40,
                "layer": (call - 1) % 40,
                "layouts": {},
            }
            for time_count, row_count in layouts:
                if time_count > frames or row_count > rows:
                    raise ValueError(
                        f"layout {time_count}x{row_count} exceeds {frames}x{rows}"
                    )
                label = f"t{time_count}_h{row_count}"
                time_anchors = _uniform_anchor_indices(
                    frames, time_count, residual.device
                )
                row_anchors = _uniform_anchor_indices(rows, row_count, residual.device)
                anchor_residual = residual.index_select(1, time_anchors).index_select(
                    2, row_anchors
                )
                interpolation = _interpolate_anchor_grid_residual(
                    anchor_residual,
                    time_anchors,
                    row_anchors,
                    frames=frames,
                    rows=rows,
                )
                fit_stats[label].add(residual, interpolation)
                residual_f = residual.float().reshape(-1)
                interpolation_f = interpolation.float().reshape(-1)
                per_call_scale = float(
                    torch.dot(residual_f, interpolation_f)
                    / torch.dot(interpolation_f, interpolation_f).clamp_min(1e-20)
                )
                call_result["layouts"][label] = {
                    "anchors": time_count * row_count,
                    "anchor_query_fraction": time_count
                    * row_count
                    / (frames * rows),
                    "per_call_optimal_scale": per_call_scale,
                }
                for scale in scales:
                    scale_label = str(scale).replace("-", "m").replace(".", "p")
                    name = f"{label}_scale_{scale_label}"
                    corrected = hard_blocks + scale * interpolation
                    error_stats[name].add(corrected, dense_blocks)
                del anchor_residual, interpolation

            state["evaluated"].append(call_result)
            del q_selected, k_selected, v_selected, dense_selected
            del hard, dense_blocks, hard_blocks, residual

        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe residual-anchor layouts on native StepDistill Wan QKV."
    )
    parser.add_argument("--config", default="configs/dense_wan_14b_720p_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--heads", default="0,20")
    parser.add_argument(
        "--record_calls",
        default="1,14,27,40,41,54,67,80,81,94,107,120,121,134,147,160",
    )
    parser.add_argument(
        "--layouts",
        default="3x6,5x9,6x12,4x18,8x9,9x8,12x6,18x4",
    )
    parser.add_argument("--scales", default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    error_stats = defaultdict(ErrorStats)
    fit_stats = defaultdict(ResidualFitStats)
    state = install_probe(error_stats, fit_stats, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: stats.summary() for name, stats in sorted(error_stats.items())}
    hard_mse = errors["hard_c4"]["mse_vs_dense"]
    for result in errors.values():
        result["mse_ratio_vs_hard_c4"] = result["mse_vs_dense"] / hard_mse
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
        "wall_seconds": time.perf_counter() - start,
        "errors_vs_dense": errors,
        "residual_fits": {
            name: stats.summary() for name, stats in sorted(fit_stats.items())
        },
        "per_call": state["evaluated"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
