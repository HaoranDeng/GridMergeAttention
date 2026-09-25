"""Test a bi-axial GMA hypothesis on native Wan Q/K/V.

The two orientations independently approximate the dense attention numerator
and partition function.  Their principled fusion therefore happens before
normalization:

    O_hat = (N_hat_hw + N_hat_wh) / (Z_hat_hw + Z_hat_wh).

This probe keeps the model trajectory dense and evaluates only selected
attention calls, query frames, and heads.  It also reports plain output
averaging and an oracle per-query orientation as diagnostics.
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

from analysis.compare_hard_c4_backends_dense import (
    load_pipeline,
    run_denoise_only,
    sync,
)
from analysis.explore_axis_block_layouts import (
    gather_seed_queries,
    local_seed_indices,
    query_blocks_to_video,
    to_key_blocks,
    to_query_blocks,
)
from analysis.explore_rowcol_block_layouts import merge_kv
from attention.hard_route_pytorch import (
    gather_seed_queries as gather_hard_seed_queries,
    wide_local_c4_seed_rows,
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
            "mse_vs_dense": self.sqerr / self.numel,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
        }


class ComplementarityStats:
    def __init__(self):
        self.queries = 0
        self.row_better = 0
        self.sum_row_error = 0.0
        self.sum_col_error = 0.0
        self.sum_row_error_sq = 0.0
        self.sum_col_error_sq = 0.0
        self.sum_cross_error = 0.0
        self.sum_row_gate = 0.0

    def add(self, row_output, col_output, reference, row_gate):
        row_error = (row_output.float() - reference.float()).square().mean(-1)
        col_error = (col_output.float() - reference.float()).square().mean(-1)
        row_error = row_error.reshape(-1).double()
        col_error = col_error.reshape(-1).double()
        gate = row_gate.reshape(-1).double()
        self.queries += row_error.numel()
        self.row_better += int((row_error <= col_error).sum())
        self.sum_row_error += float(row_error.sum())
        self.sum_col_error += float(col_error.sum())
        self.sum_row_error_sq += float(row_error.square().sum())
        self.sum_col_error_sq += float(col_error.square().sum())
        self.sum_cross_error += float((row_error * col_error).sum())
        self.sum_row_gate += float(gate.sum())

    def summary(self):
        count = max(self.queries, 1)
        mean_row = self.sum_row_error / count
        mean_col = self.sum_col_error / count
        covariance = self.sum_cross_error / count - mean_row * mean_col
        row_variance = self.sum_row_error_sq / count - mean_row**2
        col_variance = self.sum_col_error_sq / count - mean_col**2
        denominator = math.sqrt(max(row_variance * col_variance, 0.0))
        correlation = covariance / denominator if denominator > 0 else float("nan")
        return {
            "queries": self.queries,
            "row_better_fraction": self.row_better / count,
            "row_col_query_error_correlation": correlation,
            "mean_moment_gate_weight_on_row": self.sum_row_gate / count,
        }


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").split()]


def compressed_attention_moments(query, merged_key, merged_value, routing_bias, scale):
    """Return normalized output and log partition for one GMA orientation."""

    q = rearrange(query, "b a i j h d -> (b h a j) i d").float()
    k = rearrange(merged_key, "b a f j k c h d -> (b h a j) (f k c) d").float()
    v = rearrange(merged_value, "b a f j k c h d -> (b h a j) (f k c) d").float()
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)").float()
    logits = torch.einsum("bid,bmd->bim", q, k) * float(scale)
    logits = logits - bias.unsqueeze(1)
    log_partition = torch.logsumexp(logits, dim=-1)
    output = torch.einsum("bim,bmd->bid", torch.softmax(logits, dim=-1), v)
    output = rearrange(
        output,
        "(b h a j) i d -> b a i j h d",
        b=query.size(0),
        h=query.size(4),
        a=query.size(1),
        j=query.size(3),
    ).to(query.dtype)
    log_partition = rearrange(
        log_partition,
        "(b h a j) i -> b a i j h",
        b=query.size(0),
        h=query.size(4),
        a=query.size(1),
        j=query.size(3),
    )
    return output, log_partition


def axis_moments_for_frame(
    q,
    k,
    v,
    grid_size,
    query_frame,
    scale,
    *,
    q_axis,
    k_axis,
    clusters,
):
    q_blocks = to_query_blocks(q, grid_size, q_axis)
    q_blocks = q_blocks[:, query_frame : query_frame + 1].contiguous()
    k_blocks = to_key_blocks(k, grid_size, k_axis)
    v_blocks = to_key_blocks(v, grid_size, k_axis)
    seeds = local_seed_indices(q_axis, k_axis, grid_size, clusters, q.device)
    seed_queries = gather_seed_queries(q_blocks, seeds)
    merged_key, merged_value, routing_bias = merge_kv(
        seed_queries, k_blocks, v_blocks, scale
    )
    output, log_partition = compressed_attention_moments(
        q_blocks, merged_key, merged_value, routing_bias, scale
    )
    _, height, width = [int(value) for value in grid_size]
    output = query_blocks_to_video(output, (1, height, width), q_axis)
    log_partition = query_blocks_to_video(
        log_partition.unsqueeze(-1), (1, height, width), q_axis
    ).squeeze(-1)
    return output, log_partition


def production_row_c4_moments_for_frame(q, k, v, grid_size, query_frame, scale):
    """Match the maintained 14B 720p wide-seed C4 reference exactly."""

    frames, rows, width = [int(value) for value in grid_size]
    q_blocks = to_query_blocks(q, grid_size, "h")
    q_blocks = q_blocks[:, query_frame : query_frame + 1].contiguous()
    k_blocks = to_key_blocks(k, grid_size, "w")
    v_blocks = to_key_blocks(v, grid_size, "w")
    seed_rows = wide_local_c4_seed_rows(
        torch.arange(1, device=q.device),
        torch.arange(frames, device=q.device),
        rows,
        width,
        rows,
    )
    seed_queries = gather_hard_seed_queries(q_blocks, seed_rows)
    merged_key, merged_value, routing_bias = merge_kv(
        seed_queries, k_blocks, v_blocks, scale
    )
    output, log_partition = compressed_attention_moments(
        q_blocks, merged_key, merged_value, routing_bias, scale
    )
    output = query_blocks_to_video(output, (1, rows, width), "h")
    log_partition = query_blocks_to_video(
        log_partition.unsqueeze(-1), (1, rows, width), "h"
    ).squeeze(-1)
    return output, log_partition


def fuse_moments(row_output, row_log_z, col_output, col_log_z):
    log_z = torch.stack((row_log_z, col_log_z), dim=0)
    gate = torch.softmax(log_z, dim=0)
    output = gate[0].unsqueeze(-1) * row_output.float()
    output = output + gate[1].unsqueeze(-1) * col_output.float()
    return output.to(row_output.dtype), gate[0]


def oracle_orientation(row_output, col_output, reference):
    row_error = (row_output.float() - reference.float()).square().mean(-1)
    col_error = (col_output.float() - reference.float()).square().mean(-1)
    return torch.where(
        (row_error <= col_error).unsqueeze(-1), row_output, col_output
    )


def allocation_flops(rows, width, row_clusters, column_clusters):
    dense = 2.0 * rows * width
    row = (
        3.0 * row_clusters * width + 2.0 * rows * row_clusters
    ) / dense
    col = (
        3.0 * column_clusters * rows + 2.0 * width * column_clusters
    ) / dense
    return row, col, row + col


def theoretical_flops(rows, width, clusters, svg_ear_density):
    row, col, biaxial = allocation_flops(
        rows, width, clusters, clusters
    )
    anchor_row, anchor_col, anchor_biaxial = allocation_flops(
        rows, width, 4, 2
    )
    return {
        "row_c3_ratio_vs_dense": row,
        "column_c3_ratio_vs_dense": col,
        "biaxial_c3_ratio_vs_dense": biaxial,
        "svg_ear_realized_density": svg_ear_density,
        "biaxial_ratio_vs_svg_ear": biaxial / svg_ear_density,
        "passes_70pct_svg_ear_limit": biaxial <= 0.70 * svg_ear_density,
        "anchor_c4_c2": {
            "row_ratio_vs_dense": anchor_row,
            "column_ratio_vs_dense": anchor_col,
            "biaxial_ratio_vs_dense": anchor_biaxial,
            "biaxial_ratio_vs_svg_ear": anchor_biaxial / svg_ear_density,
            "passes_70pct_svg_ear_limit": (
                anchor_biaxial <= 0.70 * svg_ear_density
            ),
        },
    }


def install_probe(stats, per_head_stats, complementarity, args):
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
            q_selected = q.index_select(2, head_index).contiguous()
            k_selected = k.index_select(2, head_index).contiguous()
            v_selected = v.index_select(2, head_index).contiguous()
            step = (call - 1) // 40
            layer = (call - 1) % 40
            query_frame = (step * 7 + layer * 3) % frames
            frame_start = query_frame * rows * width
            frame_end = frame_start + rows * width
            reference = dense[:, frame_start:frame_end].index_select(
                2, head_index
            ).contiguous()
            scale = dim**-0.5

            row_c4, row_c4_log_z = production_row_c4_moments_for_frame(
                q_selected,
                k_selected,
                v_selected,
                grid_size,
                query_frame,
                scale,
            )
            row_c3, row_log_z = axis_moments_for_frame(
                q_selected,
                k_selected,
                v_selected,
                grid_size,
                query_frame,
                scale,
                q_axis="h",
                k_axis="w",
                clusters=args.clusters,
            )
            col_c3, col_log_z = axis_moments_for_frame(
                q_selected,
                k_selected,
                v_selected,
                grid_size,
                query_frame,
                scale,
                q_axis="w",
                k_axis="h",
                clusters=args.clusters,
            )
            col_c2, col_c2_log_z = axis_moments_for_frame(
                q_selected,
                k_selected,
                v_selected,
                grid_size,
                query_frame,
                scale,
                q_axis="w",
                k_axis="h",
                clusters=2,
            )
            moment, row_gate = fuse_moments(
                row_c3, row_log_z, col_c3, col_log_z
            )
            anchor_moment, anchor_row_gate = fuse_moments(
                row_c4, row_c4_log_z, col_c2, col_c2_log_z
            )
            output_mean = 0.5 * (row_c3.float() + col_c3.float())
            output_mean = output_mean.to(row_c3.dtype)
            anchor_output_mean = 0.5 * (row_c4.float() + col_c2.float())
            anchor_output_mean = anchor_output_mean.to(row_c4.dtype)
            outputs = {
                "row_c4": row_c4,
                "row_c3": row_c3,
                "column_c3": col_c3,
                "biaxial_c3_moment": moment,
                "biaxial_c3_output_mean": output_mean,
                "biaxial_c3_oracle_axis": oracle_orientation(
                    row_c3, col_c3, reference
                ),
                "biaxial_row_c4_column_c2_moment": anchor_moment,
                "biaxial_row_c4_column_c2_output_mean": anchor_output_mean,
                "biaxial_row_c4_column_c2_oracle_axis": oracle_orientation(
                    row_c4, col_c2, reference
                ),
            }
            for name, output in outputs.items():
                stats[name].add(output, reference)
                for position, head in enumerate(selected_heads):
                    per_head_stats[head][name].add(
                        output[..., position : position + 1, :],
                        reference[..., position : position + 1, :],
                    )
            complementarity["symmetric_c3_c3"].add(
                row_c3, col_c3, reference, row_gate
            )
            complementarity["anchor_c4_c2"].add(
                row_c4, col_c2, reference, anchor_row_gate
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
    return state


def add_ratios(summaries, baseline_name):
    baseline = summaries[baseline_name]["mse_vs_dense"]
    for value in summaries.values():
        value[f"mse_ratio_vs_{baseline_name}"] = value["mse_vs_dense"] / baseline


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
        default="1,14,27,40,41,54,67,80,81,94,107,120",
    )
    parser.add_argument("--clusters", type=int, default=3)
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
    per_head_stats = defaultdict(lambda: defaultdict(ErrorStats))
    complementarity = defaultdict(ComplementarityStats)
    state = install_probe(stats, per_head_stats, complementarity, args)
    sync()
    started = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    errors = {name: value.summary() for name, value in sorted(stats.items())}
    add_ratios(errors, "row_c4")
    per_head_errors = {}
    for head, values in sorted(per_head_stats.items()):
        summaries = {
            name: value.summary() for name, value in sorted(values.items())
        }
        add_ratios(summaries, "row_c4")
        per_head_errors[str(head)] = summaries

    rows, width = state["grid_size"][1:]
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
        "per_head_errors_vs_dense": per_head_errors,
        "complementarity": {
            name: value.summary()
            for name, value in sorted(complementarity.items())
        },
        "flops": theoretical_flops(
            rows, width, args.clusters, args.svg_ear_density
        ),
        "derivation": {
            "dense_moments": "O=N/Z, N=sum_j exp(s_ij) v_j, Z=sum_j exp(s_ij)",
            "biaxial_fusion": "O_hat=(N_hat_hw+N_hat_wh)/(Z_hat_hw+Z_hat_wh)",
            "equivalent_gate": "softmax(log Z_hat_hw, log Z_hat_wh)",
            "trajectory": "dense",
            "intended_schedule": "all 40 layers at every denoise step",
        },
        "evaluated": state["evaluated"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
