"""Compare hard and fractional C4 routing on native StepDistill Wan QKV.

The model always continues with dense attention. Approximate outputs are only
measured against the same call-local dense output, so every variant observes
identical RoPE-transformed Q/K/V tensors.
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
    blocks_to_video,
    load_pipeline,
    run_denoise_only,
    sync,
    video_to_blocks,
)
from attention.hard_route_pytorch import biased_sdpa, gather_seed_queries
from generate import load_prompts


FLOAT_NEG_INF = torch.finfo(torch.float32).min
SEED_COUNT = 4


class ErrorStats:
    def __init__(self):
        self.calls = 0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0
        self.max_abs = 0.0

    def add(self, output, reference):
        output_f = output.float().reshape(-1)
        reference_f = reference.float().reshape(-1)
        difference = output_f - reference_f
        self.calls += 1
        self.sqerr += float(torch.dot(difference, difference).item())
        self.ref_sq += float(torch.dot(reference_f, reference_f).item())
        self.numel += difference.numel()
        self.max_abs = max(self.max_abs, float(difference.abs().max().item()))

    def summary(self):
        return {
            "calls": self.calls,
            "numel": self.numel,
            "mse_vs_dense": self.sqerr / self.numel if self.numel else 0.0,
            "rel_l2_vs_dense": math.sqrt(self.sqerr / max(self.ref_sq, 1e-20)),
            "max_abs_vs_dense": self.max_abs,
        }


def parse_ints(value):
    return [int(item) for item in value.replace(",", " ").replace(":", " ").split()]


def parse_floats(value):
    return [float(item) for item in value.replace(",", " ").replace(":", " ").split()]


def wide_c4_seed_rows(query_outer_count, key_outer_count, key_rows, query_columns, query_rows, device):
    """Match the current 720p Triton C4 seed rows exactly."""
    key_row = torch.arange(key_rows, device=device).view(1, 1, key_rows, 1, 1)
    seed_ids = torch.arange(SEED_COUNT, device=device).view(1, 1, 1, 1, SEED_COUNT)
    seed_start = (key_row - 1).clamp(0, query_rows - SEED_COUNT)
    contiguous = seed_start + seed_ids
    wide_offsets = torch.tensor([-2, 0, 1, 3], device=device).view(1, 1, 1, 1, SEED_COUNT)
    wide = key_row + wide_offsets
    use_wide = (key_row >= 2) & (key_row <= query_rows - 4)
    rows = torch.where(use_wide, wide, contiguous).long()
    return rows.expand(
        query_outer_count,
        key_outer_count,
        key_rows,
        query_columns,
        SEED_COUNT,
    )


def hard_representatives(seed_queries, key, value, scale):
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    logits = logits * float(scale)
    assignment_score = torch.log_softmax(logits, dim=-1)
    assignment = torch.zeros_like(logits, dtype=torch.bool)
    assignment.scatter_(-2, assignment_score.argmax(dim=-2, keepdim=True), True)
    selected = logits.masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(selected, dim=-1).masked_fill(~assignment, 0.0)
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    bias = (weights * log_weights).sum(dim=-1)
    bias.masked_fill_(~assignment.any(dim=-1), float("inf"))
    merged_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", weights.to(key.dtype), key
    )
    merged_value = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", weights.to(value.dtype), value
    )
    return merged_key, merged_value, bias


def soft_representatives(seed_queries, key, value, scale, temperature):
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    logits = logits * float(scale)
    assignment_score = torch.log_softmax(logits, dim=-1)
    log_responsibility = torch.log_softmax(assignment_score / temperature, dim=-2)
    responsibility = log_responsibility.exp()

    log_weight_numerator = logits + log_responsibility
    log_weights = log_weight_numerator - torch.logsumexp(
        log_weight_numerator, dim=-1, keepdim=True
    )
    weights = log_weights.exp()
    entropy_bias = (weights * log_weights).sum(dim=-1)
    corrected_bias = (weights * (log_weights - log_responsibility)).sum(dim=-1)

    merged_key = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", weights.to(key.dtype), key
    )
    merged_value = torch.einsum(
        "bhafkjcl,bfklhd->bafjkchd", weights.to(value.dtype), value
    )
    responsibility_entropy = -(responsibility * log_responsibility).sum(dim=-2)
    diagnostics = {
        "mean_key_responsibility_entropy": float(responsibility_entropy.mean().item()),
        "mean_key_max_responsibility": float(responsibility.amax(dim=-2).mean().item()),
    }
    return merged_key, merged_value, corrected_bias, entropy_bias, diagnostics


def route(query, merged_key, merged_value, bias, scale):
    query_flat = rearrange(query, "b a i j h d -> (b h a j) i d")
    key_flat = rearrange(merged_key, "b a f j k c h d -> (b h a j) (f k c) d")
    value_flat = rearrange(merged_value, "b a f j k c h d -> (b h a j) (f k c) d")
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


def error_metrics(output, reference):
    difference = output.float() - reference.float()
    sqerr = float(difference.square().sum().item())
    ref_sq = float(reference.float().square().sum().item())
    return {
        "mse_vs_dense": sqerr / difference.numel(),
        "rel_l2_vs_dense": math.sqrt(sqerr / max(ref_sq, 1e-20)),
        "max_abs_vs_dense": float(difference.abs().max().item()),
    }


def install_probe(stats, pairwise_stats, args):
    import wan.modules.model as wan_model

    temperatures = parse_floats(args.temperatures)
    selected_heads = parse_ints(args.heads)
    record_calls = set(parse_ints(args.record_calls))
    state = {
        "calls_total": 0,
        "grid_size": None,
        "evaluated": [],
        "responsibility_diagnostics": defaultdict(list),
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
        dense = dense_attention(q, k, v, self)

        state["calls_total"] += 1
        call = state["calls_total"]
        frames, height, width = [int(item) for item in grid_size]
        state["grid_size"] = [frames, height, width]
        if call in record_calls:
            for head in selected_heads:
                if not 0 <= head < heads:
                    raise ValueError(f"head index {head} is outside [0, {heads})")
            step = (call - 1) // 40
            layer = (call - 1) % 40
            frame = (step * 7 + layer * 3) % frames
            head_index = torch.tensor(selected_heads, device=q.device)
            q_selected = q.index_select(2, head_index)
            k_selected = k.index_select(2, head_index)
            v_selected = v.index_select(2, head_index)
            dense_selected = dense.index_select(2, head_index)

            q_blocks = video_to_blocks(q_selected, frames, height, width)
            k_blocks = video_to_blocks(k_selected, frames, height, width)
            v_blocks = video_to_blocks(v_selected, frames, height, width)
            dense_blocks = video_to_blocks(dense_selected, frames, height, width)
            query = q_blocks[:, frame : frame + 1].contiguous()
            reference = dense_blocks[:, frame : frame + 1]
            seed_rows = wide_c4_seed_rows(
                query.size(1),
                k_blocks.size(1),
                k_blocks.size(2),
                query.size(3),
                query.size(2),
                query.device,
            )
            seeds = gather_seed_queries(query, seed_rows)
            scale = dim**-0.5

            call_result = {
                "call": call,
                "step": step,
                "layer": layer,
                "query_frame": frame,
                "heads": selected_heads,
                "variants": {},
                "pairwise_corrected_vs_uncorrected": {},
            }

            merged_key, merged_value, hard_bias = hard_representatives(
                seeds, k_blocks, v_blocks, scale
            )
            hard_output = route(query, merged_key, merged_value, hard_bias, scale)
            stats["hard_wide"].add(hard_output, reference)
            call_result["variants"]["hard_wide"] = error_metrics(hard_output, reference)
            del merged_key, merged_value, hard_bias, hard_output

            for temperature in temperatures:
                label = str(temperature).replace(".", "p")
                corrected_name = f"soft_tau_{label}_corrected"
                uncorrected_name = f"soft_tau_{label}_uncorrected"
                (
                    merged_key,
                    merged_value,
                    corrected_bias,
                    uncorrected_bias,
                    diagnostics,
                ) = soft_representatives(seeds, k_blocks, v_blocks, scale, temperature)
                corrected = route(query, merged_key, merged_value, corrected_bias, scale)
                uncorrected = route(query, merged_key, merged_value, uncorrected_bias, scale)
                stats[corrected_name].add(corrected, reference)
                stats[uncorrected_name].add(uncorrected, reference)
                pairwise_stats[f"tau_{label}_corrected_vs_uncorrected"].add(
                    corrected, uncorrected
                )
                call_result["variants"][corrected_name] = error_metrics(corrected, reference)
                call_result["variants"][uncorrected_name] = error_metrics(
                    uncorrected, reference
                )
                call_result["pairwise_corrected_vs_uncorrected"][f"tau_{label}"] = (
                    error_metrics(corrected, uncorrected)
                )
                state["responsibility_diagnostics"][f"tau_{label}"].append(diagnostics)
                del (
                    merged_key,
                    merged_value,
                    corrected_bias,
                    uncorrected_bias,
                    corrected,
                    uncorrected,
                )

            state["evaluated"].append(call_result)
            del q_selected, k_selected, v_selected, dense_selected
            del q_blocks, k_blocks, v_blocks, dense_blocks, query, reference, seeds

        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare fractional-assignment responsibility correction on native Wan QKV."
    )
    parser.add_argument(
        "--config", default="configs/dense_wan_14b_720p_fewstep.yaml"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompt_path", default="prompts/MovieGenVideoBench_extended.txt"
    )
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--heads", default="0,20")
    parser.add_argument("--temperatures", default="0.1,0.25,0.5,1.0")
    parser.add_argument(
        "--record_calls",
        default="1,14,27,40,41,54,67,80,81,94,107,120,121,134,147,160",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, device, args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    stats = defaultdict(ErrorStats)
    pairwise_stats = defaultdict(ErrorStats)
    state = install_probe(stats, pairwise_stats, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    responsibility_summary = {}
    for label, records in state["responsibility_diagnostics"].items():
        responsibility_summary[label] = {
            key: sum(record[key] for record in records) / len(records)
            for key in records[0]
        }
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "grid_size": state["grid_size"],
        "calls_total": state["calls_total"],
        "record_calls": parse_ints(args.record_calls),
        "heads": parse_ints(args.heads),
        "temperatures": parse_floats(args.temperatures),
        "seed_rule": "wide interior [-2,0,1,3], clamped contiguous boundary",
        "wall_seconds": time.perf_counter() - start,
        "variants_vs_dense": {
            name: value.summary() for name, value in sorted(stats.items())
        },
        "pairwise_corrected_vs_uncorrected": {
            name: value.summary() for name, value in sorted(pairwise_stats.items())
        },
        "responsibility_diagnostics": responsibility_summary,
        "per_call": state["evaluated"],
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
