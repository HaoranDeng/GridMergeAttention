"""Profile PyTorch stage costs for low-C prefilter-greedy seed selection.

This is a diagnostic script for the C3 adaptive-seed direction. It uses real
Wan Q/K/V, keeps the same hard-route merge and full compressed attention math,
and reports where the PyTorch analysis path spends time.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

from analysis.compare_lowc_seed_modes import (
    Stats,
    blocks_to_video,
    greedy_from_candidates,
    hard_route_attention,
    load_pipeline,
    prefilter_rows,
    run_denoise_only,
    sync,
    video_to_blocks,
)
from attention.hard_route_pytorch import biased_sdpa, gather_seed_queries, hard_route_merge_kv_pytorch
from generate import load_prompts


class StageStats:
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0

    def add(self, seconds):
        self.calls += 1
        self.seconds += float(seconds)

    def summary(self):
        return {
            "calls": self.calls,
            "seconds": self.seconds,
            "mean_ms": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
        }


def timed_stage(stats, name, fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    bucket = stats[name]
    seconds = time.perf_counter() - start
    if hasattr(bucket, "add"):
        bucket.add(seconds)
    else:
        bucket.add_time(seconds)
    return out


def profile_prefilter_greedy_chunk(q, k, v, scale, count, candidate_count, stage_stats):
    candidates = timed_stage(
        stage_stats,
        "prefilter_rows",
        lambda: prefilter_rows(q, k, scale, candidate_count),
    )
    logits = timed_stage(
        stage_stats,
        "candidate_logits",
        lambda: (
            torch.einsum(
                "bafkjchd,bfklhd->bhafkjcl",
                gather_seed_queries(q, candidates),
                k,
            ).float()
            * float(scale)
        ).mean(dim=(0, 1)),
    )
    seed_rows = timed_stage(
        stage_stats,
        "greedy_select",
        lambda: greedy_from_candidates(candidates, logits, count),
    )
    seed_queries = timed_stage(
        stage_stats,
        "gather_final_seeds",
        lambda: gather_seed_queries(q, seed_rows),
    )
    merged_k, merged_v, bias = timed_stage(
        stage_stats,
        "merge_kv",
        lambda: hard_route_merge_kv_pytorch(seed_queries, k, v, scale),
    )

    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_k, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_v, "b a f j k c h d -> (b h a j) (f k c) d")
    route_bias = rearrange(bias, "b h a f k j c -> (b h a j) (f k c)")
    out = timed_stage(
        stage_stats,
        "compressed_sdpa",
        lambda: biased_sdpa(query, key, value, route_bias, scale=scale),
    )
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


def profile_prefilter_greedy_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    count,
    candidate_count,
    query_outer_chunk,
    head_chunk,
    stage_stats,
):
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = video_to_blocks(q, frames, height, width)
    k_blocks = video_to_blocks(k, frames, height, width)
    v_blocks = video_to_blocks(v, frames, height, width)
    out = torch.empty_like(q_blocks)

    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            out[:, q_start:q_end, :, :, h_start:h_end] = profile_prefilter_greedy_chunk(
                q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous(),
                k_blocks[:, :, :, :, h_start:h_end].contiguous(),
                v_blocks[:, :, :, :, h_start:h_end].contiguous(),
                scale,
                count,
                candidate_count,
                stage_stats,
            )
    return blocks_to_video(out)


def install_probe(stats, stage_stats, args):
    import wan.modules.model as wan_model

    state = {"calls_total": 0, "calls_current_prompt": 0, "grid_size": None}

    def dense_attention(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len, heads, dim = *x.shape[:2], self.num_heads, self.head_dim
        q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(batch, seq_len, heads, dim)
        k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(batch, seq_len, heads, dim)
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        scale = dim**-0.5

        state["grid_size"] = [int(value) for value in grid_size]
        state["calls_total"] += 1
        state["calls_current_prompt"] += 1
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] <= args.max_calls
        if not should_record:
            out = dense_attention(q, k, v, self)
            return self.o(out.flatten(2))

        dense = timed_stage(stats, "dense", lambda: dense_attention(q, k, v, self))
        local = timed_stage(
            stats,
            f"c{args.count}_local",
            lambda: hard_route_attention(
                q,
                k,
                v,
                grid_size,
                scale,
                args.count,
                "local",
                args.query_outer_chunk,
                args.head_chunk,
            ),
        )
        adaptive = timed_stage(
            stats,
            f"c{args.count}_prefilter{args.candidates}_greedy",
            lambda: profile_prefilter_greedy_attention(
                q,
                k,
                v,
                grid_size,
                scale,
                args.count,
                args.candidates,
                args.query_outer_chunk,
                args.head_chunk,
                stage_stats,
            ),
        )
        stats[f"c{args.count}_local_error"].add_error(local, dense)
        stats[f"c{args.count}_prefilter{args.candidates}_greedy_error"].add_error(adaptive, dense)
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Profile low-C prefilter-greedy stages on real Wan QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=4)
    parser.add_argument("--query_outer_chunk", type=int, default=1)
    parser.add_argument("--head_chunk", type=int, default=1)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, torch.device("cuda"), args.use_ema)
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    stats = defaultdict(Stats)
    stage_stats = defaultdict(StageStats)
    state = install_probe(stats, stage_stats, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "count": args.count,
        "candidates": args.candidates,
        "query_outer_chunk": args.query_outer_chunk,
        "head_chunk": args.head_chunk,
        "grid_size": state["grid_size"],
        "calls_total": state["calls_total"],
        "wall_seconds": wall_seconds,
        "variants": {name: value.summary() for name, value in sorted(stats.items())},
        "stages": {name: value.summary() for name, value in sorted(stage_stats.items())},
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
