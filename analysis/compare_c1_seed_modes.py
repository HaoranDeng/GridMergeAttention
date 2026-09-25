"""Screen one-seed GridMerge choices on real Wan Q/K/V.

C1 is the only current grid-local setting that can plausibly reach the
``~3x vs FlashAttention`` speed target. Its known problem is dense error. This
script keeps the same C1 merge and full compressed attention math, then tests
whether choosing a better single seed query row can lower that error enough to
justify a future kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.compare_lowc_seed_modes import (
    Stats,
    blocks_to_video,
    load_pipeline,
    run_denoise_only,
    sync,
    timed,
    video_to_blocks,
)
from attention.hard_route_pytorch import biased_sdpa, gather_seed_queries, hard_route_merge_kv_pytorch
from generate import load_prompts


FLOAT_NEG_INF = torch.finfo(torch.float32).min


def ith_seed_rows(q, k):
    key_row = torch.arange(k.size(2), device=q.device).view(1, 1, k.size(2), 1, 1)
    return key_row.expand(q.size(1), k.size(1), k.size(2), q.size(3), 1).long()


def center_seed_rows(q, k):
    row = torch.full((1, 1, 1, 1, 1), q.size(2) // 2, device=q.device, dtype=torch.long)
    return row.expand(q.size(1), k.size(1), k.size(2), q.size(3), 1)


def row_mean_scores(q, k, scale):
    mean_key = k.mean(dim=3)
    scores = torch.einsum("baijhd,bfkhd->bhafkji", q, mean_key).float()
    return (scores * float(scale)).mean(dim=(0, 1))


def top_mean_seed_rows(q, k, scale, local_radius=None):
    scores = row_mean_scores(q, k, scale)
    if local_radius is not None:
        key_row = torch.arange(k.size(2), device=q.device).view(1, 1, k.size(2), 1, 1)
        rows = torch.arange(q.size(2), device=q.device).view(1, 1, 1, 1, q.size(2))
        local = (rows >= key_row - local_radius) & (rows <= key_row + local_radius)
        scores = scores.masked_fill(~local, FLOAT_NEG_INF)
    return scores.argmax(dim=-1, keepdim=True).long()


def top_lse_seed_rows(q, k, scale):
    logits = torch.einsum("baijhd,bfklhd->bhafkjil", q, k).float()
    scores = torch.logsumexp(logits * float(scale), dim=-1).mean(dim=(0, 1))
    return scores.argmax(dim=-1, keepdim=True).long()


def seed_rows_for_mode(q, k, scale, mode):
    if mode == "ith":
        return ith_seed_rows(q, k)
    if mode == "center":
        return center_seed_rows(q, k)
    if mode == "top_mean_all":
        return top_mean_seed_rows(q, k, scale)
    if mode == "top_mean_local3":
        return top_mean_seed_rows(q, k, scale, local_radius=1)
    if mode == "top_mean_local5":
        return top_mean_seed_rows(q, k, scale, local_radius=2)
    if mode == "top_lse_all":
        return top_lse_seed_rows(q, k, scale)
    raise ValueError(f"unknown C1 seed mode: {mode}")


def c1_attention_with_seed_mode(q, k, v, scale, mode):
    seed_rows = seed_rows_for_mode(q, k, scale, mode)
    seed_queries = gather_seed_queries(q, seed_rows)
    merged_k, merged_v, bias = hard_route_merge_kv_pytorch(seed_queries, k, v, scale)
    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_k, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_v, "b a f j k c h d -> (b h a j) (f k c) d")
    route_bias = rearrange(bias, "b h a f k j c -> (b h a j) (f k c)")
    out = biased_sdpa(query, key, value, route_bias, scale=scale)
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


def c1_attention(q, k, v, grid_size, scale, mode, query_outer_chunk, head_chunk):
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = video_to_blocks(q, frames, height, width)
    k_blocks = video_to_blocks(k, frames, height, width)
    v_blocks = video_to_blocks(v, frames, height, width)
    out = torch.empty_like(q_blocks)

    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            out[:, q_start:q_end, :, :, h_start:h_end] = c1_attention_with_seed_mode(
                q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous(),
                k_blocks[:, :, :, :, h_start:h_end].contiguous(),
                v_blocks[:, :, :, :, h_start:h_end].contiguous(),
                scale,
                mode,
            )
    return blocks_to_video(out)


def install_probe(stats, args):
    import wan.modules.model as wan_model

    modes = [item.strip() for item in args.modes.replace(",", " ").split() if item.strip()]
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

        dense, seconds = timed(lambda: dense_attention(q, k, v, self))
        stats["dense"].add_time(seconds)
        for mode in modes:
            out, seconds = timed(
                lambda mode=mode: c1_attention(
                    q,
                    k,
                    v,
                    grid_size,
                    scale,
                    mode,
                    args.query_outer_chunk,
                    args.head_chunk,
                )
            )
            stats[mode].add_time(seconds)
            stats[mode].add_error(out, dense)
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", default="ith,center,top_mean_local3,top_mean_local5,top_mean_all,top_lse_all")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=8)
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
    state = install_probe(stats, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    dense_seconds = stats["dense"].seconds
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "modes": args.modes,
        "query_outer_chunk": args.query_outer_chunk,
        "head_chunk": args.head_chunk,
        "grid_size": state["grid_size"],
        "calls_total": state["calls_total"],
        "wall_seconds": wall_seconds,
        "variants_vs_dense": {
            name: value.summary(dense_seconds=dense_seconds if name != "dense" else None)
            for name, value in sorted(stats.items())
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
