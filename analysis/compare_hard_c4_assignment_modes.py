import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

from analysis.compare_hard_c4_seed_modes import (
    Stats,
    blocks_to_video,
    load_pipeline,
    run_denoise_only,
    sync,
    timed,
    video_to_blocks,
)
from attention.hard_route_pytorch import (
    biased_sdpa,
    cached_local_c4_seed_rows,
    gather_seed_queries,
)
from generate import load_prompts


FLOAT_NEG_INF = torch.finfo(torch.float32).min


class AssignmentStats:
    def __init__(self):
        self.changed = 0
        self.total = 0

    def add(self, changed, total):
        self.changed += int(changed)
        self.total += int(total)

    def summary(self):
        return {
            "changed_vs_raw": self.changed,
            "total": self.total,
            "changed_fraction_vs_raw": self.changed / self.total if self.total else 0.0,
        }


def assignment_score(logits, mode):
    if mode == "raw_dot":
        return logits
    if mode == "key_softmax":
        return torch.log_softmax(logits, dim=-1)
    raise ValueError(f"unknown assignment mode: {mode}")


def merge_kv_with_assignment_mode(seed_queries, key, value, scale, mode):
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    logits = logits * float(scale)

    raw_best = logits.argmax(dim=-2, keepdim=True)
    best = assignment_score(logits, mode).argmax(dim=-2, keepdim=True)
    changed = int((best != raw_best).sum().item())
    total = best.numel()

    assignment = torch.zeros_like(logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best, value=True)

    selected_logits = logits.masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(selected_logits, dim=-1).masked_fill(~assignment, 0.0)
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(~assignment.any(dim=-1), float("inf"))

    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", weights.to(key.dtype), key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", weights.to(value.dtype), value)
    return merged_key, merged_value, routing_bias, changed, total


def hard_c4_assignment_chunk(q, k, v, scale, mode, query_outer_offset):
    seed_rows = cached_local_c4_seed_rows(
        q.device,
        query_outer_offset,
        q.size(1),
        k.size(1),
        k.size(2),
        q.size(3),
        q.size(2),
    )
    seed_queries = gather_seed_queries(q, seed_rows)
    merged_keys, merged_values, routing_bias, changed, total = merge_kv_with_assignment_mode(
        seed_queries,
        k,
        v,
        scale,
        mode,
    )

    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    out = biased_sdpa(query, key, value, bias, scale=scale)
    out = rearrange(out, "(b h a j) i d -> b a i j h d", b=q.size(0), h=q.size(4), a=q.size(1), j=q.size(3))
    return out, changed, total


def hard_c4_assignment_attention(q, k, v, grid_size, scale, mode, query_outer_chunk, head_chunk):
    frames, height, width = [int(x) for x in grid_size]
    q_blocks = video_to_blocks(q, frames, height, width)
    k_blocks = video_to_blocks(k, frames, height, width)
    v_blocks = video_to_blocks(v, frames, height, width)
    out = torch.empty_like(q_blocks)
    assignment_stats = AssignmentStats()

    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            chunk, changed, total = hard_c4_assignment_chunk(
                q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous(),
                k_blocks[:, :, :, :, h_start:h_end].contiguous(),
                v_blocks[:, :, :, :, h_start:h_end].contiguous(),
                scale,
                mode,
                q_start,
            )
            out[:, q_start:q_end, :, :, h_start:h_end] = chunk
            assignment_stats.add(changed, total)
    return blocks_to_video(out), assignment_stats


def install_probe(stats, pair_stats, assignment_stats, args):
    import wan.modules.model as wan_model

    modes = [mode.strip() for mode in args.assignment_modes.replace(",", " ").split() if mode.strip()]
    state = {"calls_total": 0, "calls_current_prompt": 0, "grid_size": None}

    def dense_attention(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len, heads, dim = *x.shape[:2], self.num_heads, self.head_dim
        q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(batch, seq_len, heads, dim)
        k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(batch, seq_len, heads, dim)
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        scale = dim**-0.5

        state["grid_size"] = [int(value) for value in grid_size]
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] < args.max_calls
        state["calls_current_prompt"] += 1
        state["calls_total"] += 1

        if should_record:
            dense, seconds = timed(lambda: dense_attention(q, k, v, self))
            stats["dense"].add_time(seconds)

            outputs = {}
            for mode in modes:
                (out, assign), seconds = timed(
                    lambda mode=mode: hard_c4_assignment_attention(
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
                outputs[mode] = out
                stats[mode].add_time(seconds)
                stats[mode].add_error(out, dense)
                assignment_stats[mode].add(assign.changed, assign.total)
            if "raw_dot" in outputs:
                for mode, out in outputs.items():
                    if mode != "raw_dot":
                        pair_stats[f"{mode}_vs_raw_dot"].add_time(0.0)
                        pair_stats[f"{mode}_vs_raw_dot"].add_error(out, outputs["raw_dot"])
            x = dense
        else:
            x = dense_attention(q, k, v, self)
        return self.o(x.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Compare hard C4 key-to-seed assignment scores on real Wan QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assignment_modes", default="raw_dot,key_softmax")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=4)
    parser.add_argument("--query_outer_chunk", type=int, default=1)
    parser.add_argument("--head_chunk", type=int, default=2)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]

    stats = defaultdict(Stats)
    pair_stats = defaultdict(Stats)
    assignment = defaultdict(AssignmentStats)
    state = install_probe(stats, pair_stats, assignment, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    dense_seconds = stats["dense"].seconds
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "query_outer_chunk": args.query_outer_chunk,
        "head_chunk": args.head_chunk,
        "grid_size": state["grid_size"],
        "wall_seconds": time.perf_counter() - start,
        "variants_vs_dense": {
            name: value.summary(dense_seconds=dense_seconds if name != "dense" else None)
            for name, value in sorted(stats.items())
        },
        "assignment_vs_raw_dot": {name: value.summary() for name, value in sorted(assignment.items())},
        "pairwise": {name: value.summary() for name, value in sorted(pair_stats.items())},
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
