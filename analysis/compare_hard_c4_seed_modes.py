import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf

from attention.hard_route_pytorch import (
    biased_sdpa,
    gather_seed_queries,
    hard_route_merge_kv_pytorch,
    local_c4_seed_rows,
)
from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


HASH_MOD = 2147483647


class Stats:
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0
        self.max_abs = 0.0

    def add_time(self, seconds):
        self.calls += 1
        self.seconds += float(seconds)

    def add_error(self, out, ref):
        out_f = out.float().reshape(-1)
        ref_f = ref.float().reshape(-1)
        diff = out_f - ref_f
        self.sqerr += float(torch.dot(diff, diff).item())
        self.ref_sq += float(torch.dot(ref_f, ref_f).clamp_min(1e-20).item())
        self.numel += diff.numel()
        self.max_abs = max(self.max_abs, float(diff.abs().max().item()))

    def summary(self, dense_seconds=None):
        payload = {
            "calls": self.calls,
            "seconds": self.seconds,
            "mean_ms": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
        }
        if dense_seconds is not None and self.seconds:
            payload["speedup_vs_dense_attention"] = dense_seconds / self.seconds
        if self.numel:
            payload.update(
                {
                    "mse": self.sqerr / self.numel,
                    "rel_l2": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                    "max_abs": self.max_abs,
                }
            )
        return payload


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def video_to_blocks(x, frames, height, width):
    batch, _seq_len, heads, dim = x.shape
    return x.reshape(batch, frames, height, width, heads, dim)


def blocks_to_video(x):
    batch, frames, height, width, heads, dim = x.shape
    return x.reshape(batch, frames * height * width, heads, dim)


def hashed_candidate_scores(base, num_choices):
    candidate = torch.arange(num_choices, device=base.device).view(*((1,) * base.ndim), num_choices)
    scores = (base.unsqueeze(-1).remainder(HASH_MOD) + candidate * 374761393).remainder(HASH_MOD)
    scores = (scores * (scores + 1274126177)).remainder(HASH_MOD)
    scores = (scores * (scores + 60493)).remainder(HASH_MOD)
    return scores * num_choices + candidate, candidate


def random_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, seed, include_ith):
    key_row = torch.arange(key_rows, device=query_outer.device).view(1, 1, key_rows, 1)
    query_col = torch.arange(query_columns, device=query_outer.device).view(1, 1, 1, query_columns)
    base = (
        query_outer.view(-1, 1, 1, 1) * 73856093
        + key_outer.view(1, -1, 1, 1) * 19349663
        + key_row * 83492791
        + query_col * 2654435761
        + int(seed)
    )
    scores, candidate = hashed_candidate_scores(base, query_rows)
    if include_ith:
        scores = scores.masked_fill(candidate == key_row.unsqueeze(-1), -1)
        random_rows = scores.topk(3, dim=-1).indices.long()
        return torch.cat((key_row.expand_as(base).unsqueeze(-1).long(), random_rows), dim=-1)
    return scores.topk(4, dim=-1).indices.long()


def nearest_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, prefer):
    key_row = torch.arange(key_rows, device=query_outer.device).view(1, 1, key_rows, 1, 1)
    row = torch.arange(query_rows, device=query_outer.device).view(1, 1, 1, 1, query_rows)
    relative = row - key_row
    if prefer == "right":
        tie = torch.where(relative == 0, 0, torch.where(relative > 0, 1, 2))
    else:
        tie = torch.where(relative == 0, 0, torch.where(relative < 0, 1, 2))
    score = relative.abs() * 10 + tie
    seed_rows = score.topk(4, dim=-1, largest=False).indices.long()
    return seed_rows.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns, 4)


def alternating_nearest_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows):
    left = nearest_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, prefer="left")
    right = nearest_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, prefer="right")
    key_row = torch.arange(key_rows, device=query_outer.device).view(1, 1, key_rows, 1, 1)
    return torch.where(key_row.remainder(2) == 0, left, right)


def ring_offset_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, offsets):
    key_row = torch.arange(key_rows, device=query_outer.device).view(1, 1, key_rows, 1)
    rows = [torch.remainder(key_row + int(offset), query_rows) for offset in offsets]
    seed_rows = torch.stack(rows, dim=-1).long()
    return seed_rows.expand(query_outer.numel(), key_outer.numel(), key_rows, query_columns, 4)


def seed_rows_for_mode(mode, device, query_outer_offset, query_outer_count, key_outer_count, key_rows, query_columns, query_rows, seed):
    query_outer = torch.arange(query_outer_offset, query_outer_offset + query_outer_count, device=device)
    key_outer = torch.arange(key_outer_count, device=device)
    if mode == "local_with_ith":
        return local_c4_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows)
    if mode == "nearest_right_with_ith":
        return nearest_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, prefer="right")
    if mode == "nearest_alternating_with_ith":
        return alternating_nearest_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows)
    if mode == "ring_stride5_with_ith":
        return ring_offset_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, (0, 5, 10, 15))
    if mode == "ring_stride7_with_ith":
        return ring_offset_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, (0, 7, 14, 21))
    if mode == "ring_quarter_with_ith":
        return ring_offset_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, (0, 8, 15, 22))
    if mode == "ring_symmetric7_with_ith":
        return ring_offset_seed_rows(query_outer, key_outer, key_rows, query_columns, query_rows, (0, -7, 7, 14))
    if mode == "random_with_ith":
        return random_seed_rows(
            query_outer,
            key_outer,
            key_rows,
            query_columns,
            query_rows,
            seed,
            include_ith=True,
        )
    if mode == "fully_random":
        return random_seed_rows(
            query_outer,
            key_outer,
            key_rows,
            query_columns,
            query_rows,
            seed,
            include_ith=False,
        )
    raise ValueError(f"unknown seed mode: {mode}")


def hard_c4_seed_mode_chunk(q, k, v, scale, seed_rows):
    seed_queries = gather_seed_queries(q, seed_rows)
    merged_keys, merged_values, routing_bias = hard_route_merge_kv_pytorch(seed_queries, k, v, scale)
    query = rearrange(q, "b a i j h d -> (b h a j) i d")
    key = rearrange(merged_keys, "b a f j k c h d -> (b h a j) (f k c) d")
    value = rearrange(merged_values, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    out = biased_sdpa(query, key, value, bias, scale=scale)
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=q.size(0),
        h=q.size(4),
        a=q.size(1),
        j=q.size(3),
    )


def hard_c4_seed_mode_attention(q, k, v, grid_size, scale, mode, seed, query_outer_chunk, head_chunk):
    frames, height, width = [int(x) for x in grid_size]
    q_blocks = video_to_blocks(q, frames, height, width)
    k_blocks = video_to_blocks(k, frames, height, width)
    v_blocks = video_to_blocks(v, frames, height, width)
    out = torch.empty_like(q_blocks)

    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        seed_rows = seed_rows_for_mode(
            mode,
            q.device,
            q_start,
            q_end - q_start,
            k_blocks.size(1),
            k_blocks.size(2),
            q_blocks.size(3),
            q_blocks.size(2),
            seed,
        )
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            out[:, q_start:q_end, :, :, h_start:h_end] = hard_c4_seed_mode_chunk(
                q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous(),
                k_blocks[:, :, :, :, h_start:h_end].contiguous(),
                v_blocks[:, :, :, :, h_start:h_end].contiguous(),
                scale,
                seed_rows,
            )
    return blocks_to_video(out)


def load_pipeline(config_path, checkpoint, dtype, device, use_ema):
    config = OmegaConf.load(config_path)
    pipeline = TokenMergingWanPipeline(config, device=device)
    load_generator_checkpoint(pipeline, checkpoint, use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()
    return config, pipeline


def install_probe(stats, pair_stats, args):
    import wan.modules.model as wan_model

    modes = [mode.strip() for mode in args.seed_modes.replace(",", " ").split() if mode.strip()]
    state = {"calls_total": 0, "calls_current_prompt": 0, "grid_size": None}

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def run_seed_mode(q, k, v, grid_size, scale, mode):
        return hard_c4_seed_mode_attention(
            q,
            k,
            v,
            grid_size,
            scale,
            mode=mode,
            seed=args.seed_mode_seed,
            query_outer_chunk=args.query_outer_chunk,
            head_chunk=args.head_chunk,
        )

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
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)

            outputs = {}
            for mode in modes:
                out, seconds = timed(lambda mode=mode: run_seed_mode(q, k, v, grid_size, scale, mode))
                outputs[mode] = out
                stats[mode].add_time(seconds)
                stats[mode].add_error(out, dense)

            if "local_with_ith" in outputs:
                local = outputs["local_with_ith"]
                for mode, out in outputs.items():
                    if mode != "local_with_ith":
                        pair_stats[f"{mode}_vs_local_with_ith"].add_error(out, local)
            x = dense
        else:
            x = run_dense(q, k, v, self)

        return self.o(x.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


@torch.inference_mode()
def run_denoise_only(pipeline, config, prompt, seed, dtype):
    device = torch.device("cuda")
    set_seed(seed)
    noise = torch.randn(list(config.image_or_video_shape), device=device, dtype=dtype)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt] * noise.shape[0])

    noisy = noise
    for index, timestep in enumerate(config.denoising_step_list[:-1]):
        _, pred = pipeline.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=device) * timestep,
        )
        next_timestep = config.denoising_step_list[index + 1] * torch.ones(
            noise.shape[:2],
            dtype=torch.long,
            device=device,
        )
        noisy = pipeline.scheduler.add_noise(
            pred.flatten(0, 1),
            torch.randn_like(pred.flatten(0, 1)),
            next_timestep.flatten(0, 1),
        ).unflatten(0, noise.shape[:2])


def parse_args():
    parser = argparse.ArgumentParser(description="Compare hard C4 seed-row choices on real Wan self-attention QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_mode_seed", type=int, default=0)
    parser.add_argument("--seed_modes", default="local_with_ith,random_with_ith,fully_random")
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
    state = install_probe(stats, pair_stats, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    dense_seconds = stats["dense"].seconds
    variants = {}
    for name, value in sorted(stats.items()):
        variants[name] = value.summary(dense_seconds=dense_seconds if name != "dense" else None)
    pairwise = {name: value.summary() for name, value in sorted(pair_stats.items())}

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
        "seed_mode_seed": args.seed_mode_seed,
        "grid_size": state["grid_size"],
        "wall_seconds": wall_seconds,
        "variants_vs_dense": variants,
        "pairwise": pairwise,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
