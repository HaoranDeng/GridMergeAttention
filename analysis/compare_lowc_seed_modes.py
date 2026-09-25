"""Compare low-C hard-route seed choices on real Wan Q/K/V.

The fast Triton kernels make C2/C3 attractive for speed, but their local seed
choice has noticeably larger dense error than C4. This script keeps the same
hard-route merge and full compressed attention math, then screens whether
content-adaptive seed rows can recover enough C2/C3 quality to justify a future
kernel.
"""

from __future__ import annotations

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


FLOAT_NEG_INF = torch.finfo(torch.float32).min


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
        result = {
            "calls": self.calls,
            "seconds": self.seconds,
            "mean_ms": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
        }
        if dense_seconds is not None and self.seconds:
            result["speedup_vs_dense_attention"] = dense_seconds / self.seconds
        if self.numel:
            result.update(
                {
                    "rel_l2": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                    "mse": self.sqerr / self.numel,
                    "max_abs": self.max_abs,
                }
            )
        return result


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


def local_seed_rows(q, k, count):
    query_outer = torch.arange(q.size(1), device=q.device)
    key_outer = torch.arange(k.size(1), device=q.device)
    return local_c4_seed_rows(
        query_outer,
        key_outer,
        k.size(2),
        q.size(3),
        q.size(2),
    )[..., :count].contiguous()


def ith_row_grid(q, k):
    key_row = torch.arange(k.size(2), device=q.device).view(1, 1, k.size(2), 1)
    return key_row.expand(q.size(1), k.size(1), k.size(2), q.size(3))


def all_rows_with_ith_first(q, k):
    first = ith_row_grid(q, k)
    rows = torch.arange(q.size(2), device=q.device).view(1, 1, 1, 1, q.size(2))
    rows = rows.expand(*first.shape, q.size(2))
    extra = rows.masked_fill(rows == first.unsqueeze(-1), q.size(2)).sort(dim=-1).values
    return torch.cat((first.unsqueeze(-1).long(), extra[..., : q.size(2) - 1].long()), dim=-1)


def gather_candidate_queries(q, candidate_rows):
    return gather_seed_queries(q, candidate_rows)


def prefilter_rows(q, k, scale, candidate_count):
    mean_key = k.mean(dim=3)
    row_score = torch.einsum("baijhd,bfkhd->bhafkji", q, mean_key).float()
    row_score = (row_score * float(scale)).mean(dim=(0, 1))
    first = ith_row_grid(q, k)
    row_score = row_score.scatter(-1, first.unsqueeze(-1), FLOAT_NEG_INF)
    extra = row_score.topk(candidate_count - 1, dim=-1).indices.long()
    return torch.cat((first.unsqueeze(-1).long(), extra), dim=-1)


def candidate_logits(q, k, candidate_rows, scale):
    queries = gather_candidate_queries(q, candidate_rows)
    logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", queries, k).float()
    return (logits * float(scale)).mean(dim=(0, 1))


def greedy_from_candidates(candidate_rows, logits, count):
    first_pos = torch.zeros(candidate_rows.shape[:-1], device=candidate_rows.device, dtype=torch.long)
    best = logits.gather(
        dim=-2,
        index=first_pos.unsqueeze(-1).unsqueeze(-1).expand(*first_pos.shape, 1, logits.size(-1)),
    ).squeeze(-2)
    available = torch.ones(candidate_rows.shape, device=candidate_rows.device, dtype=torch.bool)
    available.scatter_(dim=-1, index=first_pos.unsqueeze(-1), value=False)
    chosen = [first_pos]
    for _ in range(count - 1):
        row_score = torch.maximum(best.unsqueeze(-2), logits).mean(dim=-1)
        pos = row_score.masked_fill(~available, FLOAT_NEG_INF).argmax(dim=-1)
        chosen.append(pos)
        available.scatter_(dim=-1, index=pos.unsqueeze(-1), value=False)
        value = logits.gather(
            dim=-2,
            index=pos.unsqueeze(-1).unsqueeze(-1).expand(*pos.shape, 1, logits.size(-1)),
        ).squeeze(-2)
        best = torch.maximum(best, value)
    return candidate_rows.gather(dim=-1, index=torch.stack(chosen, dim=-1))


def adaptive_seed_rows(q, k, scale, count, mode):
    first = ith_row_grid(q, k)
    if mode.startswith("prefilter"):
        candidate_count = int(mode.removeprefix("prefilter").split("_", 1)[0])
        candidates = prefilter_rows(q, k, scale, candidate_count)
        return greedy_from_candidates(candidates, candidate_logits(q, k, candidates, scale), count)

    logits = torch.einsum("baijhd,bfklhd->bhafkjil", q, k).float()
    logits = (logits * float(scale)).mean(dim=(0, 1))
    if mode == "top_mean":
        score = logits.mean(dim=-1)
    elif mode == "top_lse":
        score = torch.logsumexp(logits, dim=-1)
    elif mode == "greedy_cover":
        return greedy_from_candidates(all_rows_with_ith_first(q, k), logits, count)
    else:
        raise ValueError(f"unknown adaptive mode: {mode}")

    score = score.scatter(-1, first.unsqueeze(-1), FLOAT_NEG_INF)
    extra = score.topk(count - 1, dim=-1).indices.long()
    return torch.cat((first.unsqueeze(-1).long(), extra), dim=-1)


def seed_rows_for_mode(q, k, scale, count, mode):
    if mode == "local":
        return local_seed_rows(q, k, count)
    return adaptive_seed_rows(q, k, scale, count, mode)


def hard_route_with_seed_mode(q, k, v, scale, count, mode):
    seed_rows = seed_rows_for_mode(q, k, scale, count, mode)
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


def hard_route_attention(q, k, v, grid_size, scale, count, mode, query_outer_chunk, head_chunk):
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = video_to_blocks(q, frames, height, width)
    k_blocks = video_to_blocks(k, frames, height, width)
    v_blocks = video_to_blocks(v, frames, height, width)
    out = torch.empty_like(q_blocks)

    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            out[:, q_start:q_end, :, :, h_start:h_end] = hard_route_with_seed_mode(
                q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous(),
                k_blocks[:, :, :, :, h_start:h_end].contiguous(),
                v_blocks[:, :, :, :, h_start:h_end].contiguous(),
                scale,
                count,
                mode,
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


def install_probe(stats, args):
    import wan.modules.model as wan_model

    modes = [item.strip() for item in args.seed_modes.replace(",", " ").split() if item.strip()]
    counts = [int(item) for item in args.seed_counts.replace(",", " ").split() if item.strip()]
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
        for count in counts:
            for mode in modes:
                name = f"c{count}_{mode}"
                out, seconds = timed(
                    lambda count=count, mode=mode: hard_route_attention(
                        q,
                        k,
                        v,
                        grid_size,
                        scale,
                        count,
                        mode,
                        args.query_outer_chunk,
                        args.head_chunk,
                    )
                )
                stats[name].add_time(seconds)
                stats[name].add_error(out, dense)
        return self.o(dense.flatten(2))

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
    parser = argparse.ArgumentParser(description="Screen low-C hard-route seed choices on real Wan QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed_counts", default="2,3")
    parser.add_argument("--seed_modes", default="local,top_mean,top_lse,greedy_cover,prefilter12_greedy")
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
        "seed_counts": args.seed_counts,
        "seed_modes": args.seed_modes,
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
