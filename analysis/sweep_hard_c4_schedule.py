"""Sweep hard-C4 query/head chunk schedules on the same real Wan QKV calls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.compare_hard_c4_backends_dense import load_pipeline  # noqa: E402
from attention.hard_c4_wan21_1_3b_480p_triton import hard_c4_rt_attention_triton_blocks  # noqa: E402
from generate import load_prompts  # noqa: E402
from utils.misc import set_seed  # noqa: E402


@dataclass(frozen=True)
class Schedule:
    q_chunk: int
    h_chunk: int

    @property
    def name(self) -> str:
        return f"q{self.q_chunk}_h{self.h_chunk}"


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
            "total_seconds": self.seconds,
            "mean_ms_per_call": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
        }
        if dense_seconds is not None and self.seconds:
            payload["speedup_vs_dense_attention"] = dense_seconds / self.seconds
        if self.numel:
            payload.update(
                {
                    "mse_vs_dense": self.sqerr / self.numel,
                    "rel_l2_vs_dense": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                    "max_abs_vs_dense": self.max_abs,
                }
            )
        return payload


def parse_schedules(text):
    schedules = []
    for item in text.replace(",", " ").split():
        q_text, h_text = item.lower().replace("x", ":").split(":")
        schedules.append(Schedule(int(q_text), int(h_text)))
    return schedules


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def dense_attention(q, k, v, self):
    import wan.modules.model as wan_model

    return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)


def hard_c4_schedule(q, k, v, grid_size, scale, schedule):
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = q.reshape(q.size(0), frames, height, width, q.size(2), q.size(3))
    k_blocks = k.reshape(k.size(0), frames, height, width, k.size(2), k.size(3))
    v_blocks = v.reshape(v.size(0), frames, height, width, v.size(2), v.size(3))
    out = torch.empty_like(q_blocks)
    heads = q_blocks.size(4)

    for q_start in range(0, frames, schedule.q_chunk):
        q_end = min(q_start + schedule.q_chunk, frames)
        q_chunk = q_blocks[:, q_start:q_end]
        out_slice = out[:, q_start:q_end]
        for h_start in range(0, heads, schedule.h_chunk):
            h_end = min(h_start + schedule.h_chunk, heads)
            hard_c4_rt_attention_triton_blocks(
                q_chunk,
                k_blocks,
                v_blocks,
                scale,
                output=out_slice,
                head_start=h_start,
                head_count=h_end - h_start,
            )
    return out.reshape(q.size(0), -1, q.size(2), q.size(3))


def validate_schedule(schedule, frames, heads):
    # The current compressed K/V workspace uses 32-bit element offsets safely
    # when q_chunk * h_chunk <= 21 * 6 for Wan 480p/H200.
    if schedule.q_chunk > frames or schedule.h_chunk > heads:
        return False
    return schedule.q_chunk * schedule.h_chunk <= 126


def install_probe(stats, args, schedules):
    import wan.modules.model as wan_model

    state = {
        "calls_total": 0,
        "calls_recorded": 0,
        "calls_current_prompt": 0,
        "grid_size": None,
        "shape": None,
        "did_warmup": False,
    }

    def warmup(q, k, v, grid_size, scale, self):
        dense_attention(q, k, v, self)
        for schedule in schedules:
            hard_c4_schedule(q, k, v, grid_size, scale, schedule)
        sync()

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

        state["calls_total"] += 1
        state["calls_current_prompt"] += 1
        state["grid_size"] = [int(value) for value in grid_size]
        state["shape"] = {"batch": batch, "seq_len": seq_len, "heads": heads, "dim": dim}

        should_record = args.max_calls <= 0 or state["calls_current_prompt"] <= args.max_calls
        if not should_record:
            return self.o(dense_attention(q, k, v, self).flatten(2))

        active_schedules = [
            schedule for schedule in schedules if validate_schedule(schedule, int(grid_size[0]), heads)
        ]
        if not state["did_warmup"]:
            warmup(q, k, v, grid_size, scale, self)
            state["did_warmup"] = True

        dense, dense_seconds = timed(lambda: dense_attention(q, k, v, self))
        stats["sdpa"].add_time(dense_seconds)
        for schedule in active_schedules:
            out, seconds = timed(lambda schedule=schedule: hard_c4_schedule(q, k, v, grid_size, scale, schedule))
            stats[schedule.name].add_time(seconds)
            stats[schedule.name].add_error(out, dense)
        state["calls_recorded"] += 1
        return self.o(dense.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


@torch.inference_mode()
def run_full_denoise(pipeline, config, prompt, seed, dtype):
    device = torch.device("cuda")
    set_seed(seed)
    noise = torch.randn(list(config.image_or_video_shape), device=device, dtype=dtype)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt] * noise.shape[0])
    noisy = noise
    timesteps = pipeline.denoising_step_list.to(device=device)
    for index, current_timestep in enumerate(timesteps):
        timestep = current_timestep.expand(noise.shape[:2])
        _, pred = pipeline.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        if index + 1 < len(timesteps):
            next_timestep = timesteps[index + 1].expand(noise.shape[:2])
            noisy = pipeline.scheduler.add_noise(
                pred.flatten(0, 1),
                torch.randn_like(pred.flatten(0, 1)),
                next_timestep.flatten(0, 1),
            ).unflatten(0, noise.shape[:2])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=12)
    parser.add_argument("--schedules", default="21:6,10:12,14:9,15:8,18:7,21:4,7:12")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    schedules = parse_schedules(args.schedules)
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    stats = defaultdict(Stats)
    state = install_probe(stats, args, schedules)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        run_full_denoise(pipeline, config, prompt, args.seed + offset, dtype)
    sync()

    dense_seconds = stats["sdpa"].seconds
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "schedules": [schedule.name for schedule in schedules],
        "grid_size": state["grid_size"],
        "shape": state["shape"],
        "calls_total": state["calls_total"],
        "calls_recorded": state["calls_recorded"],
        "wall_seconds": time.perf_counter() - start,
        "variants_vs_dense": {
            name: stats[name].summary(dense_seconds if name != "sdpa" else None)
            for name in ["sdpa", *[schedule.name for schedule in schedules]]
            if stats[name].calls
        },
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
