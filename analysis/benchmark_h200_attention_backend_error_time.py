"""Benchmark the four maintained Wan attention paths on real model QKV."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.compare_hard_c4_backends_dense import load_pipeline
from attention._grid_utils import blocks_to_video as grid_blocks_to_video
from attention._grid_utils import video_to_blocks as grid_video_to_blocks
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from attention.hard_c4_wan21_1_3b_480p_triton import hard_c4_rt_attention_triton_blocks
from attention.monarch_rt_wan21_1_3b_480p_triton import monarch_rt_attention_triton_blocks
from generate import load_prompts
from utils.misc import set_seed


SUPPORTED_BACKENDS = {
    "sdpa",
    "flash_attn_triton",
    "monarch_rt",
    "hard_c4",
}


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

    def summary(self, *, dense_seconds=None, videos=1):
        payload = {
            "calls": self.calls,
            "seconds": self.seconds,
            "mean_ms_per_call": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
            "seconds_per_video": self.seconds / videos if videos else 0.0,
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


def hard_c4_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    query_outer_chunk,
    head_chunk=4,
):
    """Run the current best hard-C4 kernel with production chunking."""
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = video_to_blocks(q, frames, height, width)
    k_blocks = video_to_blocks(k, frames, height, width)
    v_blocks = video_to_blocks(v, frames, height, width)
    out = torch.empty_like(q_blocks)
    heads = q_blocks.size(4)
    head_chunk = heads if not head_chunk or head_chunk >= heads else int(head_chunk)
    for start in range(0, frames, query_outer_chunk):
        end = min(start + query_outer_chunk, frames)
        out_slice = out[:, start:end]
        for head_start in range(0, heads, head_chunk):
            head_count = min(head_chunk, heads - head_start)
            hard_c4_rt_attention_triton_blocks(
                q_blocks[:, start:end],
                k_blocks,
                v_blocks,
                scale,
                output=out_slice,
                head_start=head_start,
                head_count=head_count,
            )
    return blocks_to_video(out)


def monarch_rt_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    query_outer_chunk,
    h_reduce,
    w_reduce,
):
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = grid_video_to_blocks(q, 1, h_reduce, w_reduce, height, width)
    k_blocks = grid_video_to_blocks(k, 1, h_reduce, w_reduce, height, width)
    v_blocks = grid_video_to_blocks(v, 1, h_reduce, w_reduce, height, width)
    out = torch.empty_like(q_blocks)
    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        out[:, start:end] = monarch_rt_attention_triton_blocks(
            q_blocks[:, start:end], k_blocks, v_blocks, scale
        )
    return grid_blocks_to_video(out, 1, h_reduce, w_reduce)


def install_probe(stats, args):
    import wan.modules.model as wan_model

    requested = [name.strip() for name in args.backends.split(",") if name.strip()]
    requested = ["monarch_rt" if name == "monarch_rt_triton" else name for name in requested]
    unknown = set(requested) - SUPPORTED_BACKENDS
    if unknown:
        raise ValueError(f"unsupported attention backends: {sorted(unknown)}")
    selected = list(dict.fromkeys(name for name in requested if name != "sdpa"))
    state = {
        "backends": ["sdpa", *selected],
        "calls_total": 0,
        "calls_recorded": 0,
        "calls_current_prompt": 0,
        "grid_size": None,
        "shape": None,
        "did_warmup": False,
    }

    def dense_attention(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len = x.shape[:2]
        heads, dim = self.num_heads, self.head_dim
        q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(
            batch, seq_len, heads, dim
        )
        k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(
            batch, seq_len, heads, dim
        )
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        scale = dim**-0.5

        runners = {
            "flash_attn_triton": lambda: flash_attention_triton(q, k, v, scale),
            "monarch_rt": lambda: monarch_rt_attention(
                q,
                k,
                v,
                grid_size,
                scale,
                args.monarch_query_outer_chunk,
                args.monarch_h_reduce,
                args.monarch_w_reduce,
            ),
            "hard_c4": lambda: hard_c4_attention(
                q,
                k,
                v,
                grid_size,
                scale,
                args.query_outer_chunk,
                args.hard_head_chunk,
            ),
        }

        state["calls_total"] += 1
        state["calls_current_prompt"] += 1
        state["grid_size"] = [int(value) for value in grid_size]
        state["shape"] = {
            "batch": batch,
            "seq_len": seq_len,
            "heads": heads,
            "dim": dim,
        }
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] <= args.max_calls
        if not should_record:
            return self.o(dense_attention(q, k, v, self).flatten(2))

        if not state["did_warmup"]:
            dense_attention(q, k, v, self)
            for name in selected:
                runners[name]()
            sync()
            state["did_warmup"] = True

        dense, seconds = timed(lambda: dense_attention(q, k, v, self))
        stats["sdpa"].add_time(seconds)
        for name in selected:
            output, seconds = timed(runners[name])
            stats[name].add_time(seconds)
            stats[name].add_error(output, dense)

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
    parser = argparse.ArgumentParser(
        description="Compare maintained H200 attention backends on real Wan QKV."
    )
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=0)
    parser.add_argument("--query_outer_chunk", type=int, default=21)
    parser.add_argument("--hard_head_chunk", type=int, default=4)
    parser.add_argument("--monarch_query_outer_chunk", type=int, default=21)
    parser.add_argument("--monarch_h_reduce", type=int, default=1)
    parser.add_argument("--monarch_w_reduce", type=int, default=1)
    parser.add_argument(
        "--backends",
        default="sdpa,flash_attn_triton,monarch_rt,hard_c4",
        help="Comma-separated subset of sdpa, flash_attn_triton, monarch_rt, hard_c4.",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(
        args.config, args.checkpoint, dtype, device, args.use_ema
    )
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]

    stats = defaultdict(Stats)
    state = install_probe(stats, args)
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
        "backends": state["backends"],
        "query_outer_chunk": args.query_outer_chunk,
        "hard_head_chunk": args.hard_head_chunk,
        "monarch_query_outer_chunk": args.monarch_query_outer_chunk,
        "monarch_h_reduce": args.monarch_h_reduce,
        "monarch_w_reduce": args.monarch_w_reduce,
        "grid_size": state["grid_size"],
        "shape": state["shape"],
        "calls_total": state["calls_total"],
        "calls_recorded": state["calls_recorded"],
        "wall_seconds": time.perf_counter() - start,
        "variants_vs_dense": {
            name: value.summary(
                dense_seconds=dense_seconds if name != "sdpa" else None,
                videos=len(prompts),
            )
            for name, value in sorted(stats.items())
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
