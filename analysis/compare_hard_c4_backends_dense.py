import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from attention.hard_c4_wan21_1_3b_480p_triton import hard_c4_rt_attention_triton_blocks
from attention.hard_route_pytorch import hard_route_attention_pytorch
from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


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


def hard_c4_attention(q, k, v, grid_size, scale, backend, query_outer_chunk):
    frames, height, width = [int(x) for x in grid_size]
    q_blocks = video_to_blocks(q, frames, height, width)
    k_blocks = video_to_blocks(k, frames, height, width)
    v_blocks = video_to_blocks(v, frames, height, width)
    out = torch.empty_like(q_blocks)

    for start in range(0, q_blocks.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q_blocks.size(1))
        query_chunk = q_blocks[:, start:end].contiguous()
        out_slice = out[:, start:end]
        if backend == "pytorch":
            chunk = hard_route_attention_pytorch(
                query_chunk,
                k_blocks,
                v_blocks,
                scale,
                query_outer_offset=start,
            )
        else:
            chunk = hard_c4_rt_attention_triton_blocks(
                query_chunk,
                k_blocks,
                v_blocks,
                scale,
                output=out_slice,
            )
        if chunk is not out_slice:
            out_slice.copy_(chunk)
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

    state = {
        "calls_total": 0,
        "calls_recorded": 0,
        "calls_current_prompt": 0,
        "grid_size": None,
        "did_warmup": False,
    }

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def probed_self_attention(self, x, grid_size, rope_cache):
        batch, seq_len, heads, dim = *x.shape[:2], self.num_heads, self.head_dim
        q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(batch, seq_len, heads, dim)
        k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(batch, seq_len, heads, dim)
        v = self.v(x).view(batch, seq_len, heads, dim)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)
        scale = dim**-0.5

        state["calls_current_prompt"] += 1
        state["calls_total"] += 1
        state["grid_size"] = [int(value) for value in grid_size]
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] <= args.max_calls

        if should_record:
            if not state["did_warmup"]:
                run_dense(q, k, v, self)
                hard_c4_attention(q, k, v, grid_size, scale, "pytorch", args.query_outer_chunk)
                hard_c4_attention(q, k, v, grid_size, scale, "triton", args.query_outer_chunk)
                sync()
                state["did_warmup"] = True

            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)

            pytorch, seconds = timed(
                lambda: hard_c4_attention(q, k, v, grid_size, scale, "pytorch", args.query_outer_chunk)
            )
            stats["hard_c4_pytorch"].add_time(seconds)
            stats["hard_c4_pytorch"].add_error(pytorch, dense)

            triton, seconds = timed(
                lambda: hard_c4_attention(q, k, v, grid_size, scale, "triton", args.query_outer_chunk)
            )
            stats["hard_c4_triton"].add_time(seconds)
            stats["hard_c4_triton"].add_error(triton, dense)
            pair_stats["triton_vs_pytorch"].add_time(0.0)
            pair_stats["triton_vs_pytorch"].add_error(triton, pytorch)
            state["calls_recorded"] += 1

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
    timesteps = list(config.denoising_step_list)
    for index, timestep in enumerate(timesteps):
        _, pred = pipeline.generator(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=device) * timestep,
        )
        if index + 1 < len(timesteps):
            next_timestep = timesteps[index + 1] * torch.ones(
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
    parser = argparse.ArgumentParser(description="Compare dense, PyTorch hard C4, and Triton hard C4 on real Wan QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=4)
    parser.add_argument("--query_outer_chunk", type=int, default=1)
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
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "query_outer_chunk": args.query_outer_chunk,
        "grid_size": state["grid_size"],
        "calls_total": state["calls_total"],
        "calls_recorded": state["calls_recorded"],
        "denoise_timesteps_per_prompt": len(config.denoising_step_list),
        "wall_seconds": wall_seconds,
        "variants_vs_dense": {
            name: value.summary(dense_seconds=dense_seconds if name != "dense" else None)
            for name, value in sorted(stats.items())
        },
        "pairwise": {name: value.summary() for name, value in sorted(pair_stats.items())},
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
