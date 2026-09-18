import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from analysis.compare_transpose_monarchrt import run_one_token_variant
from attention.hard_c4_wan21_1_3b_480p_triton import HARD_C4_BACKEND
from attention.token_merging import token_merging_attn
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
        diff = out.float().reshape(-1) - ref.float().reshape(-1)
        ref_f = ref.float().reshape(-1)
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
                    "mse_vs_dense": self.sqerr / self.numel,
                    "rel_l2_vs_dense": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                    "max_abs_vs_dense": self.max_abs,
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


def current_hard_c4(q, k, v, grid_size, scale, query_outer_chunk):
    _frames, height, width = [int(x) for x in grid_size]
    return token_merging_attn(
        q,
        k,
        v,
        f_tied=1,
        h_reduce=1,
        w_reduce=1,
        h=height,
        w=width,
        sm_scale=scale,
        query_outer_chunk=query_outer_chunk,
        layout="hard_route",
        num_merge_tokens=4,
        seed_mode="local_with_ith",
        seed=0,
        hard_route_backend=HARD_C4_BACKEND,
    )


def monarchrt_w_reduce4(q, k, v, grid_size, scale, head_chunk):
    return run_one_token_variant(
        q,
        k,
        v,
        grid_size,
        {"h_reduce": 1, "w_reduce": 4, "transpose": False},
        scale,
        head_chunk,
    )


def variant_specs(frames, height, width):
    dense_tokens = frames * height * width
    compressed = frames * height * 4
    density = compressed / dense_tokens
    base = {
        "compressed_tokens_per_query": compressed,
        "density": density,
        "sparsity": 1.0 - density,
    }
    return {
        "current_hard_c4": {
            **base,
            "description": "52 key columns -> 4 compressed K/V tokens per grid block.",
        },
        "monarchrt_w_reduce4": {
            **base,
            "h_reduce": 1,
            "w_reduce": 4,
            "q_init": "ith",
            "description": "Original one-token MonarchRT: 13 key columns -> 1 compressed K/V token.",
        },
    }


def install_probe(stats, args):
    import wan.modules.model as wan_model

    state = {"calls_total": 0, "calls_current_prompt": 0, "warmed": False, "grid_size": None}

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

        if should_record and not state["warmed"]:
            _ = dense_attention(q, k, v, self)
            _ = current_hard_c4(q, k, v, grid_size, scale, args.query_outer_chunk)
            _ = monarchrt_w_reduce4(q, k, v, grid_size, scale, args.head_chunk)
            sync()
            state["warmed"] = True

        if should_record:
            dense, seconds = timed(lambda: dense_attention(q, k, v, self))
            stats["dense"].add_time(seconds)

            hard, seconds = timed(lambda: current_hard_c4(q, k, v, grid_size, scale, args.query_outer_chunk))
            stats["current_hard_c4"].add_time(seconds)
            stats["current_hard_c4"].add_error(hard, dense)

            monarch, seconds = timed(lambda: monarchrt_w_reduce4(q, k, v, grid_size, scale, args.head_chunk))
            stats["monarchrt_w_reduce4"].add_time(seconds)
            stats["monarchrt_w_reduce4"].add_error(monarch, dense)

            x = dense
        else:
            x = dense_attention(q, k, v, self)
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
    parser = argparse.ArgumentParser(description="Compare current hard C4 against equal-density MonarchRT w_reduce=4.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=0)
    parser.add_argument("--query_outer_chunk", type=int, default=10)
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
    state = install_probe(stats, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        state["warmed"] = False
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    frames, height, width = state["grid_size"]
    specs = variant_specs(frames, height, width)
    dense_seconds = stats["dense"].seconds
    variants = {}
    for name, value in sorted(stats.items()):
        variants[name] = value.summary(dense_seconds=dense_seconds if name != "dense" else None)
        if name in specs:
            variants[name].update(specs[name])

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
        "grid_size": [frames, height, width],
        "dense_tokens_per_query": frames * height * width,
        "wall_seconds": wall_seconds,
        "variants": variants,
        "sparsity_check": {
            "current_hard_c4_density": specs["current_hard_c4"]["density"],
            "monarchrt_w_reduce4_density": specs["monarchrt_w_reduce4"]["density"],
            "equal_density": specs["current_hard_c4"]["density"] == specs["monarchrt_w_reduce4"]["density"],
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
