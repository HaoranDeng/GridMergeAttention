import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from attention.common import blocks_to_video, video_to_blocks
from attention.hard_route import hard_route_grid_compressed_attention
from attention.hard_route_shared import _merge_shared, cached_shared_seed_rows
from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


class Stats:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0
        self.rel_l2_sum = 0.0
        self.cosine_error_sum = 0.0
        self.max_abs = 0.0
        self.error_calls = 0

    def add_time(self, name, seconds):
        self.seconds[name] += float(seconds)
        self.counts[name] += 1

    def add_error(self, out, ref):
        diff = (out.float() - ref.float()).reshape(-1)
        ref_flat = ref.float().reshape(-1)
        out_flat = out.float().reshape(-1)
        sqerr = torch.dot(diff, diff)
        ref_sq = torch.dot(ref_flat, ref_flat).clamp_min(1e-20)
        out_sq = torch.dot(out_flat, out_flat).clamp_min(1e-20)
        dot = torch.dot(out_flat, ref_flat)

        self.sqerr += float(sqerr.item())
        self.ref_sq += float(ref_sq.item())
        self.numel += diff.numel()
        self.rel_l2_sum += float(torch.sqrt(sqerr / ref_sq).item())
        self.cosine_error_sum += float((1.0 - dot / torch.sqrt(out_sq * ref_sq)).item())
        self.max_abs = max(self.max_abs, float(diff.abs().max().item()))
        self.error_calls += 1

    def summary(self):
        out = {}
        for name, seconds in sorted(self.seconds.items()):
            calls = self.counts[name]
            out[name] = {
                "seconds": seconds,
                "calls": calls,
                "mean_ms": 1000.0 * seconds / calls if calls else 0.0,
            }
        if self.numel:
            out["error_vs_dense"] = {
                "calls": self.error_calls,
                "mse": self.sqerr / self.numel,
                "rel_l2": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                "mean_call_rel_l2": self.rel_l2_sum / self.error_calls,
                "cosine_error": self.cosine_error_sum / self.error_calls,
                "max_abs": self.max_abs,
            }
        return out


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


def install_probe(stats, args):
    import wan.modules.model as wan_model

    warmed = {"value": False}

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def run_split(q, k, v, grid_size, scale, record):
        _, height, width = grid_size
        if record:
            q_blocks, seconds = timed(lambda: video_to_blocks(q, 1, 1, 1, height, width))
            stats.add_time("video_to_q_blocks", seconds)
            k_blocks, seconds = timed(lambda: video_to_blocks(k, 1, 1, 1, height, width))
            stats.add_time("video_to_k_blocks", seconds)
            v_blocks, seconds = timed(lambda: video_to_blocks(v, 1, 1, 1, height, width))
            stats.add_time("video_to_v_blocks", seconds)
            seed_rows, seconds = timed(
                lambda: cached_shared_seed_rows(
                    q.device,
                    k_blocks.size(1),
                    k_blocks.size(2),
                    q_blocks.size(3),
                    q_blocks.size(2),
                    args.num_merge_tokens,
                    args.seed,
                    include_ith=True,
                    fast_with_include=True,
                )
            )
            stats.add_time("seed_rows", seconds)
            merge_out, seconds = timed(
                lambda: _merge_shared(
                    q_blocks,
                    seed_rows,
                    k_blocks,
                    v_blocks,
                    args.hard_route_backend,
                    scale,
                )
            )
            stats.add_time("merge", seconds)
            merged_keys, merged_values, routing_bias, cluster_has_keys = merge_out
            del cluster_has_keys
            routed, seconds = timed(
                lambda: hard_route_grid_compressed_attention(
                    q_blocks,
                    merged_keys,
                    merged_values,
                    routing_bias,
                    args.hard_route_routing_backend,
                    scale,
                )
            )
            stats.add_time("routing", seconds)
            out, seconds = timed(lambda: blocks_to_video(routed, 1, 1, 1))
            stats.add_time("blocks_to_video", seconds)
            return out

        q_blocks = video_to_blocks(q, 1, 1, 1, height, width)
        k_blocks = video_to_blocks(k, 1, 1, 1, height, width)
        v_blocks = video_to_blocks(v, 1, 1, 1, height, width)
        seed_rows = cached_shared_seed_rows(
            q.device,
            k_blocks.size(1),
            k_blocks.size(2),
            q_blocks.size(3),
            q_blocks.size(2),
            args.num_merge_tokens,
            args.seed,
            include_ith=True,
            fast_with_include=True,
        )
        merged_keys, merged_values, routing_bias, cluster_has_keys = _merge_shared(
            q_blocks,
            seed_rows,
            k_blocks,
            v_blocks,
            args.hard_route_backend,
            scale,
        )
        del cluster_has_keys
        routed = hard_route_grid_compressed_attention(
            q_blocks,
            merged_keys,
            merged_values,
            routing_bias,
            args.hard_route_routing_backend,
            scale,
        )
        return blocks_to_video(routed, 1, 1, 1)

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

        if not warmed["value"]:
            _ = run_dense(q, k, v, self)
            _ = run_split(q, k, v, grid_size, dim**-0.5, record=False)
            sync()
            warmed["value"] = True

        if stats.counts["dense"] < args.max_calls:
            ref, seconds = timed(lambda: run_dense(q, k, v, self))
            stats.add_time("dense", seconds)
            out = run_split(q, k, v, grid_size, dim**-0.5, record=True)
            stats.add_error(out, ref)
            x = ref
        else:
            x = run_dense(q, k, v, self)
        return self.o(x.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention


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
    parser = argparse.ArgumentParser(description="Profile hard-route merge and routing components.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=90)
    parser.add_argument("--num_merge_tokens", type=int, default=4)
    parser.add_argument("--hard_route_backend", required=True)
    parser.add_argument("--hard_route_routing_backend", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    prompt = load_prompts(args.prompt_path, args.prompt_index + 1)[args.prompt_index]
    stats = Stats()
    install_probe(stats, args)

    sync()
    start = time.perf_counter()
    run_denoise_only(pipeline, config, prompt, args.seed, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dtype": args.dtype,
        "max_calls": args.max_calls,
        "hard_route_backend": args.hard_route_backend,
        "hard_route_routing_backend": args.hard_route_routing_backend,
        "wall_seconds": wall_seconds,
        "summary": stats.summary(),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
