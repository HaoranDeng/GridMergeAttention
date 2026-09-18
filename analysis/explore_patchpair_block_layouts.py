"""Screen 2D query-patch plus 2D key-patch GridMerge layouts.

The maintained hard-C4 layout uses one frame column as the query block
(`30` queries) and one frame row as the key block (`52` keys). This script
keeps the same hard-route merge and full compressed second-stage attention,
but lets both sides be spatial patches, for example:

  qp15x2_kp2x26_c4

which uses `15 x 2 = 30` query tokens and `2 x 26 = 52` key tokens per block.
Timings here are PyTorch screening timings, so use them for quality triage
before writing a tuned Triton kernel.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_h200_attention_backend_error_time import Stats, hard_c4_attention, load_pipeline
from analysis.explore_rowcol_block_layouts import biased_compressed_sdpa, gather_seed_queries, merge_kv
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from generate import load_prompts
from utils.misc import set_seed


VARIANT_RE = re.compile(
    r"qp(?P<qh>\d+)x(?P<qw>\d+)_kp(?P<kh>\d+)x(?P<kw>\d+)_c(?P<c>\d+)$"
)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def video_to_query_patches(x, frames, height, width, patch_h, patch_w):
    video = x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3))
    return rearrange(
        video,
        "b t (qhg ph) (qwg pw) head dim -> b t (ph pw) (qhg qwg) head dim",
        ph=patch_h,
        pw=patch_w,
    ).contiguous()


def query_patches_to_video(x, height, width, patch_h, patch_w):
    video = rearrange(
        x,
        "b t (ph pw) (qhg qwg) head dim -> b t (qhg ph) (qwg pw) head dim",
        ph=patch_h,
        pw=patch_w,
        qhg=height // patch_h,
        qwg=width // patch_w,
    )
    return video.reshape(x.size(0), -1, x.size(4), x.size(5)).contiguous()


def video_to_key_patches(x, frames, height, width, patch_h, patch_w):
    video = x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3))
    return rearrange(
        video,
        "b t (khg ph) (kwg pw) head dim -> b t (khg kwg) (ph pw) head dim",
        ph=patch_h,
        pw=patch_w,
    ).contiguous()


def local_patch_seed_indices(height, width, qh, qw, kh, kw, num_seeds, device):
    """Pick seed query positions in each query patch closest to each key patch.

    The original row/column layout uses the query row aligned to the key row as
    the first seed. For 2D patches there is no single row/column intersection,
    so the analogous local rule is nearest query-patch positions to the key
    patch center.
    """

    q_row_groups = height // qh
    q_col_groups = width // qw
    k_row_groups = height // kh
    k_col_groups = width // kw
    seeds = torch.empty(k_row_groups * k_col_groups, q_row_groups * q_col_groups, num_seeds, dtype=torch.long)

    for key_group in range(k_row_groups * k_col_groups):
        kr = key_group // k_col_groups
        kc = key_group % k_col_groups
        key_center_r = kr * kh + (kh - 1) / 2
        key_center_c = kc * kw + (kw - 1) / 2

        for query_group in range(q_row_groups * q_col_groups):
            qr = query_group // q_col_groups
            qc = query_group % q_col_groups
            candidates = []
            for local_r in range(qh):
                row = qr * qh + local_r
                for local_c in range(qw):
                    col = qc * qw + local_c
                    distance = abs(row - key_center_r) + abs(col - key_center_c)
                    candidates.append((distance, local_r * qw + local_c))
            candidates.sort(key=lambda item: (item[0], item[1]))
            seeds[key_group, query_group] = torch.tensor(
                [index for _, index in candidates[:num_seeds]],
                dtype=torch.long,
            )
    return seeds.to(device=device)


def patchpair_layout_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    *,
    query_patch_h,
    query_patch_w,
    key_patch_h,
    key_patch_w,
    num_seeds,
    query_outer_chunk,
    head_chunk,
):
    frames, height, width = [int(value) for value in grid_size]
    if height % query_patch_h or width % query_patch_w:
        raise ValueError("query patch must divide the latent grid.")
    if height % key_patch_h or width % key_patch_w:
        raise ValueError("key patch must divide the latent grid.")
    if num_seeds > query_patch_h * query_patch_w:
        raise ValueError("num_seeds exceeds query tokens inside one patch.")

    q_blocks = video_to_query_patches(q, frames, height, width, query_patch_h, query_patch_w)
    k_blocks = video_to_key_patches(k, frames, height, width, key_patch_h, key_patch_w)
    v_blocks = video_to_key_patches(v, frames, height, width, key_patch_h, key_patch_w)
    seed_indices = local_patch_seed_indices(
        height,
        width,
        query_patch_h,
        query_patch_w,
        key_patch_h,
        key_patch_w,
        num_seeds,
        q.device,
    )

    out = torch.empty_like(q_blocks)
    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            q_chunk = q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous()
            k_chunk = k_blocks[:, :, :, :, h_start:h_end].contiguous()
            v_chunk = v_blocks[:, :, :, :, h_start:h_end].contiguous()
            seeds = seed_indices.view(1, 1, seed_indices.size(0), seed_indices.size(1), seed_indices.size(2))
            seeds = seeds.expand(q_chunk.size(1), k_chunk.size(1), -1, -1, -1)
            seed_queries = gather_seed_queries(q_chunk, seeds)
            merged_key, merged_value, routing_bias = merge_kv(seed_queries, k_chunk, v_chunk, scale)
            out[:, q_start:q_end, :, :, h_start:h_end] = biased_compressed_sdpa(
                q_chunk,
                merged_key,
                merged_value,
                routing_bias,
                scale,
            )
    return query_patches_to_video(out, height, width, query_patch_h, query_patch_w)


def parse_variant(name):
    match = VARIANT_RE.match(name)
    if not match:
        raise ValueError(f"variant must look like qp15x2_kp2x26_c4, got {name!r}")
    return {
        "query_patch_h": int(match.group("qh")),
        "query_patch_w": int(match.group("qw")),
        "key_patch_h": int(match.group("kh")),
        "key_patch_w": int(match.group("kw")),
        "num_seeds": int(match.group("c")),
    }


def variant_metadata(grid_size, spec):
    frames, height, width = [int(value) for value in grid_size]
    key_groups = (height // spec["key_patch_h"]) * (width // spec["key_patch_w"])
    compressed_tokens = frames * key_groups * spec["num_seeds"]
    return {
        **spec,
        "query_block_tokens": spec["query_patch_h"] * spec["query_patch_w"],
        "key_block_tokens": spec["key_patch_h"] * spec["key_patch_w"],
        "key_groups_per_frame": key_groups,
        "compressed_tokens_per_query": compressed_tokens,
        "compressed_token_density": compressed_tokens / (frames * height * width),
    }


def install_probe(stats, args):
    import wan.modules.model as wan_model

    variants = [name.strip() for name in args.variants.replace(",", " ").split() if name.strip()]
    variant_specs = {name: parse_variant(name) for name in variants}
    backends = {name.strip() for name in args.reference_backends.replace(",", " ").split() if name.strip()}
    state = {"calls_total": 0, "calls_recorded": 0, "calls_current_prompt": 0, "grid_size": None, "shape": None}

    def run_dense(q, k, v, self):
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

        state["calls_total"] += 1
        state["calls_current_prompt"] += 1
        state["grid_size"] = [int(value) for value in grid_size]
        state["shape"] = {"batch": batch, "seq_len": seq_len, "heads": heads, "dim": dim}
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] <= args.max_calls

        if should_record:
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)
            if "flash_attn_triton" in backends:
                out, seconds = timed(lambda: flash_attention_triton(q, k, v, scale))
                stats["flash_attn_triton"].add_time(seconds)
                stats["flash_attn_triton"].add_error(out, dense)
            if "hard_c4_triton" in backends:
                out, seconds = timed(lambda: hard_c4_attention(q, k, v, grid_size, scale, args.hard_query_outer_chunk))
                stats["hard_c4_triton"].add_time(seconds)
                stats["hard_c4_triton"].add_error(out, dense)
            for name, spec in variant_specs.items():
                out, seconds = timed(
                    lambda spec=spec: patchpair_layout_attention(
                        q,
                        k,
                        v,
                        grid_size,
                        scale,
                        **spec,
                        query_outer_chunk=args.query_outer_chunk,
                        head_chunk=args.head_chunk,
                    )
                )
                stats[name].add_time(seconds)
                stats[name].add_error(out, dense)
            state["calls_recorded"] += 1
            out = dense
        else:
            out = run_dense(q, k, v, self)
        return self.o(out.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state, variant_specs


@torch.inference_mode()
def run_denoise_only(pipeline, config, prompt, seed, dtype):
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
    parser.add_argument("--max_calls", type=int, default=8)
    parser.add_argument("--query_outer_chunk", type=int, default=1)
    parser.add_argument("--head_chunk", type=int, default=1)
    parser.add_argument("--hard_query_outer_chunk", type=int, default=10)
    parser.add_argument(
        "--variants",
        default="qp15x2_kp2x26_c2,qp15x2_kp2x26_c3,qp15x2_kp2x26_c4,qp15x2_kp2x26_c6",
    )
    parser.add_argument("--reference_backends", default="flash_attn_triton,hard_c4_triton")
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
    state, variant_specs = install_probe(stats, args)

    sync()
    start = time.perf_counter()
    for offset, prompt in enumerate(prompts):
        state["calls_current_prompt"] = 0
        run_denoise_only(pipeline, config, prompt, args.seed + offset, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    dense_seconds = stats["dense"].seconds
    variants = {
        name: value.summary(dense_seconds=dense_seconds if name != "dense" else None, videos=len(prompts))
        for name, value in sorted(stats.items())
    }
    metadata = {}
    if state["grid_size"] is not None:
        metadata = {name: variant_metadata(state["grid_size"], spec) for name, spec in variant_specs.items()}

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
        "reference_backends": args.reference_backends,
        "grid_size": state["grid_size"],
        "shape": state["shape"],
        "calls_total": state["calls_total"],
        "calls_recorded": state["calls_recorded"],
        "wall_seconds": wall_seconds,
        "layout_metadata": metadata,
        "variants_vs_dense": variants,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
