"""Screen temporal-frame grouped row/column GridMerge layouts on real Wan Q/K/V.

The maintained layout uses one query frame and one key frame per block:

  q block: one frame, one column -> 30 query tokens
  k block: one frame, one row    -> 52 key tokens

This script groups adjacent frames before the same row/column split:

  qf3_kf1_c4: three query frames share one 90-query-token block, while key
              blocks remain one frame row of 52 keys.
  qf1_kf3_c6: query blocks stay one frame column, while each key block is the
              same spatial row across three frames, 156 keys compressed to C=6.

Timings are PyTorch screening timings for quality triage. Only variants that
improve the speed/quality frontier should get a tuned Triton kernel.
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
from analysis.explore_rowcol_block_layouts import biased_compressed_sdpa, merge_kv
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from generate import load_prompts
from utils.misc import set_seed


VARIANT_RE = re.compile(r"qf(?P<qframes>\d+)_kf(?P<kframes>\d+)_c(?P<c>\d+)$")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def video_to_query_frame_blocks(x, frames, height, width, query_frame_group):
    video = x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3))
    return rearrange(
        video,
        "b (a qf) h w head dim -> b a (qf h) w head dim",
        qf=query_frame_group,
    ).contiguous()


def query_frame_blocks_to_video(x, height, width, query_frame_group):
    video = rearrange(
        x,
        "b a (qf h) w head dim -> b (a qf) h w head dim",
        qf=query_frame_group,
        h=height,
    )
    return video.reshape(video.size(0), -1, video.size(4), video.size(5)).contiguous()


def video_to_key_frame_blocks(x, frames, height, width, key_frame_group):
    video = x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3))
    return rearrange(
        video,
        "b (f kf) h w head dim -> b f h (kf w) head dim",
        kf=key_frame_group,
    ).contiguous()


def spatial_seed_rows(key_row, height):
    rows = [key_row]
    for delta in range(1, height):
        rows.extend((key_row - delta, key_row + delta))
    return [row for row in rows if 0 <= row < height]


def temporal_seed_indices(height, key_rows, query_frame_group, num_seeds, device):
    if num_seeds > height * query_frame_group:
        raise ValueError("num_seeds exceeds query tokens inside one temporal block.")

    rows = []
    for key_row in range(key_rows):
        selected = []
        for spatial_row in spatial_seed_rows(key_row, height):
            for local_frame in range(query_frame_group):
                selected.append(local_frame * height + spatial_row)
                if len(selected) == num_seeds:
                    break
            if len(selected) == num_seeds:
                break
        rows.append(selected)
    return torch.tensor(rows, device=device, dtype=torch.long)


def gather_seed_queries(query, seed_indices, key_outer_count):
    query_outer = torch.arange(query.size(1), device=query.device).view(-1, 1, 1, 1, 1)
    query_col = torch.arange(query.size(3), device=query.device).view(1, 1, 1, -1, 1)
    seed = seed_indices.view(1, 1, seed_indices.size(0), 1, seed_indices.size(1))
    seed = seed.expand(query.size(1), key_outer_count, -1, query.size(3), -1)
    return query[:, query_outer, seed, query_col]


def temporal_grouped_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    *,
    query_frame_group,
    key_frame_group,
    num_seeds,
    query_outer_chunk,
    head_chunk,
):
    frames, height, width = [int(value) for value in grid_size]
    if frames % query_frame_group or frames % key_frame_group:
        raise ValueError("query/key frame groups must divide frame count.")

    q_blocks = video_to_query_frame_blocks(q, frames, height, width, query_frame_group)
    k_blocks = video_to_key_frame_blocks(k, frames, height, width, key_frame_group)
    v_blocks = video_to_key_frame_blocks(v, frames, height, width, key_frame_group)
    seed_indices = temporal_seed_indices(height, k_blocks.size(2), query_frame_group, num_seeds, q.device)

    out = torch.empty_like(q_blocks)
    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            q_chunk = q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous()
            k_chunk = k_blocks[:, :, :, :, h_start:h_end].contiguous()
            v_chunk = v_blocks[:, :, :, :, h_start:h_end].contiguous()
            seed_queries = gather_seed_queries(q_chunk, seed_indices, k_chunk.size(1))
            merged_key, merged_value, routing_bias = merge_kv(seed_queries, k_chunk, v_chunk, scale)
            out[:, q_start:q_end, :, :, h_start:h_end] = biased_compressed_sdpa(
                q_chunk,
                merged_key,
                merged_value,
                routing_bias,
                scale,
            )
    return query_frame_blocks_to_video(out, height, width, query_frame_group)


def parse_variant(name):
    match = VARIANT_RE.match(name)
    if not match:
        raise ValueError(f"variant must look like qf3_kf1_c4, got {name!r}")
    return {
        "query_frame_group": int(match.group("qframes")),
        "key_frame_group": int(match.group("kframes")),
        "num_seeds": int(match.group("c")),
    }


def variant_metadata(grid_size, spec):
    frames, height, width = [int(value) for value in grid_size]
    qf = spec["query_frame_group"]
    kf = spec["key_frame_group"]
    c = spec["num_seeds"]
    key_groups = (frames // kf) * height
    query_groups = (frames // qf) * width
    return {
        "query_block_tokens": qf * height,
        "key_block_tokens": kf * width,
        "query_groups": query_groups,
        "key_groups": key_groups,
        "compressed_tokens_per_query": key_groups * c,
        "compressed_token_density": c / (kf * width),
    }


def install_probe(stats, args):
    import wan.modules.model as wan_model

    variants = [name.strip() for name in args.variants.replace(",", " ").split() if name.strip()]
    variant_specs = {name: parse_variant(name) for name in variants}
    backends = {name.strip() for name in args.reference_backends.replace(",", " ").split() if name.strip()}
    state = {
        "calls_total": 0,
        "calls_recorded": 0,
        "calls_current_prompt": 0,
        "grid_size": None,
        "shape": None,
        "variant_specs": variant_specs,
    }

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

        state["calls_total"] += 1
        state["calls_current_prompt"] += 1
        state["grid_size"] = [int(value) for value in grid_size]
        state["shape"] = {"batch": batch, "seq_len": seq_len, "heads": heads, "dim": dim}
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] <= args.max_calls

        if should_record:
            dense, seconds = timed(lambda: dense_attention(q, k, v, self))
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
                    lambda spec=spec: temporal_grouped_attention(
                        q,
                        k,
                        v,
                        grid_size,
                        scale,
                        query_frame_group=spec["query_frame_group"],
                        key_frame_group=spec["key_frame_group"],
                        num_seeds=spec["num_seeds"],
                        query_outer_chunk=args.query_outer_chunk,
                        head_chunk=args.head_chunk,
                    )
                )
                stats[name].add_time(seconds)
                stats[name].add_error(out, dense)
            state["calls_recorded"] += 1
            out = dense
        else:
            out = dense_attention(q, k, v, self)
        return self.o(out.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state


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
        default="qf3_kf1_c3,qf3_kf1_c4,qf3_kf1_c6,qf1_kf3_c3,qf1_kf3_c6,qf1_kf3_c9,qf3_kf3_c6,qf3_kf3_c12",
    )
    parser.add_argument("--reference_backends", default="flash_attn_triton,hard_c4_triton")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

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
        for name, spec in state["variant_specs"].items():
            metadata[name] = {**spec, **variant_metadata(state["grid_size"], spec)}

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
