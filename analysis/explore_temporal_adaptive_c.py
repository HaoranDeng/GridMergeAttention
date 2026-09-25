"""Screen temporal-distance adaptive C on the original row/column grid.

The maintained layout compresses every `(query frame, key frame, key row,
query column)` block with the same number of merged tokens C.

This script keeps the original geometry:

  query block: one frame column -> 30 query tokens
  key block:   one frame row    -> 52 key tokens

but chooses C from the temporal distance between the query frame and key frame.
For example, ``ta_r0c4_far1`` uses C=4 for same-frame key blocks and C=1 for
all other key frames. The second stage still attends over all compressed key
blocks; far frames are compressed more aggressively rather than removed.

Timings are PyTorch screening timings and should not be used as kernel speed.
The reported density is the average compressed-token density implied by the C
schedule.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_h200_attention_backend_error_time import Stats, hard_c4_attention, load_pipeline
from analysis.explore_rowcol_block_layouts import (
    biased_compressed_sdpa,
    gather_seed_queries,
    local_seed_indices,
    merge_kv,
    q_to_grouped_columns,
    grouped_columns_to_video,
    kv_to_grouped_rows,
)
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from generate import load_prompts
from utils.misc import set_seed


VARIANT_RE = re.compile(r"ta_(?P<terms>(?:r\d+c\d+_)*r\d+c\d+)_far(?P<far>\d+)$")
TERM_RE = re.compile(r"r(?P<radius>\d+)c(?P<c>\d+)")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def parse_variant(name):
    match = VARIANT_RE.fullmatch(name)
    if not match:
        raise ValueError(
            "variant must look like ta_r0c4_far1 or ta_r0c4_r1c2_far1, "
            f"got {name!r}"
        )
    terms = []
    for token in match.group("terms").split("_"):
        term = TERM_RE.fullmatch(token)
        if not term:
            raise ValueError(f"bad temporal C term {token!r}")
        terms.append({"radius": int(term.group("radius")), "c": int(term.group("c"))})
    return {"terms": terms, "far_c": int(match.group("far"))}


def temporal_c_matrix(query_outer, key_outer, spec, device):
    q = torch.arange(query_outer, device=device).view(-1, 1)
    k = torch.arange(key_outer, device=device).view(1, -1)
    distance = (q - k).abs()
    c_pair = torch.full((query_outer, key_outer), int(spec["far_c"]), device=device, dtype=torch.long)
    for term in sorted(spec["terms"], key=lambda item: item["radius"], reverse=True):
        c_pair = torch.where(distance <= int(term["radius"]), int(term["c"]), c_pair)
    return c_pair


def pack_temporal_c_outputs(q_chunk, k_chunk, v_chunk, scale, c_pair, max_c):
    batch, query_outer, rows, cols, heads, dim = q_chunk.shape
    key_outer = k_chunk.size(1)
    key_rows = k_chunk.size(2)

    merged_key = torch.zeros(
        (batch, query_outer, key_outer, cols, key_rows, max_c, heads, dim),
        device=q_chunk.device,
        dtype=k_chunk.dtype,
    )
    merged_value = torch.zeros_like(merged_key)
    routing_bias = torch.full(
        (batch, heads, query_outer, key_outer, key_rows, cols, max_c),
        float("inf"),
        device=q_chunk.device,
        dtype=torch.float32,
    )

    for c in sorted(set(int(value) for value in c_pair.flatten().tolist())):
        seed_indices = local_seed_indices(
            rows,
            query_column_group=1,
            key_row_group=1,
            key_groups=key_rows,
            num_seeds=c,
            device=q_chunk.device,
        )
        seeds = seed_indices.view(1, 1, key_rows, 1, c)
        seeds = seeds.expand(query_outer, key_outer, -1, cols, -1)
        seed_queries = gather_seed_queries(q_chunk, seeds)
        mk, mv, bias = merge_kv(seed_queries, k_chunk, v_chunk, scale)

        kv_mask = (c_pair[:query_outer, :key_outer] == c).view(1, query_outer, key_outer, 1, 1, 1, 1, 1)
        bias_mask = (c_pair[:query_outer, :key_outer] == c).view(1, 1, query_outer, key_outer, 1, 1, 1)
        merged_key[:, :, :, :, :, :c] = torch.where(kv_mask, mk, merged_key[:, :, :, :, :, :c])
        merged_value[:, :, :, :, :, :c] = torch.where(kv_mask, mv, merged_value[:, :, :, :, :, :c])
        routing_bias[:, :, :, :, :, :, :c] = torch.where(
            bias_mask,
            bias,
            routing_bias[:, :, :, :, :, :, :c],
        )

    return merged_key, merged_value, routing_bias


def temporal_adaptive_c_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    *,
    spec,
    query_outer_chunk,
    head_chunk,
):
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = q_to_grouped_columns(q, frames, height, width, query_column_group=1)
    k_blocks = kv_to_grouped_rows(k, frames, height, width, key_row_group=1)
    v_blocks = kv_to_grouped_rows(v, frames, height, width, key_row_group=1)
    c_pair_full = temporal_c_matrix(q_blocks.size(1), k_blocks.size(1), spec, q.device)
    max_c = int(c_pair_full.max().item())

    out = torch.empty_like(q_blocks)
    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        c_pair = c_pair_full[q_start:q_end]
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            q_chunk = q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous()
            k_chunk = k_blocks[:, :, :, :, h_start:h_end].contiguous()
            v_chunk = v_blocks[:, :, :, :, h_start:h_end].contiguous()
            merged_key, merged_value, routing_bias = pack_temporal_c_outputs(
                q_chunk,
                k_chunk,
                v_chunk,
                scale,
                c_pair,
                max_c,
            )
            out[:, q_start:q_end, :, :, h_start:h_end] = biased_compressed_sdpa(
                q_chunk,
                merged_key,
                merged_value,
                routing_bias,
                scale,
            )
    return grouped_columns_to_video(out, height, width, query_column_group=1)


def variant_metadata(grid_size, spec):
    frames, height, width = [int(value) for value in grid_size]
    c_pair = temporal_c_matrix(frames, frames, spec, device=torch.device("cpu"))
    compressed_per_query_frame = c_pair.sum(dim=1).float() * height
    density_by_query_frame = compressed_per_query_frame / (frames * height * width)
    return {
        "terms": spec["terms"],
        "far_c": spec["far_c"],
        "min_c": int(c_pair.min().item()),
        "max_c": int(c_pair.max().item()),
        "mean_c": float(c_pair.float().mean().item()),
        "query_block_tokens": height,
        "key_block_tokens": width,
        "compressed_tokens_per_query_mean": float(compressed_per_query_frame.mean().item()),
        "compressed_token_density": float(density_by_query_frame.mean().item()),
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
                    lambda spec=spec: temporal_adaptive_c_attention(
                        q,
                        k,
                        v,
                        grid_size,
                        scale,
                        spec=spec,
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
        default="ta_r0c4_far1,ta_r1c4_far1,ta_r0c4_r1c2_far1,ta_r1c4_far2,ta_r2c4_far1",
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
