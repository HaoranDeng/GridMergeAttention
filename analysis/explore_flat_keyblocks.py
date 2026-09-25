"""Screen row-major contiguous key-block GridMerge layouts on real Wan Q/K/V.

The maintained hard-C4 layout compresses one key row at a time:

  query block: one frame, one column -> 30 query tokens
  key block:   one frame, one row    -> 52 key/value tokens

This script keeps the same query block but partitions each frame's K/V tokens
in row-major order into fixed-size chunks such as 64 or 128 tokens. That tests
whether semantically local contiguous KV chunks are a better quality signal
than strict image rows before writing a specialized kernel.
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
import torch.nn.functional as F
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.benchmark_h200_attention_backend_error_time import Stats, hard_c4_attention, load_pipeline
from analysis.explore_rowcol_block_layouts import FLOAT_NEG_INF
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from generate import load_prompts
from utils.misc import set_seed


VARIANT_RE = re.compile(r"flat(?P<block>\d+)_c(?P<c>\d+)(?:_qg(?P<qg>\d+))?(?:_p(?P<pattern>[a-z0-9]+))?$")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def video_to_query_blocks(x, frames, height, width):
    return x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3)).contiguous()


def query_blocks_to_video(x):
    return x.reshape(x.size(0), -1, x.size(4), x.size(5)).contiguous()


def video_to_flat_key_blocks(x, frames, height, width, block_tokens):
    video = x.reshape(x.size(0), frames, height * width, x.size(2), x.size(3))
    groups = (height * width + block_tokens - 1) // block_tokens
    padded = video.new_zeros(x.size(0), frames, groups * block_tokens, x.size(2), x.size(3))
    padded[:, :, : height * width] = video
    return padded.reshape(x.size(0), frames, groups, block_tokens, x.size(2), x.size(3)).contiguous()


def flat_key_mask(height, width, block_tokens, device):
    groups = (height * width + block_tokens - 1) // block_tokens
    offsets = torch.arange(groups * block_tokens, device=device).reshape(groups, block_tokens)
    return offsets < (height * width)


def seed_rows_for_block(start, end, height, width, num_seeds, pattern):
    rows = []
    used = set()
    for token in range(start, end):
        row = token // width
        if row not in used:
            rows.append(row)
            used.add(row)
    if pattern == "center" and rows:
        center = (rows[0] + rows[-1]) // 2
        ordered = [center]
        for delta in range(1, height):
            ordered.extend((center - delta, center + delta))
    elif pattern == "right" and rows:
        ordered = list(range(rows[-1], height)) + list(range(rows[-1] - 1, -1, -1))
    elif pattern == "left" and rows:
        ordered = list(range(rows[0], -1, -1)) + list(range(rows[0] + 1, height))
    else:
        ordered = []
        for row in rows:
            ordered.append(row)
        for delta in range(1, height):
            ordered.extend((rows[0] - delta, rows[-1] + delta))
    for row in ordered:
        if 0 <= row < height and row not in used:
            rows.append(row)
            used.add(row)
        if len(rows) >= num_seeds:
            break
    return rows[:num_seeds]


def local_seed_indices(height, width, block_tokens, num_seeds, device, pattern):
    groups = (height * width + block_tokens - 1) // block_tokens
    rows = []
    for group in range(groups):
        start = group * block_tokens
        end = min(start + block_tokens, height * width)
        rows.append(seed_rows_for_block(start, end, height, width, num_seeds, pattern))
    return torch.tensor(rows, device=device, dtype=torch.long)


def gather_seed_queries(query, seed_indices, query_col_indices=None):
    query_outer = torch.arange(query.size(1), device=query.device).view(-1, 1, 1, 1, 1)
    if query_col_indices is None:
        query_col = torch.arange(query.size(3), device=query.device).view(1, 1, 1, -1, 1)
    else:
        query_col = query_col_indices
    return query[:, query_outer, seed_indices, query_col]


def merge_kv_masked(seed_queries, key, value, key_mask, scale):
    seed_key_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    seed_key_logits = seed_key_logits * float(scale)
    valid_key = key_mask.view(1, 1, 1, 1, key_mask.size(0), 1, 1, key_mask.size(1))
    seed_key_logits = seed_key_logits.masked_fill(~valid_key, FLOAT_NEG_INF)

    assignment_scores = torch.log_softmax(seed_key_logits, dim=-1)
    best_seed = assignment_scores.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    assignment = assignment & valid_key

    logits = seed_key_logits.masked_fill(~assignment, FLOAT_NEG_INF)
    has_key = assignment.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_key, logits, torch.zeros_like(logits))
    log_weights = torch.log_softmax(safe_logits, dim=-1).masked_fill(~assignment, 0.0)
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(~has_key.squeeze(-1), float("inf"))

    merged_key = torch.einsum("bhafkjcl,bfklhd->bafjkchd", weights.to(key.dtype), key)
    merged_value = torch.einsum("bhafkjcl,bfklhe->bafjkche", weights.to(value.dtype), value)
    return merged_key, merged_value, routing_bias


def biased_compressed_sdpa(query, merged_key, merged_value, routing_bias, scale):
    q = rearrange(query, "b a i j h d -> (b h a j) i d")
    k = rearrange(merged_key, "b a f j k c h d -> (b h a j) (f k c) d")
    v = rearrange(merged_value, "b a f j k c h d -> (b h a j) (f k c) d")
    bias = rearrange(routing_bias, "b h a f k j c -> (b h a j) (f k c)")
    out = F.scaled_dot_product_attention(
        q.to(k.dtype).unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        attn_mask=(-bias.float()).to(k.dtype).unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).squeeze(1)
    return rearrange(
        out,
        "(b h a j) i d -> b a i j h d",
        b=query.size(0),
        h=query.size(4),
        a=query.size(1),
        j=query.size(3),
    )


def flat_keyblock_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    *,
    block_tokens,
    num_seeds,
    seed_pattern,
    query_col_group,
    query_outer_chunk,
    head_chunk,
):
    frames, height, width = [int(value) for value in grid_size]
    q_blocks = video_to_query_blocks(q, frames, height, width)
    k_blocks = video_to_flat_key_blocks(k, frames, height, width, block_tokens)
    v_blocks = video_to_flat_key_blocks(v, frames, height, width, block_tokens)
    key_mask = flat_key_mask(height, width, block_tokens, q.device)
    seed_indices = local_seed_indices(height, width, block_tokens, num_seeds, q.device, seed_pattern)

    out = torch.empty_like(q_blocks)
    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            q_chunk = q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous()
            k_chunk = k_blocks[:, :, :, :, h_start:h_end].contiguous()
            v_chunk = v_blocks[:, :, :, :, h_start:h_end].contiguous()
            if query_col_group == 1:
                seeds = seed_indices.view(1, 1, seed_indices.size(0), 1, seed_indices.size(1))
                seeds = seeds.expand(q_chunk.size(1), k_chunk.size(1), -1, q_chunk.size(3), -1)
                seed_queries = gather_seed_queries(q_chunk, seeds)
                merged_key, merged_value, routing_bias = merge_kv_masked(
                    seed_queries,
                    k_chunk,
                    v_chunk,
                    key_mask,
                    scale,
                )
                out[:, q_start:q_end, :, :, h_start:h_end] = biased_compressed_sdpa(
                    q_chunk,
                    merged_key,
                    merged_value,
                    routing_bias,
                    scale,
                )
                continue

            for col_start in range(0, width, query_col_group):
                col_end = min(col_start + query_col_group, width)
                q_cols = q_chunk[:, :, :, col_start:col_end]
                rep_col = col_start + (col_end - col_start - 1) // 2

                seeds = seed_indices.view(1, 1, seed_indices.size(0), 1, seed_indices.size(1))
                seeds = seeds.expand(q_chunk.size(1), k_chunk.size(1), -1, 1, -1)
                seed_col = torch.full((1, 1, 1, 1, 1), rep_col, device=q.device, dtype=torch.long)
                seed_queries = gather_seed_queries(q_chunk, seeds, seed_col)
                merged_key, merged_value, routing_bias = merge_kv_masked(seed_queries, k_chunk, v_chunk, key_mask, scale)

                num_cols = col_end - col_start
                merged_key = merged_key.expand(-1, -1, -1, num_cols, -1, -1, -1, -1)
                merged_value = merged_value.expand_as(merged_key)
                routing_bias = routing_bias.expand(-1, -1, -1, -1, -1, num_cols, -1)
                out[:, q_start:q_end, :, col_start:col_end, h_start:h_end] = biased_compressed_sdpa(
                    q_cols,
                    merged_key,
                    merged_value,
                    routing_bias,
                    scale,
                )
    return query_blocks_to_video(out)


def parse_variant(name):
    match = VARIANT_RE.match(name)
    if not match:
        raise ValueError(f"variant must look like flat64_c4, got {name!r}")
    return {
        "block_tokens": int(match.group("block")),
        "num_seeds": int(match.group("c")),
        "query_col_group": int(match.group("qg") or 1),
        "seed_pattern": match.group("pattern") or "balanced",
    }


def variant_metadata(grid_size, spec):
    frames, height, width = [int(value) for value in grid_size]
    groups = (height * width + spec["block_tokens"] - 1) // spec["block_tokens"]
    compressed_tokens = frames * groups * spec["num_seeds"]
    return {
        **spec,
        "query_block_tokens": height * spec["query_col_group"],
        "key_block_tokens": spec["block_tokens"],
        "key_groups_per_frame": groups,
        "query_col_group": spec["query_col_group"],
        "query_groups_per_frame": (width + spec["query_col_group"] - 1) // spec["query_col_group"],
        "approx_merge_cost_scale_vs_qg1": 1.0 / spec["query_col_group"],
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
                    lambda spec=spec: flat_keyblock_attention(
                        q,
                        k,
                        v,
                        grid_size,
                        scale,
                        block_tokens=spec["block_tokens"],
                        num_seeds=spec["num_seeds"],
                        seed_pattern=spec["seed_pattern"],
                        query_col_group=spec["query_col_group"],
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
    parser.add_argument("--variants", default="flat64_c4,flat64_c5,flat64_c6,flat128_c8,flat128_c10,flat128_c12")
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
        "variants_vs_dense": {
            name: value.summary(dense_seconds=dense_seconds if name != "dense" else None, videos=len(prompts))
            for name, value in sorted(stats.items())
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
