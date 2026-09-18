"""Screen GridMerge layouts that change both query and key blocks.

Variants:

- q60_k64_c4: query block is two adjacent image columns, so 30*2=60 tokens.
  key/value blocks are row-major contiguous 64-token chunks.
- q60_k64_c4_x2r2: the same 60 x 64 block, but each C4 seed set uses two
  row/column intersection queries and two deterministic random queries.
- q64_k64_c4: query and key/value blocks are both row-major contiguous
  64-token chunks.

Both variants keep the key-softmax assignment and the full compressed
second-stage attention.
"""

from __future__ import annotations

import argparse
import json
import random
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


VARIANT_RE = re.compile(r"q(?P<q>\d+)_k(?P<k>\d+)_c(?P<c>\d+)(?:_(?P<seed>[a-z0-9]+))?$")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def video_tokens(x, frames, height, width):
    return x.reshape(x.size(0), frames, height * width, x.size(2), x.size(3)).contiguous()


def flat_blocks(video, block_tokens):
    batch, frames, tokens, heads, dim = video.shape
    groups = (tokens + block_tokens - 1) // block_tokens
    padded = video.new_zeros(batch, frames, groups * block_tokens, heads, dim)
    padded[:, :, :tokens] = video
    return padded.reshape(batch, frames, groups, block_tokens, heads, dim).contiguous()


def flat_indices(tokens, block_tokens, device):
    groups = (tokens + block_tokens - 1) // block_tokens
    indices = torch.arange(groups * block_tokens, device=device).reshape(groups, block_tokens)
    return indices.clamp_max(tokens - 1), indices < tokens


def column_pair_indices(height, width, device):
    cols_per_group = 2
    groups = (width + cols_per_group - 1) // cols_per_group
    indices = torch.zeros(groups, height * cols_per_group, device=device, dtype=torch.long)
    mask = torch.zeros_like(indices, dtype=torch.bool)
    local = 0
    for group in range(groups):
        cols = [col for col in range(group * cols_per_group, min((group + 1) * cols_per_group, width))]
        local = 0
        for row in range(height):
            for col in cols:
                indices[group, local] = row * width + col
                mask[group, local] = True
                local += 1
    return indices, mask


def gather_query_blocks(video, indices):
    batch, frames, _tokens, heads, dim = video.shape
    groups, block_tokens = indices.shape
    gather_idx = indices.view(1, 1, groups, block_tokens, 1, 1).expand(batch, frames, groups, block_tokens, heads, dim)
    return torch.gather(video[:, :, None].expand(batch, frames, groups, -1, heads, dim), 3, gather_idx).contiguous()


def q60_seed_positions(q_indices, k_indices, height, width, num_seeds, seed_mode):
    if num_seeds != 4:
        raise ValueError("q60 key-row seeds currently expect C=4.")
    query_cols = q_indices % width
    first_col = query_cols[:, 0]
    second_col = torch.clamp(first_col + 1, max=width - 1)
    key_rows = k_indices // width
    row0 = key_rows[:, 0]
    row1 = key_rows[:, -1]

    if seed_mode == "x2r2":
        rng = random.Random(20260813)
        center_rows = ((row0 + row1) // 2).tolist()
        positions = []
        for _query_group in range(q_indices.size(0)):
            group_positions = []
            for row in center_rows:
                intersections = [int(row) * 2, int(row) * 2 + 1]
                pool = [idx for idx in range(height * 2) if idx not in intersections]
                group_positions.append(intersections + rng.sample(pool, 2))
            positions.append(group_positions)
        return torch.tensor(positions, device=q_indices.device, dtype=torch.long)

    local = []
    for row in (row0, row1):
        for col in (first_col, second_col):
            local.append(row.view(1, -1) * 2 + (col - first_col).view(-1, 1))
    return torch.stack(local, dim=2).clamp(0, height * 2 - 1)


def balanced_seed_positions(q_mask, k_groups, num_seeds):
    counts = q_mask.long().sum(dim=1)
    fractions = torch.linspace(0, 1, num_seeds, device=q_mask.device)
    positions = torch.round((counts[:, None] - 1).clamp_min(0) * fractions[None]).long()
    return positions[:, None, :].expand(q_mask.size(0), k_groups, num_seeds)


def gather_seed_queries(query_blocks, seed_positions):
    batch, query_outer, q_groups, _q_tokens, heads, dim = query_blocks.shape
    k_groups, num_seeds = seed_positions.size(1), seed_positions.size(2)
    gather_idx = seed_positions.view(1, 1, q_groups, k_groups, num_seeds, 1, 1)
    gather_idx = gather_idx.expand(batch, query_outer, q_groups, k_groups, num_seeds, heads, dim)
    source = query_blocks[:, :, :, None].expand(batch, query_outer, q_groups, k_groups, -1, heads, dim)
    return torch.gather(source, 4, gather_idx).contiguous()


def merge_kv(seed_queries, key_blocks, value_blocks, key_mask, scale):
    logits = torch.einsum("bagkchd,bfklhd->bhafgkcl", seed_queries, key_blocks).float()
    logits = logits * float(scale)
    valid_key = key_mask.view(1, 1, 1, 1, 1, key_mask.size(0), 1, key_mask.size(1))
    logits = logits.masked_fill(~valid_key, FLOAT_NEG_INF)

    assignment_scores = torch.log_softmax(logits, dim=-1)
    best_seed = assignment_scores.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)
    assignment = assignment & valid_key

    masked_logits = logits.masked_fill(~assignment, FLOAT_NEG_INF)
    has_key = assignment.any(dim=-1, keepdim=True)
    safe_logits = torch.where(has_key, masked_logits, torch.zeros_like(masked_logits))
    log_weights = torch.log_softmax(safe_logits, dim=-1).masked_fill(~assignment, 0.0)
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1).masked_fill(~has_key.squeeze(-1), float("inf"))

    merged_key = torch.einsum("bhafgkcl,bfklhd->bafgkchd", weights.to(key_blocks.dtype), key_blocks)
    merged_value = torch.einsum("bhafgkcl,bfklhe->bafgkche", weights.to(value_blocks.dtype), value_blocks)
    return merged_key, merged_value, routing_bias


def compressed_attention(query_blocks, merged_key, merged_value, routing_bias, scale):
    q = rearrange(query_blocks, "b a g q h d -> (b h a g) q d")
    k = rearrange(merged_key, "b a f g k c h d -> (b h a g) (f k c) d")
    v = rearrange(merged_value, "b a f g k c h d -> (b h a g) (f k c) d")
    bias = rearrange(routing_bias, "b h a f g k c -> (b h a g) (f k c)")
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
        "(b h a g) q d -> b a g q h d",
        b=query_blocks.size(0),
        h=query_blocks.size(4),
        a=query_blocks.size(1),
        g=query_blocks.size(2),
    )


def scatter_query_blocks(blocks, indices, mask, tokens):
    batch, frames, groups, block_tokens, heads, dim = blocks.shape
    flat_blocks_out = blocks.masked_fill(~mask.view(1, 1, groups, block_tokens, 1, 1), 0.0)
    out = blocks.new_zeros(batch, frames, tokens, heads, dim)
    scatter_idx = indices.view(1, 1, groups * block_tokens, 1, 1).expand(batch, frames, -1, heads, dim)
    out.scatter_add_(2, scatter_idx, flat_blocks_out.reshape(batch, frames, groups * block_tokens, heads, dim))
    return out.reshape(batch, frames * tokens, heads, dim).contiguous()


def parse_variant(name):
    match = VARIANT_RE.match(name)
    if not match:
        raise ValueError(f"variant must look like q60_k64_c4 or q64_k64_c4, got {name!r}")
    return {
        "query_tokens": int(match.group("q")),
        "key_tokens": int(match.group("k")),
        "num_seeds": int(match.group("c")),
        "seed_mode": match.group("seed") or "default",
    }


def block_attention(q, k, v, grid_size, scale, spec, query_outer_chunk, head_chunk):
    frames, height, width = [int(value) for value in grid_size]
    tokens = height * width
    query_tokens = spec["query_tokens"]
    key_tokens = spec["key_tokens"]
    num_seeds = spec["num_seeds"]

    q_video = video_tokens(q, frames, height, width)
    k_video = video_tokens(k, frames, height, width)
    v_video = video_tokens(v, frames, height, width)
    key_blocks = flat_blocks(k_video, key_tokens)
    value_blocks = flat_blocks(v_video, key_tokens)
    key_indices, key_mask = flat_indices(tokens, key_tokens, q.device)

    if query_tokens == 60:
        query_indices, query_mask = column_pair_indices(height, width, q.device)
        seed_positions = q60_seed_positions(query_indices, key_indices, height, width, num_seeds, spec["seed_mode"])
    elif query_tokens == 64:
        query_indices, query_mask = flat_indices(tokens, query_tokens, q.device)
        seed_positions = balanced_seed_positions(query_mask, key_indices.size(0), num_seeds)
    else:
        raise ValueError(f"unsupported query block size: {query_tokens}")

    query_blocks = gather_query_blocks(q_video, query_indices)
    out_blocks = torch.empty_like(query_blocks)
    for q_start in range(0, frames, query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, frames)
        for h_start in range(0, q.size(2), head_chunk):
            h_end = min(h_start + head_chunk, q.size(2))
            query_chunk = query_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous()
            key_chunk = key_blocks[:, :, :, :, h_start:h_end].contiguous()
            value_chunk = value_blocks[:, :, :, :, h_start:h_end].contiguous()
            seed_queries = gather_seed_queries(query_chunk, seed_positions)
            merged_key, merged_value, routing_bias = merge_kv(seed_queries, key_chunk, value_chunk, key_mask, scale)
            out_blocks[:, q_start:q_end, :, :, h_start:h_end] = compressed_attention(
                query_chunk,
                merged_key,
                merged_value,
                routing_bias,
                scale,
            )
    return scatter_query_blocks(out_blocks, query_indices, query_mask, tokens)


def variant_metadata(grid_size, spec):
    frames, height, width = [int(value) for value in grid_size]
    tokens = height * width
    query_groups = (width + 1) // 2 if spec["query_tokens"] == 60 else (tokens + spec["query_tokens"] - 1) // spec["query_tokens"]
    key_groups = (tokens + spec["key_tokens"] - 1) // spec["key_tokens"]
    compressed_tokens = frames * key_groups * spec["num_seeds"]
    return {
        **spec,
        "query_groups_per_frame": query_groups,
        "key_groups_per_frame": key_groups,
        "compressed_tokens_per_query": compressed_tokens,
        "compressed_token_density": compressed_tokens / (frames * tokens),
        "paper_density_c": spec["num_seeds"] * (spec["query_tokens"] + spec["key_tokens"]) / (spec["query_tokens"] * spec["key_tokens"]),
    }


def install_probe(stats, args):
    import wan.modules.model as wan_model

    variants = [name.strip() for name in args.variants.replace(",", " ").split() if name.strip()]
    variant_specs = {name: parse_variant(name) for name in variants}
    backends = {name.strip() for name in args.reference_backends.replace(",", " ").split() if name.strip()}
    state = {"calls_total": 0, "calls_recorded": 0, "calls_current_prompt": 0, "grid_size": None, "shape": None}

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
                    lambda spec=spec: block_attention(q, k, v, grid_size, scale, spec, args.query_outer_chunk, args.head_chunk)
                )
                stats[name].add_time(seconds)
                stats[name].add_error(out, dense)
            state["calls_recorded"] += 1
            out = dense
        else:
            out = dense_attention(q, k, v, self)
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
    parser.add_argument("--variants", default="q60_k64_c4,q64_k64_c4")
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
