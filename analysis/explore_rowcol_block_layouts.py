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

from analysis.benchmark_h200_attention_backend_error_time import (
    Stats,
    hard_c4_attention,
    load_pipeline,
)
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from generate import load_prompts
from utils.misc import set_seed


FLOAT_NEG_INF = torch.finfo(torch.float32).min
VARIANT_RE = re.compile(
    r"qc(?P<qcols>\d+)_kr(?P<krows>\d+)_c(?P<c>\d+)"
    r"(?:_p(?P<pattern>[a-z0-9]+))?(?:_g(?P<group>[a-z0-9]+))?$"
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


def dense_sdpa(query, key, value, scale):
    batch, seq_len, heads, dim = query.shape
    q = rearrange(query, "b n h d -> (b h) n d")
    k = rearrange(key, "b n h d -> (b h) n d")
    v = rearrange(value, "b n h d -> (b h) n d")
    out = F.scaled_dot_product_attention(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).squeeze(1)
    return rearrange(out, "(b h) n d -> b n h d", b=batch, h=heads)


def q_to_grouped_columns(x, frames, height, width, query_column_group):
    video = x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3))
    return rearrange(
        video,
        "b t h (j gc) head dim -> b t (gc h) j head dim",
        gc=query_column_group,
    ).contiguous()


def grouped_columns_to_video(x, height, width, query_column_group):
    video = rearrange(
        x,
        "b t (gc h) j head dim -> b t h (j gc) head dim",
        gc=query_column_group,
        h=height,
    )
    return video.reshape(x.size(0), -1, x.size(4), x.size(5)).contiguous()


def kv_to_grouped_rows(x, frames, height, width, key_row_group, row_group_mode="contiguous"):
    video = x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3))
    if row_group_mode == "interleave":
        row_groups = height // key_row_group
        return rearrange(
            video,
            "b t (gr k) w head dim -> b t k (gr w) head dim",
            gr=key_row_group,
            k=row_groups,
        ).contiguous()
    return rearrange(
        video,
        "b t (k gr) w head dim -> b t k (gr w) head dim",
        gr=key_row_group,
    ).contiguous()


def seed_row_candidates(base_row, height, key_row_group, pattern):
    if pattern == "right":
        return list(range(base_row, height)) + list(range(base_row - 1, -1, -1))
    if pattern == "left":
        return list(range(base_row + key_row_group - 1, -1, -1)) + list(
            range(base_row + key_row_group, height)
        )
    if pattern == "wide":
        rows = []
        for delta in range(height):
            rows.append(base_row + delta)
            rows.append(base_row + key_row_group - 1 - delta)
        return rows

    offsets = [0]
    for delta in range(1, height):
        offsets.extend((-delta, delta))
    rows = []
    for offset in offsets:
        for local_key_row in range(key_row_group):
            rows.append(base_row + local_key_row + offset)
    return rows


def interleaved_seed_candidates(key_group, height, key_row_group, key_groups):
    key_rows = [key_group + local_row * key_groups for local_row in range(key_row_group)]
    rows = []
    for delta in range(height):
        for row in key_rows:
            if delta == 0:
                rows.append(row)
            else:
                rows.extend((row - delta, row + delta))
    return rows


def local_seed_indices(
    height,
    query_column_group,
    key_row_group,
    key_groups,
    num_seeds,
    device,
    seed_pattern="balanced",
    row_group_mode="contiguous",
):
    query_rows = height * query_column_group
    if num_seeds > query_rows:
        raise ValueError(
            f"num_seeds={num_seeds} exceeds query rows per block={query_rows}"
        )
    rows = []
    for key_group in range(key_groups):
        base_row = key_group * key_row_group
        selected = []
        used = set()
        if row_group_mode == "interleave":
            candidates = interleaved_seed_candidates(key_group, height, key_row_group, key_groups)
        else:
            candidates = seed_row_candidates(base_row, height, key_row_group, seed_pattern)
        for row in candidates:
            for local_col in range(query_column_group):
                if row < 0 or row >= height:
                    continue
                index = local_col * height + row
                if index in used:
                    continue
                used.add(index)
                selected.append(index)
                if len(selected) == num_seeds:
                    break
            if len(selected) == num_seeds:
                break
        rows.append(selected)
    return torch.tensor(rows, device=device, dtype=torch.long)


def gather_seed_queries(query, seed_indices):
    query_outer = torch.arange(query.size(1), device=query.device).view(-1, 1, 1, 1, 1)
    query_group = torch.arange(query.size(3), device=query.device).view(1, 1, 1, -1, 1)
    return query[:, query_outer, seed_indices, query_group]


def merge_kv(seed_queries, key, value, scale):
    seed_key_logits = torch.einsum("bafkjchd,bfklhd->bhafkjcl", seed_queries, key).float()
    seed_key_logits = seed_key_logits * float(scale)

    assignment_scores = torch.log_softmax(seed_key_logits, dim=-1)
    best_seed = assignment_scores.argmax(dim=-2, keepdim=True)
    assignment = torch.zeros_like(seed_key_logits, dtype=torch.bool)
    assignment.scatter_(dim=-2, index=best_seed, value=True)

    logits = seed_key_logits.masked_fill(~assignment, FLOAT_NEG_INF)
    log_weights = torch.log_softmax(logits, dim=-1).masked_fill(~assignment, 0.0)
    weights = log_weights.exp().masked_fill(~assignment, 0.0)
    routing_bias = (weights * log_weights).sum(dim=-1)
    routing_bias = routing_bias.masked_fill(~assignment.any(dim=-1), float("inf"))

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
    return rearrange(out, "(b h a j) i d -> b a i j h d", b=query.size(0), h=query.size(4), a=query.size(1), j=query.size(3))


def rowcol_layout_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    *,
    query_column_group,
    key_row_group,
    num_seeds,
    seed_pattern,
    row_group_mode,
    query_outer_chunk,
    head_chunk,
):
    frames, height, width = [int(value) for value in grid_size]
    if width % query_column_group:
        raise ValueError("query_column_group must divide width for this screening script.")
    if height % key_row_group:
        raise ValueError("key_row_group must divide height for this screening script.")
    if row_group_mode not in {"contiguous", "interleave"}:
        raise ValueError(f"unknown row_group_mode={row_group_mode!r}")

    q_blocks = q_to_grouped_columns(q, frames, height, width, query_column_group)
    k_blocks = kv_to_grouped_rows(k, frames, height, width, key_row_group, row_group_mode)
    v_blocks = kv_to_grouped_rows(v, frames, height, width, key_row_group, row_group_mode)
    seed_indices = local_seed_indices(
        height,
        query_column_group,
        key_row_group,
        k_blocks.size(2),
        num_seeds,
        q.device,
        seed_pattern,
        row_group_mode,
    )

    out = torch.empty_like(q_blocks)
    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            q_chunk = q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous()
            k_chunk = k_blocks[:, :, :, :, h_start:h_end].contiguous()
            v_chunk = v_blocks[:, :, :, :, h_start:h_end].contiguous()
            seeds = seed_indices.view(1, 1, seed_indices.size(0), 1, seed_indices.size(1))
            seeds = seeds.expand(q_chunk.size(1), k_chunk.size(1), -1, q_chunk.size(3), -1)
            seed_queries = gather_seed_queries(q_chunk, seeds)
            merged_key, merged_value, routing_bias = merge_kv(seed_queries, k_chunk, v_chunk, scale)
            out[:, q_start:q_end, :, :, h_start:h_end] = biased_compressed_sdpa(
                q_chunk,
                merged_key,
                merged_value,
                routing_bias,
                scale,
            )
    return grouped_columns_to_video(out, height, width, query_column_group)


def parse_variant(name):
    match = VARIANT_RE.match(name)
    if not match:
        raise ValueError(f"variant must look like qc2_kr2_c8, got {name!r}")
    return {
        "query_column_group": int(match.group("qcols")),
        "key_row_group": int(match.group("krows")),
        "num_seeds": int(match.group("c")),
        "seed_pattern": match.group("pattern") or "balanced",
        "row_group_mode": match.group("group") or "contiguous",
    }


def variant_metadata(grid_size, spec):
    frames, height, width = [int(value) for value in grid_size]
    qcg = spec["query_column_group"]
    krg = spec["key_row_group"]
    c = spec["num_seeds"]
    key_groups = height // krg
    query_groups = width // qcg
    compressed_tokens = frames * key_groups * c
    return {
        "query_block_tokens": height * qcg,
        "key_block_tokens": width * krg,
        "query_groups_per_frame": query_groups,
        "key_groups_per_frame": key_groups,
        "row_group_mode": spec["row_group_mode"],
        "seed_pattern": spec["seed_pattern"],
        "compressed_tokens_per_query": compressed_tokens,
        "compressed_token_density": compressed_tokens / (frames * height * width),
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
        state["shape"] = {
            "batch": batch,
            "seq_len": seq_len,
            "heads": heads,
            "dim": dim,
        }
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
                    lambda spec=spec: rowcol_layout_attention(
                        q,
                        k,
                        v,
                        grid_size,
                        scale,
                        query_column_group=spec["query_column_group"],
                        key_row_group=spec["key_row_group"],
                        num_seeds=spec["num_seeds"],
                        seed_pattern=spec["seed_pattern"],
                        row_group_mode=spec["row_group_mode"],
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
    parser = argparse.ArgumentParser(description="Screen row/column GridMerge block layouts on real Wan QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=2)
    parser.add_argument("--query_outer_chunk", type=int, default=1)
    parser.add_argument("--head_chunk", type=int, default=1)
    parser.add_argument("--hard_query_outer_chunk", type=int, default=10)
    parser.add_argument(
        "--variants",
        default="qc1_kr2_c8,qc2_kr2_c8,qc4_kr2_c8,qc1_kr3_c12,qc2_kr3_c12,qc1_kr5_c20",
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
