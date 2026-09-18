import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.benchmark_h200_attention_backend_error_time import (
    Stats,
    hard_c4_attention,
    load_pipeline,
)
from analysis.explore_rowcol_block_layouts import (
    biased_compressed_sdpa,
    merge_kv,
)
from attention.flash_attn_wan21_1_3b_480p_triton import flash_attention_triton
from generate import load_prompts
from utils.misc import set_seed


AXES = ("t", "h", "w")
VARIANT_RE = re.compile(r"q(?P<q_axis>[thw])_k(?P<k_axis>[thw])_c(?P<c>\d+)$")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def axis_lengths(grid_size):
    frames, height, width = [int(value) for value in grid_size]
    return {"t": frames, "h": height, "w": width}


def video_view(x, grid_size):
    frames, height, width = [int(value) for value in grid_size]
    return x.reshape(x.size(0), frames, height, width, x.size(2), x.size(3))


def to_query_blocks(x, grid_size, q_axis):
    other = [axis for axis in AXES if axis != q_axis]
    axis_to_dim = {"t": 1, "h": 2, "w": 3}
    video = video_view(x, grid_size)
    return video.permute(
        0,
        axis_to_dim[other[0]],
        axis_to_dim[q_axis],
        axis_to_dim[other[1]],
        4,
        5,
    ).contiguous()


def query_blocks_to_video(x, grid_size, q_axis):
    other = [axis for axis in AXES if axis != q_axis]
    position = {other[0]: 0, q_axis: 1, other[1]: 2}
    perm = [0, *(1 + position[axis] for axis in AXES), 4, 5]
    video = x.permute(*perm).contiguous()
    batch, frames, height, width, heads, dim = video.shape
    return video.reshape(batch, frames * height * width, heads, dim)


def to_key_blocks(x, grid_size, k_axis):
    other = [axis for axis in AXES if axis != k_axis]
    axis_to_dim = {"t": 1, "h": 2, "w": 3}
    video = video_view(x, grid_size)
    return video.permute(
        0,
        axis_to_dim[other[0]],
        axis_to_dim[other[1]],
        axis_to_dim[k_axis],
        4,
        5,
    ).contiguous()


def local_seed_indices(q_axis, k_axis, grid_size, num_seeds, device):
    lengths = axis_lengths(grid_size)
    q_len = lengths[q_axis]
    if q_axis == k_axis:
        raise ValueError("q_axis and k_axis must differ for matched local seeds.")
    if num_seeds > q_len:
        raise ValueError(f"num_seeds={num_seeds} exceeds query inner length={q_len}.")

    key_outer_axes = [axis for axis in AXES if axis != k_axis]
    outer0 = torch.arange(lengths[key_outer_axes[0]], device=device)
    outer1 = torch.arange(lengths[key_outer_axes[1]], device=device)
    if q_axis == key_outer_axes[0]:
        base = outer0.view(-1, 1).expand(-1, outer1.numel())
    else:
        base = outer1.view(1, -1).expand(outer0.numel(), -1)

    offsets = [0]
    for delta in range(1, q_len):
        offsets.extend((-delta, delta))

    seeds = []
    for offset in offsets:
        candidate = base + int(offset)
        candidate = candidate.clamp(0, q_len - 1)
        if seeds:
            duplicate = torch.zeros_like(candidate, dtype=torch.bool)
            for existing in seeds:
                duplicate |= candidate == existing
            candidate = torch.where(duplicate, torch.full_like(candidate, -1), candidate)
        seeds.append(candidate)
        stacked = torch.stack(seeds, dim=-1)
        if (stacked >= 0).sum(dim=-1).amin().item() >= num_seeds:
            break

    stacked = torch.stack(seeds, dim=-1)
    valid = stacked >= 0
    filled = []
    for outer_i in range(stacked.size(0)):
        row = []
        for outer_j in range(stacked.size(1)):
            values = stacked[outer_i, outer_j][valid[outer_i, outer_j]][:num_seeds]
            row.append(values)
        filled.append(torch.stack(row, dim=0))
    seed_by_key_group = torch.stack(filled, dim=0).long()
    return seed_by_key_group


def gather_seed_queries(query, seed_by_key_group):
    query_outer = torch.arange(query.size(1), device=query.device).view(-1, 1, 1, 1, 1)
    query_group = torch.arange(query.size(3), device=query.device).view(1, 1, 1, -1, 1)
    seed = seed_by_key_group.view(1, seed_by_key_group.size(0), seed_by_key_group.size(1), 1, seed_by_key_group.size(2))
    seed = seed.expand(query.size(1), -1, -1, query.size(3), -1)
    return query[:, query_outer, seed, query_group]


def axis_layout_attention(
    q,
    k,
    v,
    grid_size,
    scale,
    *,
    q_axis,
    k_axis,
    num_seeds,
    query_outer_chunk,
    head_chunk,
):
    q_blocks = to_query_blocks(q, grid_size, q_axis)
    k_blocks = to_key_blocks(k, grid_size, k_axis)
    v_blocks = to_key_blocks(v, grid_size, k_axis)
    seeds = local_seed_indices(q_axis, k_axis, grid_size, num_seeds, q.device)

    out = torch.empty_like(q_blocks)
    for q_start in range(0, q_blocks.size(1), query_outer_chunk):
        q_end = min(q_start + query_outer_chunk, q_blocks.size(1))
        for h_start in range(0, q_blocks.size(4), head_chunk):
            h_end = min(h_start + head_chunk, q_blocks.size(4))
            q_chunk = q_blocks[:, q_start:q_end, :, :, h_start:h_end].contiguous()
            k_chunk = k_blocks[:, :, :, :, h_start:h_end].contiguous()
            v_chunk = v_blocks[:, :, :, :, h_start:h_end].contiguous()
            seed_queries = gather_seed_queries(q_chunk, seeds)
            merged_key, merged_value, routing_bias = merge_kv(seed_queries, k_chunk, v_chunk, scale)
            out[:, q_start:q_end, :, :, h_start:h_end] = biased_compressed_sdpa(
                q_chunk,
                merged_key,
                merged_value,
                routing_bias,
                scale,
            )
    return query_blocks_to_video(out, grid_size, q_axis)


def parse_variant(name):
    match = VARIANT_RE.match(name)
    if not match:
        raise ValueError(f"variant must look like qh_kw_c4, got {name!r}")
    return {
        "q_axis": match.group("q_axis"),
        "k_axis": match.group("k_axis"),
        "num_seeds": int(match.group("c")),
    }


def variant_metadata(grid_size, spec):
    lengths = axis_lengths(grid_size)
    q_axis = spec["q_axis"]
    k_axis = spec["k_axis"]
    c = spec["num_seeds"]
    key_groups = 1
    query_groups = 1
    for axis in AXES:
        if axis != k_axis:
            key_groups *= lengths[axis]
        if axis != q_axis:
            query_groups *= lengths[axis]
    total_tokens = lengths["t"] * lengths["h"] * lengths["w"]
    return {
        "query_inner_tokens": lengths[q_axis],
        "key_inner_tokens": lengths[k_axis],
        "query_groups": query_groups,
        "key_groups": key_groups,
        "compressed_tokens_per_query": key_groups * c,
        "compressed_token_density": key_groups * c / total_tokens,
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
                    lambda spec=spec: axis_layout_attention(
                        q,
                        k,
                        v,
                        grid_size,
                        scale,
                        q_axis=spec["q_axis"],
                        k_axis=spec["k_axis"],
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
    parser = argparse.ArgumentParser(description="Screen axis choices for GridMerge blocks on real Wan QKV.")
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
        default="qh_kw_c4,qw_kh_c4,qt_kw_c4,qt_kh_c4,qh_kt_c4,qw_kt_c4",
    )
    parser.add_argument("--reference_backends", default="hard_c4_triton")
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
