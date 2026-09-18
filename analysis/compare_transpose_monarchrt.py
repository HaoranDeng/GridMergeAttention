import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


HASH_MOD = 2147483647


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

    def add_error(self, out, dense):
        out_f = out.float().reshape(-1)
        dense_f = dense.float().reshape(-1)
        diff = out_f - dense_f
        self.sqerr += float(torch.dot(diff, diff).item())
        self.ref_sq += float(torch.dot(dense_f, dense_f).clamp_min(1e-20).item())
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


def pad_to_multiples(x, frames, height, width, h_reduce, w_reduce):
    padded_h = math.ceil(height / h_reduce) * h_reduce
    padded_w = math.ceil(width / w_reduce) * w_reduce
    if padded_h == height and padded_w == width:
        return x, None, height, width

    batch, _, heads, dim = x.shape
    blocks = x.reshape(batch, frames, height, width, heads, dim)
    padded = x.new_zeros(batch, frames, padded_h, padded_w, heads, dim)
    padded[:, :, :height, :width] = blocks

    valid = torch.zeros(batch, frames, padded_h, padded_w, 1, 1, device=x.device, dtype=torch.bool)
    valid[:, :, :height, :width] = True
    return (
        padded.reshape(batch, frames * padded_h * padded_w, heads, dim),
        valid.reshape(batch, frames * padded_h * padded_w, 1, 1),
        padded_h,
        padded_w,
    )


def crop_from_padded(x, frames, height, width, padded_h, padded_w):
    if padded_h == height and padded_w == width:
        return x
    batch, _, heads, dim = x.shape
    return (
        x.reshape(batch, frames, padded_h, padded_w, heads, dim)[:, :, :height, :width]
        .reshape(batch, frames * height * width, heads, dim)
    )


def video_to_reduced_blocks(x, f_tied, h_reduce, w_reduce, height, width):
    """[B, F*H*W, Hh, D] -> [B, outer, row, col, Hh, D]."""
    batch, seq_len, heads, dim = x.shape
    frame_groups = seq_len // (f_tied * h_reduce * (height // h_reduce) * w_reduce * (width // w_reduce))
    return rearrange(
        x,
        "b (outer ft hr row wr col) head dim -> b (outer hr wr) (ft row) col head dim",
        outer=frame_groups,
        ft=f_tied,
        hr=h_reduce,
        row=height // h_reduce,
        wr=w_reduce,
        col=width // w_reduce,
        head=heads,
        dim=dim,
    )


def reduced_blocks_to_video(x, f_tied, h_reduce, w_reduce, height, width):
    """[B, outer, row, col, Hh, D] -> [B, F*H*W, Hh, D]."""
    batch, outer_blocks, row, col, heads, dim = x.shape
    frame_groups = outer_blocks // (h_reduce * w_reduce)
    return rearrange(
        x,
        "b (outer hr wr) (ft row) col head dim -> b (outer ft hr row wr col) head dim",
        outer=frame_groups,
        ft=f_tied,
        hr=h_reduce,
        row=row,
        wr=w_reduce,
        col=col,
        head=heads,
        dim=dim,
    )


def merge_query_row_ith(q, key_rows):
    if q.size(2) == key_rows:
        return q
    index = torch.arange(key_rows, device=q.device) % q.size(2)
    return q.index_select(2, index)


def merge_query_col_ith(q, key_cols):
    if q.size(3) == key_cols:
        return q
    index = torch.arange(key_cols, device=q.device) % q.size(3)
    return q.index_select(3, index)


def masked_soft_merge(logits, valid, dtype):
    logits = logits.float()
    logits = logits.masked_fill(~valid, float("-inf"))
    logprob = torch.log_softmax(logits, dim=-1)
    weights = logprob.exp().masked_fill(~valid, 0.0).to(dtype)
    return weights, logprob


def soft_merge(logits, dtype):
    logits = logits.float()
    logprob = torch.log_softmax(logits, dim=-1)
    return logprob.exp().to(dtype), logprob


def hashed_candidate_scores(base, num_choices):
    candidate = torch.arange(num_choices, device=base.device).view(*((1,) * base.ndim), num_choices)
    scores = (base.unsqueeze(-1).remainder(HASH_MOD) + candidate * 374761393).remainder(HASH_MOD)
    scores = (scores * (scores + 1274126177)).remainder(HASH_MOD)
    scores = (scores * (scores + 60493)).remainder(HASH_MOD)
    return scores * num_choices + candidate, candidate


def seed_indices_from_base(base, num_choices, num_seeds, include_seed=None):
    scores, candidate = hashed_candidate_scores(base, num_choices)
    if include_seed is not None:
        scores = scores.masked_fill(candidate == include_seed.unsqueeze(-1), -1)
        random_seeds = scores.topk(num_seeds - 1, dim=-1).indices.long()
        return torch.cat((include_seed.unsqueeze(-1).long(), random_seeds), dim=-1)
    return scores.topk(num_seeds, dim=-1).indices.long()


def random_seed_columns(query_outer, key_outer, query_rows, key_columns, query_columns, num_seeds, seed):
    query_row = torch.arange(query_rows, device=query_outer.device).view(1, 1, query_rows, 1)
    key_col = torch.arange(key_columns, device=query_outer.device).view(1, 1, 1, key_columns)
    base = (
        query_outer.view(-1, 1, 1, 1) * 73856093
        + key_outer.view(1, -1, 1, 1) * 19349663
        + query_row * 83492791
        + key_col * 2654435761
        + int(seed)
    )
    ith_seed = key_col.expand(query_outer.numel(), key_outer.numel(), query_rows, key_columns)
    return seed_indices_from_base(base, query_columns, num_seeds, include_seed=ith_seed)


def gather_seed_queries_transpose(q, seed_cols):
    query_outer = torch.arange(q.size(1), device=q.device).view(-1, 1, 1, 1, 1)
    query_row = torch.arange(q.size(2), device=q.device).view(1, 1, -1, 1, 1)
    return q[:, query_outer, query_row, seed_cols]


def biased_sdpa(query, key, value, bias, scale):
    out = F.scaled_dot_product_attention(
        query.unsqueeze(1),
        key.unsqueeze(1),
        value.unsqueeze(1),
        attn_mask=(-bias.float()).to(query.dtype).unsqueeze(1).unsqueeze(2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(1)


def hard_route_transpose_c4_blocks(q, k, v, scale, seed=0):
    """Historical transpose C4 route: fixed (a, f, query-row, key-column) tiles."""
    query_outer = torch.arange(q.size(1), device=q.device)
    key_outer = torch.arange(k.size(1), device=q.device)
    seed_cols = random_seed_columns(
        query_outer,
        key_outer,
        q.size(2),
        k.size(3),
        q.size(3),
        4,
        seed,
    )
    seed_queries = gather_seed_queries_transpose(q, seed_cols)

    seed_logits = torch.einsum("bafilchd,bfklhd->bhafilck", seed_queries, k).float()
    seed_logits = seed_logits * float(scale)
    assignments = seed_logits.argmax(dim=-2)
    assignment_mask = F.one_hot(assignments, num_classes=4).movedim(-1, -2).bool()

    has_seed = assignment_mask.any(dim=-1)
    key_idx = torch.arange(k.size(2), device=k.device).view(*((1,) * has_seed.ndim), k.size(2))
    assignment_mask = assignment_mask | ((~has_seed).unsqueeze(-1) & (key_idx == 0))

    masked_logits = seed_logits.masked_fill(~assignment_mask, torch.finfo(seed_logits.dtype).min)
    merge_logprob = torch.log_softmax(masked_logits, dim=-1)
    merge_weights = merge_logprob.exp().masked_fill(~assignment_mask, 0).to(k.dtype)
    merge_logprob = merge_logprob.masked_fill(~assignment_mask, 0)

    merged_k = torch.einsum("bhafilck,bfklhd->bafilchd", merge_weights, k)
    merged_v = torch.einsum("bhafilck,bfklhe->bafilche", merge_weights, v)
    routing_bias = (merge_weights.float() * merge_logprob).sum(dim=-1)

    query = rearrange(q, "b a i j h d -> (b h a i) j d")
    key = rearrange(merged_k, "b a f i l c h d -> (b h a i) (f l c) d")
    value = rearrange(merged_v, "b a f i l c h d -> (b h a i) (f l c) d")
    bias = rearrange(routing_bias, "b h a f i l c -> (b h a i) (f l c)")
    out = biased_sdpa(query, key, value, bias, scale)
    return rearrange(out, "(b h a i) j d -> b a i j h d", b=q.size(0), h=q.size(4), a=q.size(1), i=q.size(2))


def gridmerge_one_blocks(q, k, v, scale, key_valid=None):
    """Original one-token MonarchRT: each 30-query x 52-key tile becomes rank-1."""
    scale_sqrt = float(scale) ** 0.5
    q = q * scale_sqrt
    k = k * scale_sqrt
    seed_q = merge_query_row_ith(q, k.size(2))

    logits = torch.einsum("bakjhd,bfklhd->bhafkjl", seed_q, k)
    if key_valid is None:
        merge_weights, merge_logprob = soft_merge(logits, k.dtype)
        compressed_valid = None
    else:
        merge_valid = key_valid[:, None, None, :, :, None, :]
        merge_weights, merge_logprob = masked_soft_merge(logits, merge_valid, k.dtype)
        compressed_valid = key_valid.any(dim=-1)

    merged_k = torch.einsum("bhafkjl,bfklhd->bafjkhd", merge_weights, k)
    merged_v = torch.einsum("bhafkjl,bfklhe->bafjkhe", merge_weights, v)
    bias = (merge_weights.float() * merge_logprob).sum(dim=-1, keepdim=True).transpose(-2, -3)

    routing_logits = torch.einsum("bafjkhd,baijhd->bhafjki", merged_k, q) - bias
    if compressed_valid is not None:
        routing_valid = compressed_valid[:, None, None, :, None, :, None]
        routing_logits = routing_logits.masked_fill(~routing_valid, float("-inf"))

    routing_weights = rearrange(routing_logits, "b h a f j k i -> b h a j i (f k)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a j i (f k) -> b h a f j k i",
        f=k.size(1),
        k=k.size(2),
    )
    return torch.einsum("bhafjki,bafjkhe->baijhe", routing_weights, merged_v)


def gridmerge_one_transpose_blocks(q, k, v, scale, key_valid=None):
    """Transpose one-token variant: each 52-query x 30-key tile becomes rank-1."""
    scale_sqrt = float(scale) ** 0.5
    q = q * scale_sqrt
    k = k * scale_sqrt
    seed_q = merge_query_col_ith(q, k.size(3))

    logits = torch.einsum("bailhd,bfklhd->bhafilk", seed_q, k)
    if key_valid is None:
        merge_weights, merge_logprob = soft_merge(logits, k.dtype)
        compressed_valid = None
    else:
        merge_valid = key_valid.permute(0, 1, 3, 2)[:, None, None, :, None, :, :]
        merge_weights, merge_logprob = masked_soft_merge(logits, merge_valid, k.dtype)
        compressed_valid = key_valid.any(dim=2)

    merged_k = torch.einsum("bhafilk,bfklhd->bafilhd", merge_weights, k)
    merged_v = torch.einsum("bhafilk,bfklhe->bafilhe", merge_weights, v)
    bias = (merge_weights.float() * merge_logprob).sum(dim=-1, keepdim=True)

    routing_logits = torch.einsum("bafilhd,baijhd->bhafilj", merged_k, q) - bias
    if compressed_valid is not None:
        routing_valid = compressed_valid[:, None, None, :, None, :, None]
        routing_logits = routing_logits.masked_fill(~routing_valid, float("-inf"))

    routing_weights = rearrange(routing_logits, "b h a f i l j -> b h a i j (f l)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    routing_weights = rearrange(
        routing_weights,
        "b h a i j (f l) -> b h a f i l j",
        f=k.size(1),
        l=k.size(3),
    )
    return torch.einsum("bhafilj,bafilhe->baijhe", routing_weights, merged_v)


def run_one_token_variant(q, k, v, grid_size, variant, scale, head_chunk):
    if variant.get("kind") == "transpose_c4":
        frames, height, width = [int(x) for x in grid_size]
        chunks = []
        q_blocks = q.reshape(q.size(0), frames, height, width, q.size(2), q.size(3))
        k_blocks = k.reshape(k.size(0), frames, height, width, k.size(2), k.size(3))
        v_blocks = v.reshape(v.size(0), frames, height, width, v.size(2), v.size(3))
        for start in range(0, q_blocks.size(4), head_chunk):
            end = min(start + head_chunk, q_blocks.size(4))
            chunks.append(
                hard_route_transpose_c4_blocks(
                    q_blocks[:, :, :, :, start:end],
                    k_blocks[:, :, :, :, start:end],
                    v_blocks[:, :, :, :, start:end],
                    scale,
                    seed=0,
                )
            )
        return torch.cat(chunks, dim=4).reshape_as(q)

    frames, height, width = [int(x) for x in grid_size]
    h_reduce = int(variant["h_reduce"])
    w_reduce = int(variant["w_reduce"])
    transpose = bool(variant["transpose"])

    qp, _, padded_h, padded_w = pad_to_multiples(q, frames, height, width, h_reduce, w_reduce)
    kp, key_valid, _, _ = pad_to_multiples(k, frames, height, width, h_reduce, w_reduce)
    vp, _, _, _ = pad_to_multiples(v, frames, height, width, h_reduce, w_reduce)

    q_blocks = video_to_reduced_blocks(qp, 1, h_reduce, w_reduce, padded_h, padded_w)
    k_blocks = video_to_reduced_blocks(kp, 1, h_reduce, w_reduce, padded_h, padded_w)
    v_blocks = video_to_reduced_blocks(vp, 1, h_reduce, w_reduce, padded_h, padded_w)
    valid_blocks = None
    if key_valid is not None:
        valid_blocks = video_to_reduced_blocks(key_valid, 1, h_reduce, w_reduce, padded_h, padded_w)[
            ..., 0, 0
        ]

    chunks = []
    for start in range(0, q_blocks.size(4), head_chunk):
        end = min(start + head_chunk, q_blocks.size(4))
        if transpose:
            out = gridmerge_one_transpose_blocks(
                q_blocks[:, :, :, :, start:end],
                k_blocks[:, :, :, :, start:end],
                v_blocks[:, :, :, :, start:end],
                scale,
                key_valid=valid_blocks,
            )
        else:
            out = gridmerge_one_blocks(
                q_blocks[:, :, :, :, start:end],
                k_blocks[:, :, :, :, start:end],
                v_blocks[:, :, :, :, start:end],
                scale,
                key_valid=valid_blocks,
            )
        chunks.append(out)

    out_blocks = torch.cat(chunks, dim=4)
    out = reduced_blocks_to_video(out_blocks, 1, h_reduce, w_reduce, padded_h, padded_w)
    return crop_from_padded(out, frames, height, width, padded_h, padded_w)


def variant_specs(frames, height, width):
    dense_tokens = frames * height * width
    specs = {
        "monarchrt_h2": {
            "h_reduce": 2,
            "w_reduce": 1,
            "transpose": False,
            "compressed_tokens_per_query": frames * height,
        },
        "monarchrt_h4": {
            "h_reduce": 4,
            "w_reduce": 1,
            "transpose": False,
            "compressed_tokens_per_query": frames * height,
        },
        "transpose_c1_closest": {
            "h_reduce": 1,
            "w_reduce": 1,
            "transpose": True,
            "compressed_tokens_per_query": frames * width,
        },
        "transpose_c4": {
            "h_reduce": 1,
            "w_reduce": 1,
            "transpose": True,
            "kind": "transpose_c4",
            "num_merge_tokens": 4,
            "seed_mode": "random_with_ith",
            "empty_seed_policy": "force_key0",
            "compressed_tokens_per_query": frames * width * 4,
        },
    }
    for spec in specs.values():
        spec["density"] = spec["compressed_tokens_per_query"] / dense_tokens
        spec["sparsity"] = 1.0 - spec["density"]
    return specs


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

    state = {"calls_total": 0, "calls_current_prompt": 0, "warmed": False, "grid_size": None}

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

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
        scale = dim ** -0.5

        state["grid_size"] = [int(value) for value in grid_size]
        specs = variant_specs(*state["grid_size"])
        should_record = args.max_calls <= 0 or state["calls_current_prompt"] < args.max_calls
        state["calls_current_prompt"] += 1
        state["calls_total"] += 1

        if should_record and not state["warmed"]:
            _ = run_dense(q, k, v, self)
            for spec in specs.values():
                _ = run_one_token_variant(q, k, v, grid_size, spec, scale, args.head_chunk)
            sync()
            state["warmed"] = True

        if should_record:
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)
            for name, spec in specs.items():
                out, seconds = timed(
                    lambda spec=spec: run_one_token_variant(
                        q, k, v, grid_size, spec, scale, args.head_chunk
                    )
                )
                stats[name].add_time(seconds)
                stats[name].add_error(out, dense)
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
    parser = argparse.ArgumentParser(
        description="Compare old one-token MonarchRT h_reduce variants against closest transpose sparsity."
    )
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=8)
    parser.add_argument("--head_chunk", type=int, default=1)
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
            variants[name].update(
                {
                    "h_reduce": specs[name]["h_reduce"],
                    "w_reduce": specs[name]["w_reduce"],
                    "transpose": specs[name]["transpose"],
                    "compressed_tokens_per_query": specs[name]["compressed_tokens_per_query"],
                    "density": specs[name]["density"],
                    "sparsity": specs[name]["sparsity"],
                }
            )
            for field in ("num_merge_tokens", "seed_mode", "empty_seed_policy"):
                if field in specs[name]:
                    variants[name][field] = specs[name][field]

    target_density = specs["monarchrt_h2"]["density"]
    transpose_min_density = specs["transpose_c1_closest"]["density"]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_path": args.prompt_path,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "head_chunk": args.head_chunk,
        "grid_size": [frames, height, width],
        "dense_tokens": frames * height * width,
        "wall_seconds": wall_seconds,
        "sparsity_note": {
            "exact_match_possible": False,
            "reason": "MonarchRT h_reduce=2/4 one-token density is 1/width, while transpose one-token minimum density is 1/height. For 480p Wan this is 1/52 vs 1/30, so transpose cannot be made equally sparse without changing the algorithm.",
            "target_monarchrt_density": target_density,
            "closest_transpose_density": transpose_min_density,
            "closest_transpose_is_this_many_times_denser": transpose_min_density / target_density,
        },
        "variants": variants,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
