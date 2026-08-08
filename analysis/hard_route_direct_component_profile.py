import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from attention.common import blocks_to_video, video_to_blocks
from attention.hard_route import _build_seed_rows, hard_route_compressed_attention
from attention.hard_route_backend import (
    direct_compressed_attention,
    direct_indirect_value_attention,
    direct_recompute_value_attention,
    direct_seed_key_attention,
    direct_top_key_attention,
    direct_weight_kv_attention,
    direct_weight_value_attention,
    merge_kv_direct_c4,
    parse_direct_route_backend,
    uses_inline_local_seed_backend,
)
from analysis.hard_route_nonshared_backend_real_qkv import (
    estimate_direct_intermediate_bytes,
    exact_variants,
    parse_variant_names,
)
from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


class ComponentStats:
    def __init__(self):
        self.attention_calls = 0
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0
        self.max_abs = 0.0

    def add_time(self, name, seconds):
        self.seconds[name] += float(seconds)
        self.counts[name] += 1

    def add_error(self, out, ref):
        out_f = out.float().reshape(-1)
        ref_f = ref.float().reshape(-1)
        diff = out_f - ref_f
        self.sqerr += float(torch.dot(diff, diff).item())
        self.ref_sq += float(torch.dot(ref_f, ref_f).clamp_min(1e-20).item())
        self.numel += diff.numel()
        self.max_abs = max(self.max_abs, float(diff.abs().max().item()))

    def summary(self):
        components = {}
        for name, seconds in sorted(self.seconds.items()):
            count = self.counts[name]
            components[name] = {
                "seconds": seconds,
                "calls": count,
                "mean_ms_per_component_call": 1000.0 * seconds / count if count else 0.0,
                "mean_ms_per_attention_call": (
                    1000.0 * seconds / self.attention_calls if self.attention_calls else 0.0
                ),
            }

        payload = {
            "attention_calls": self.attention_calls,
            "components": components,
            "component_total_ms_per_attention_call": sum(
                value["mean_ms_per_attention_call"] for value in components.values()
            ),
        }
        if self.numel:
            payload["error_vs_dense"] = {
                "mse": self.sqerr / self.numel,
                "rel_l2": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                "max_abs": self.max_abs,
            }
        return payload


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


def selected_variants(args):
    if not args.variants:
        return {
            "profiled_backend": {
                "hard_route_backend": args.hard_route_backend,
                "hard_route_routing_backend": args.hard_route_routing_backend,
                "seed_mode": args.seed_mode,
            }
        }

    available = exact_variants()
    names = parse_variant_names(args.variants)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"unknown variants: {', '.join(missing)}")
    return {name: available[name] for name in names}


def make_seed_rows(q_chunk, k_blocks, query_outer_offset, args, variant):
    if uses_inline_local_seed_backend(variant["hard_route_backend"]):
        return None
    return _build_seed_rows(
        q_chunk,
        k_blocks,
        query_outer_offset,
        args.num_merge_tokens,
        args.seed,
        variant.get("seed_mode", args.seed_mode),
    )


def route_compressed_tokens(
    q_chunk,
    source_key,
    source_value,
    merged_key,
    merged_value,
    routing_bias,
    key_scale,
    variant,
    config,
    scale,
    output,
):
    """Run full compressed attention after direct C4 merge.

    Direct Triton routing consumes token-major [K, group, D] intermediates.
    Generic routing backends such as SDPA consume group-major [group, K, D].
    Both paths compute the same full compressed attention.
    """
    routing_backend = variant["hard_route_routing_backend"]
    if config.use_seed_key:
        routed = direct_seed_key_attention(
            q_chunk,
            merged_value,
            routing_bias,
            key_scale,
            routing_backend,
            scale,
            output=output,
            assume_contiguous=True,
        )
        if routed is None:
            raise ValueError("seed-key profiling requires a triton_direct_*_tokenmajor routing backend.")
        return routed

    uses_indirect_key = config.use_indirect_top_key or config.use_indirect_top_kv
    uses_original_value = (
        config.use_indirect_top_kv
        or config.use_indirect_top3_seed_value
        or config.use_indirect_top3_value
        or config.use_indirect_top2_seed_value
        or config.use_indirect_top2_value
        or config.use_indirect_top_value
        or config.use_indirect_seed_value
    )
    if uses_indirect_key:
        routed = direct_top_key_attention(
            q_chunk,
            source_key,
            source_value if uses_original_value else None,
            merged_key,
            merged_value,
            routing_bias,
            key_scale,
            routing_backend,
            scale,
            indirect_seed_value=config.use_indirect_seed_value,
            output=output,
            assume_contiguous=True,
        )
        if routed is None:
            raise ValueError("indirect top-key profiling requires a triton_direct_*_tokenmajor routing backend.")
        return routed

    if (
        config.use_indirect_top3_seed_value
        or config.use_indirect_top3_value
        or config.use_indirect_top2_seed_value
        or config.use_indirect_top2_value
        or config.use_indirect_top_value
        or config.use_indirect_seed_value
    ):
        routed = direct_indirect_value_attention(
            q_chunk,
            merged_key,
            source_value,
            merged_value,
            key_scale
            if (
                config.use_indirect_top3_seed_value
                or config.use_indirect_top3_value
                or config.use_indirect_top2_value
                or config.use_indirect_top2_seed_value
            )
            else None,
            routing_bias,
            routing_backend,
            scale,
            indirect_seed_value=config.use_indirect_seed_value,
            indirect_top3_seed_value=config.use_indirect_top3_seed_value,
            indirect_top3_value=config.use_indirect_top3_value,
            indirect_top2_value=config.use_indirect_top2_value,
            indirect_top2_seed_value=config.use_indirect_top2_seed_value,
            output=output,
            assume_contiguous=True,
        )
        if routed is None:
            raise ValueError("indirect-value profiling requires a triton_direct_*_tokenmajor routing backend.")
        return routed

    if config.use_recompute_value:
        routed = direct_recompute_value_attention(
            q_chunk,
            merged_key,
            source_key,
            source_value,
            routing_bias,
            routing_backend,
            scale,
            output=output,
            assume_contiguous=True,
        )
        if routed is None:
            raise ValueError("recompute-value profiling requires a triton_direct_*_tokenmajor routing backend.")
        return routed

    if config.use_merge_weight_kv:
        routed = direct_weight_kv_attention(
            q_chunk,
            source_key,
            source_value,
            key_scale,
            routing_bias,
            routing_backend,
            scale,
            output=output,
            assume_contiguous=True,
        )
        if routed is None:
            raise ValueError("merge-weight K/V profiling requires a triton_direct_*_tokenmajor routing backend.")
        return routed

    if config.use_merge_weight_value:
        routed = direct_weight_value_attention(
            q_chunk,
            merged_key,
            source_value,
            key_scale,
            routing_bias,
            routing_backend,
            scale,
            output=output,
            assume_contiguous=True,
        )
        if routed is None:
            raise ValueError("merge-weight value profiling requires a triton_direct_*_tokenmajor routing backend.")
        return routed

    routed = direct_compressed_attention(
        q_chunk,
        merged_key,
        merged_value,
        routing_bias,
        routing_backend,
        scale,
        output=output,
        assume_contiguous=True,
    )
    if routed is not None:
        return routed

    query = q_chunk.permute(0, 4, 1, 3, 2, 5).reshape(-1, q_chunk.size(2), q_chunk.size(5))
    out = hard_route_compressed_attention(
        query,
        merged_key.transpose(0, 1),
        merged_value.transpose(0, 1),
        routing_bias.transpose(0, 1),
        routing_backend,
        scale,
    )
    return out.reshape(
        q_chunk.size(0),
        q_chunk.size(4),
        q_chunk.size(1),
        q_chunk.size(3),
        q_chunk.size(2),
        q_chunk.size(5),
    ).permute(0, 2, 4, 3, 1, 5)


def install_probe(stats, args):
    import wan.modules.model as wan_model

    variants = selected_variants(args)
    direct_merge_configs = {
        name: parse_direct_route_backend(variant["hard_route_backend"])
        for name, variant in variants.items()
    }
    state = {
        "calls_current_prompt": 0,
        "warmed": False,
        "dense_calls": 0,
        "dense_seconds": 0.0,
        "intermediate_estimates": {},
    }

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def run_split(q, k, v, grid_size, scale, variant_name, variant, record):
        variant_stats = stats[variant_name]
        direct_merge_config = direct_merge_configs[variant_name]
        _, height, width = grid_size
        if record:
            q_blocks, seconds = timed(lambda: video_to_blocks(q, 1, 1, 1, height, width))
            variant_stats.add_time("video_to_q_blocks", seconds)
            k_blocks, seconds = timed(lambda: video_to_blocks(k, 1, 1, 1, height, width))
            variant_stats.add_time("video_to_k_blocks", seconds)
            v_blocks, seconds = timed(lambda: video_to_blocks(v, 1, 1, 1, height, width))
            variant_stats.add_time("video_to_v_blocks", seconds)
        else:
            q_blocks = video_to_blocks(q, 1, 1, 1, height, width)
            k_blocks = video_to_blocks(k, 1, 1, 1, height, width)
            v_blocks = video_to_blocks(v, 1, 1, 1, height, width)

        k_merge_blocks = k_blocks
        v_merge_blocks = v_blocks

        out_blocks = torch.empty_like(q_blocks)
        for start in range(0, q_blocks.size(1), args.query_outer_chunk):
            end = min(start + args.query_outer_chunk, q_blocks.size(1))
            q_chunk = q_blocks[:, start:end]
            out_slice = out_blocks[:, start:end]
            if record and not q_chunk.is_contiguous():
                q_chunk, seconds = timed(q_chunk.contiguous)
                variant_stats.add_time("q_chunk_contiguous", seconds)
            else:
                q_chunk = q_chunk.contiguous()

            if record:
                seed_rows, seconds = timed(lambda: make_seed_rows(q_chunk, k_blocks, start, args, variant))
                variant_stats.add_time("seed_rows", seconds)
                merge_out, seconds = timed(
                    lambda: merge_kv_direct_c4(
                        q_chunk,
                        seed_rows,
                        k_merge_blocks,
                        v_merge_blocks,
                        direct_merge_config,
                        scale,
                        assume_contiguous=True,
                    )
                )
                variant_stats.add_time("direct_merge", seconds)
            else:
                seed_rows = make_seed_rows(q_chunk, k_blocks, start, args, variant)
                merge_out = merge_kv_direct_c4(
                    q_chunk,
                    seed_rows,
                    k_merge_blocks,
                    v_merge_blocks,
                    direct_merge_config,
                    scale,
                    assume_contiguous=True,
                )

            if merge_out is None:
                raise RuntimeError("hard_route_backend must be a direct-route backend.")
            merged_key, merged_value, routing_bias, key_scale = merge_out
            output = out_slice if out_slice.is_contiguous() else None

            if record:
                routed, seconds = timed(
                    lambda: route_compressed_tokens(
                        q_chunk,
                        k_merge_blocks,
                        v_merge_blocks,
                        merged_key,
                        merged_value,
                        routing_bias,
                        key_scale,
                        variant,
                        direct_merge_config,
                        scale,
                        output,
                    )
                )
                variant_stats.add_time("direct_routing", seconds)
                if routed is not out_slice:
                    _, seconds = timed(lambda: out_slice.copy_(routed))
                    variant_stats.add_time("output_copy", seconds)
            else:
                routed = route_compressed_tokens(
                    q_chunk,
                    k_merge_blocks,
                    v_merge_blocks,
                    merged_key,
                    merged_value,
                    routing_bias,
                    key_scale,
                    variant,
                    direct_merge_config,
                    scale,
                    output,
                )
                if routed is not out_slice:
                    out_slice.copy_(routed)

        if record:
            out, seconds = timed(lambda: blocks_to_video(out_blocks, 1, 1, 1))
            variant_stats.add_time("blocks_to_video", seconds)
            return out
        return blocks_to_video(out_blocks, 1, 1, 1)

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
        if not state["intermediate_estimates"]:
            for name, variant in variants.items():
                state["intermediate_estimates"][name] = estimate_direct_intermediate_bytes(
                    q,
                    grid_size,
                    args.query_outer_chunk,
                    variant,
                )

        should_record = args.max_calls <= 0 or state["calls_current_prompt"] < args.max_calls
        state["calls_current_prompt"] += 1

        if should_record and not state["warmed"]:
            _ = run_dense(q, k, v, self)
            for name, variant in variants.items():
                _ = run_split(q, k, v, grid_size, dim**-0.5, name, variant, record=False)
            sync()
            state["warmed"] = True

        if should_record:
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            state["dense_calls"] += 1
            state["dense_seconds"] += float(seconds)
            for name, variant in variants.items():
                stats[name].attention_calls += 1
                out = run_split(q, k, v, grid_size, dim**-0.5, name, variant, record=True)
                stats[name].add_error(out, dense)
            x = dense
        else:
            x = run_dense(q, k, v, self)
        return self.o(x.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention
    return state, variants


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
    parser = argparse.ArgumentParser(description="Profile direct hard-route components.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=20)
    parser.add_argument("--query_outer_chunk", type=int, default=10)
    parser.add_argument("--num_merge_tokens", type=int, default=4)
    parser.add_argument("--seed_mode", default="fast_random_with_ith")
    parser.add_argument(
        "--variants",
        default="",
        help="Comma- or whitespace-separated exact variant names from hard_route_nonshared_backend_real_qkv.py.",
    )
    parser.add_argument(
        "--hard_route_backend",
        default="triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_bestlogit_tokenmajor_inline_local_seed",
    )
    parser.add_argument(
        "--hard_route_routing_backend",
        default="triton_direct_m32_n16_w1_s1_exp2_tokenmajor",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")
    if args.num_prompts < 1:
        raise ValueError("--num_prompts must be at least 1.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    if len(prompts) != args.num_prompts:
        raise ValueError(f"Requested {args.num_prompts} prompts, found {len(prompts)}.")

    stats = defaultdict(ComponentStats)
    probe_state, variants = install_probe(stats, args)

    sync()
    start = time.perf_counter()
    for prompt_offset, prompt in enumerate(prompts):
        probe_state["calls_current_prompt"] = 0
        probe_state["warmed"] = False
        run_denoise_only(pipeline, config, prompt, args.seed + prompt_offset, dtype)
    sync()

    variant_payload = {}
    for name in sorted(variants):
        variant_payload[name] = {
            "config": {
                key: value
                for key, value in variants[name].items()
                if key in {"hard_route_backend", "hard_route_routing_backend", "seed_mode"}
            },
            "summary": stats[name].summary(),
        }
        estimate = probe_state["intermediate_estimates"].get(name)
        if estimate is not None:
            variant_payload[name]["direct_intermediate_estimate"] = estimate

    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls_per_prompt": args.max_calls,
        "query_outer_chunk": args.query_outer_chunk,
        "variants_requested": args.variants,
        "dense": {
            "calls": probe_state["dense_calls"],
            "seconds": probe_state["dense_seconds"],
            "mean_ms": (
                1000.0 * probe_state["dense_seconds"] / probe_state["dense_calls"]
                if probe_state["dense_calls"]
                else 0.0
            ),
        },
        "wall_seconds": time.perf_counter() - start,
        "variants": variant_payload,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
