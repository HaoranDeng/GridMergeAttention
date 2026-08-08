import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from attention.hard_route_backend import parse_direct_route_backend
from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


DIRECT_MERGE_TEMPLATE = "triton_c4_direct_route_j{block_j}_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot"
DIRECT_ROUTING_BACKEND = "triton_direct_m32_n16_w1_s1_exp2_tokenmajor"
CHUNK_VARIANT_RE = re.compile(r"^(?P<base>.+)_chunk(?P<chunk>\d+)$")


def direct_routing_backend(block_m=32, block_n=16, num_warps=1, num_stages=1):
    return (
        f"triton_direct_m{int(block_m)}_n{int(block_n)}_w{int(num_warps)}"
        f"_s{int(num_stages)}_exp2_tokenmajor"
    )


def direct_merge_backend(*suffixes, block_j=32, block_l=None):
    tile_suffixes = [*suffixes]
    if block_l is not None:
        tile_suffixes.append(f"l{int(block_l)}")
    suffix = "_".join([*tile_suffixes, "tokenmajor", "inline_local_seed"])
    return f"{DIRECT_MERGE_TEMPLATE.format(block_j=int(block_j))}_{suffix}"


def exact_backend(hard_route_backend, seed_mode=None, routing_backend=None):
    config = {
        "hard_route_backend": hard_route_backend,
        "hard_route_routing_backend": routing_backend or DIRECT_ROUTING_BACKEND,
    }
    if seed_mode is not None:
        config["seed_mode"] = seed_mode
    return config


def _dtype_bytes(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype for estimate: {dtype}")


def estimate_direct_intermediate_bytes(q, grid_size, query_outer_chunk, variant):
    """Estimate compressed K/V/bias traffic for one self-attention call."""
    merge_config = parse_direct_route_backend(variant["hard_route_backend"])
    if not merge_config.supported:
        return None

    batch, _, heads, dim = q.shape
    frames, height, width = grid_size
    route_tokens = int(frames) * int(height) * 4
    materialized_key_bytes = _dtype_bytes(
        torch.bfloat16 if merge_config.use_bf16_merged_key else torch.float16
    )
    key_bytes = 0 if merge_config.use_seed_key else materialized_key_bytes
    value_bytes = _dtype_bytes(torch.bfloat16 if merge_config.use_bf16_merged_value else torch.float16)
    bias_bytes = _dtype_bytes(torch.bfloat16 if merge_config.use_bf16_routing_bias else torch.float16)

    chunk_count = 0
    total_bytes = 0
    max_chunk_bytes = 0
    seed_key_query_read_bytes = 0
    for start in range(0, int(frames), int(query_outer_chunk)):
        chunk_outer = min(int(query_outer_chunk), int(frames) - start)
        route_groups = int(batch) * int(heads) * chunk_outer * int(width)
        chunk_bytes = route_tokens * route_groups * (dim * key_bytes + dim * value_bytes + bias_bytes)
        total_bytes += chunk_bytes
        max_chunk_bytes = max(max_chunk_bytes, chunk_bytes)
        if merge_config.use_seed_key:
            seed_key_query_read_bytes += int(height) * 4 * route_groups * dim * materialized_key_bytes
        chunk_count += 1

    return {
        "route_tokens": route_tokens,
        "query_chunks": chunk_count,
        "max_chunk_intermediate_bytes": max_chunk_bytes,
        "intermediate_write_bytes_per_attention_call": total_bytes,
        "intermediate_read_bytes_per_attention_call": total_bytes,
        "seed_key_query_read_bytes_per_attention_call": seed_key_query_read_bytes,
        "key_dtype": "bf16" if merge_config.use_bf16_merged_key else "fp16",
        "value_dtype": "bf16" if merge_config.use_bf16_merged_value else "fp16",
        "bias_dtype": "bf16" if merge_config.use_bf16_routing_bias else "fp16",
        "block_j_target": merge_config.block_j_target,
        "block_l_target": merge_config.block_l_target or None,
        "writes_merged_key": not merge_config.use_seed_key,
    }


class Stats:
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0
        self.sqerr_dense = 0.0
        self.ref_sq_dense = 0.0
        self.sqerr_baseline = 0.0
        self.ref_sq_baseline = 0.0
        self.numel = 0
        self.max_abs_vs_dense = 0.0
        self.max_abs_vs_baseline = 0.0

    def add_time(self, seconds):
        self.calls += 1
        self.seconds += float(seconds)

    def add_error(self, out, dense, baseline=None):
        out_f = out.float().reshape(-1)
        dense_f = dense.float().reshape(-1)
        diff_dense = out_f - dense_f
        self.sqerr_dense += float(torch.dot(diff_dense, diff_dense).item())
        self.ref_sq_dense += float(torch.dot(dense_f, dense_f).clamp_min(1e-20).item())
        self.max_abs_vs_dense = max(self.max_abs_vs_dense, float(diff_dense.abs().max().item()))
        self.numel += diff_dense.numel()

        if baseline is not None:
            baseline_f = baseline.float().reshape(-1)
            diff_baseline = out_f - baseline_f
            self.sqerr_baseline += float(torch.dot(diff_baseline, diff_baseline).item())
            self.ref_sq_baseline += float(torch.dot(baseline_f, baseline_f).clamp_min(1e-20).item())
            self.max_abs_vs_baseline = max(
                self.max_abs_vs_baseline,
                float(diff_baseline.abs().max().item()),
            )

    def summary(self, dense_seconds=None, baseline_seconds=None):
        out = {
            "calls": self.calls,
            "seconds": self.seconds,
            "mean_ms": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
        }
        if dense_seconds is not None and self.seconds:
            out["speedup_vs_dense_attention"] = dense_seconds / self.seconds
        if baseline_seconds is not None and self.seconds:
            out["speedup_vs_hard_c4_baseline"] = baseline_seconds / self.seconds
        if self.numel:
            out.update(
                {
                    "mse_vs_dense": self.sqerr_dense / self.numel,
                    "rel_l2_vs_dense": (self.sqerr_dense / max(self.ref_sq_dense, 1e-20)) ** 0.5,
                    "max_abs_vs_dense": self.max_abs_vs_dense,
                }
            )
            if self.ref_sq_baseline:
                out.update(
                    {
                        "mse_vs_hard_c4_baseline": self.sqerr_baseline / self.numel,
                        "rel_l2_vs_hard_c4_baseline": (
                            self.sqerr_baseline / max(self.ref_sq_baseline, 1e-20)
                        )
                        ** 0.5,
                        "max_abs_vs_hard_c4_baseline": self.max_abs_vs_baseline,
                    }
                )
        return out


def exact_variants():
    return {
        "hard_c4_baseline": {
            "hard_route_backend": "triton",
            "hard_route_routing_backend": "triton",
        },
        "hard_c4_full_core_tokenmajor": exact_backend(
            direct_merge_backend(), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f1_tokenmajor": exact_backend(
            direct_merge_backend("f1"), seed_mode="local_with_ith"
        ),
        "hard_c4_full_core_tokenmajor_n16_s2": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=16, num_stages=2),
        ),
        "hard_c4_full_core_tokenmajor_n16_s3": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=16, num_stages=3),
        ),
        "hard_c4_full_core_tokenmajor_n32_s2": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=32, num_stages=2),
        ),
        "hard_c4_full_core_tokenmajor_n32_s3": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=32, num_stages=3),
        ),
        "hard_c4_full_core_tokenmajor_m16_n16_w1": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=16, num_warps=1),
        ),
        "hard_c4_full_core_tokenmajor_m16_n32_w1": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=32, num_warps=1),
        ),
        "hard_c4_full_core_tokenmajor_m16_n64_w1": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=64, num_warps=1),
        ),
        "hard_c4_full_core_tokenmajor_m16_n64_w2": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=64, num_warps=2),
        ),
        "hard_c4_full_core_tokenmajor_n64_w1": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=64, num_warps=1),
        ),
        "hard_c4_full_core_tokenmajor_n64_w2": exact_backend(
            direct_merge_backend(),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=64, num_warps=2),
        ),
        "hard_c4_keyouter_f2_tokenmajor": exact_backend(
            direct_merge_backend("f2"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_bf16kv_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit", "bf16kv"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_bf16k_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit", "bf16k"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_bf16v_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit", "bf16v"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_bf16bias_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit", "bf16bias"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_bf16kv_bf16bias_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit", "bf16kv", "bf16bias"),
            seed_mode="local_with_ith",
        ),
        "hard_c4_bestlogit_recompute_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit", "recompute"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_tokenmajor_sdpa": exact_backend(
            direct_merge_backend("bestlogit"),
            seed_mode="local_with_ith",
            routing_backend="sdpa",
        ),
        "hard_c4_bestlogit_tokenmajor_n32_s1": exact_backend(
            direct_merge_backend("bestlogit"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=32),
        ),
        "hard_c4_maskassign_tokenmajor": exact_backend(
            direct_merge_backend("maskassign"), seed_mode="local_with_ith"
        ),
        "hard_c4_bestlogit_maskassign_tokenmajor": exact_backend(
            direct_merge_backend("bestlogit", "maskassign"), seed_mode="local_with_ith"
        ),
        "hard_c4_stripe_tokenmajor": exact_backend(
            direct_merge_backend("stripe"), seed_mode="local_with_ith"
        ),
        "hard_c4_stripe_tokenmajor_n32_s1": exact_backend(
            direct_merge_backend("stripe"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=32),
        ),
        "hard_c4_seedkey_tokenmajor": exact_backend(
            direct_merge_backend("seedkey", "bestlogit"), seed_mode="local_with_ith"
        ),
        "hard_c4_seedkey_tokenmajor_n32_s1": exact_backend(
            direct_merge_backend("seedkey", "bestlogit"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=32),
        ),
        "hard_c4_seedkey_tokenmajor_n64_w1": exact_backend(
            direct_merge_backend("seedkey", "bestlogit"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=64, num_warps=1),
        ),
        "hard_c4_bf16kv_tokenmajor": exact_backend(
            direct_merge_backend("bf16kv"), seed_mode="local_with_ith"
        ),
        "hard_c4_bf16k_tokenmajor": exact_backend(
            direct_merge_backend("bf16k"), seed_mode="local_with_ith"
        ),
        "hard_c4_bf16v_tokenmajor": exact_backend(
            direct_merge_backend("bf16v"), seed_mode="local_with_ith"
        ),
        "hard_c4_bf16bias_tokenmajor": exact_backend(
            direct_merge_backend("bf16bias"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f2_bestlogit_tokenmajor": exact_backend(
            direct_merge_backend("f2", "bestlogit"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f3_tokenmajor": exact_backend(
            direct_merge_backend("f3"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f3_bestlogit_tokenmajor": exact_backend(
            direct_merge_backend("f3", "bestlogit"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f4_tokenmajor": exact_backend(
            direct_merge_backend("f4"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f4_bestlogit_tokenmajor": exact_backend(
            direct_merge_backend("f4", "bestlogit"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f4_bf16kv_tokenmajor": exact_backend(
            direct_merge_backend("f4", "bf16kv"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f5_tokenmajor": exact_backend(
            direct_merge_backend("f5"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f5_bestlogit_tokenmajor": exact_backend(
            direct_merge_backend("f5", "bestlogit"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f6_tokenmajor": exact_backend(
            direct_merge_backend("f6"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f6_tokenmajor_n16_s2": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=16, num_stages=2),
        ),
        "hard_c4_keyouter_f6_tokenmajor_n16_s3": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=16, num_stages=3),
        ),
        "hard_c4_keyouter_f6_tokenmajor_n32_s2": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=32, num_stages=2),
        ),
        "hard_c4_keyouter_f6_tokenmajor_n32_s3": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=32, num_stages=3),
        ),
        "hard_c4_keyouter_f6_tokenmajor_m16_n16_w1": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=16, num_warps=1),
        ),
        "hard_c4_keyouter_f6_tokenmajor_m16_n32_w1": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=32, num_warps=1),
        ),
        "hard_c4_keyouter_f6_tokenmajor_m16_n64_w1": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=64, num_warps=1),
        ),
        "hard_c4_keyouter_f6_tokenmajor_m16_n64_w2": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_m=16, block_n=64, num_warps=2),
        ),
        "hard_c4_keyouter_f6_tokenmajor_n64_w1": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=64, num_warps=1),
        ),
        "hard_c4_keyouter_f6_tokenmajor_n64_w2": exact_backend(
            direct_merge_backend("f6"),
            seed_mode="local_with_ith",
            routing_backend=direct_routing_backend(block_n=64, num_warps=2),
        ),
        "hard_c4_keyouter_f6_bestlogit_tokenmajor": exact_backend(
            direct_merge_backend("f6", "bestlogit"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f6_bf16kv_tokenmajor": exact_backend(
            direct_merge_backend("f6", "bf16kv"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f6_bf16k_tokenmajor": exact_backend(
            direct_merge_backend("f6", "bf16k"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f6_bf16v_tokenmajor": exact_backend(
            direct_merge_backend("f6", "bf16v"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f6_bf16bias_tokenmajor": exact_backend(
            direct_merge_backend("f6", "bf16bias"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f7_tokenmajor": exact_backend(
            direct_merge_backend("f7"), seed_mode="local_with_ith"
        ),
        "hard_c4_keyouter_f7_bestlogit_tokenmajor": exact_backend(
            direct_merge_backend("f7", "bestlogit"), seed_mode="local_with_ith"
        ),
    }


def resolve_variant(name, available):
    """Resolve a named benchmark variant.

    Any existing variant can be suffixed with `_chunkN` to benchmark the same
    attention backend with a different query_outer_chunk in the same run.
    """
    if name in available:
        return dict(available[name])

    match = CHUNK_VARIANT_RE.fullmatch(name)
    if match is None:
        return None

    base_name = match.group("base")
    if base_name not in available:
        return None

    variant = dict(available[base_name])
    variant["query_outer_chunk"] = int(match.group("chunk"))
    return variant


def parse_variant_names(text):
    """Accept comma-separated or whitespace-separated variant lists."""
    return [name for name in (text or "").replace(",", " ").split() if name]


def load_smoke_variant_filter(path):
    """Return smoke-derived skip reasons and optional partial-smoke metadata.

    A complete smoke JSON only filters variants that explicitly recorded an
    error. An incomplete smoke JSON may have been left behind by a hard kernel
    failure. In that case, only variants that were part of the smoke run and
    did not finish successfully are skipped. Variants outside the smoke list
    are allowed to run and handle their own warmup errors.
    """
    if not path:
        return {}, None, None
    smoke_path = Path(path)
    if not smoke_path.exists():
        return {}, None, None
    payload = json.loads(smoke_path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    intended = set(payload.get("variant_names") or [])
    errors = {
        name: str(result["error"])
        for name, result in results.items()
        if isinstance(result, dict) and result.get("error")
    }
    if payload.get("complete", True):
        return errors, None, intended or None

    succeeded = {
        name
        for name, result in results.items()
        if isinstance(result, dict) and not result.get("error")
    }
    return errors, succeeded, intended or None


def smoke_skip_reason(name, smoke_variant_errors, smoke_success_names, smoke_intended_names):
    if name in smoke_variant_errors:
        return smoke_variant_errors[name]
    if (
        smoke_success_names is not None
        and (smoke_intended_names is None or name in smoke_intended_names)
        and name not in smoke_success_names
    ):
        return "missing from incomplete smoke JSON"
    return None


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - start


def is_illegal_memory_error(exc):
    return "illegal memory" in repr(exc).lower()


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

    all_variants = exact_variants()
    selected = parse_variant_names(args.variants)
    smoke_variant_errors, smoke_success_names, smoke_intended_names = load_smoke_variant_filter(
        args.skip_variant_errors_from_json
    )
    skipped_from_smoke = {}
    if selected:
        variants = {}
        missing = []
        for name in selected:
            variant = resolve_variant(name, all_variants)
            if variant is None:
                missing.append(name)
            elif reason := smoke_skip_reason(
                name,
                smoke_variant_errors,
                smoke_success_names,
                smoke_intended_names,
            ):
                skipped_from_smoke[name] = reason
            else:
                variants[name] = variant
        if missing:
            raise ValueError(f"unknown variants: {', '.join(missing)}")
    else:
        variants = {}
        for name, variant in all_variants.items():
            if reason := smoke_skip_reason(
                name,
                smoke_variant_errors,
                smoke_success_names,
                smoke_intended_names,
            ):
                skipped_from_smoke[name] = reason
            else:
                variants[name] = variant
    if not args.skip_baseline and "hard_c4_baseline" not in variants:
        variants = {"hard_c4_baseline": all_variants["hard_c4_baseline"], **variants}

    state = {
        "calls_total": 0,
        "calls_current_prompt": 0,
        "warmed": False,
        "intermediate_estimates": {},
        "disabled_variants": set(),
        "skipped_variants_from_smoke": skipped_from_smoke,
        "variant_errors": {},
    }

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def run_variant(q, k, v, grid_size, kwargs):
        _, height, width = grid_size
        return wan_model.token_merging_attn(
            q,
            k,
            v,
            1,
            1,
            1,
            height,
            width,
            q_init="ith",
            query_outer_chunk=kwargs.get("query_outer_chunk", args.query_outer_chunk),
            layout="hard_route",
            num_merge_tokens=4,
            seed_mode=kwargs.get("seed_mode", "random_with_ith"),
            seed=args.seed,
            hard_route_backend=kwargs["hard_route_backend"],
            hard_route_routing_backend=kwargs["hard_route_routing_backend"],
        )

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
            for name, kwargs in variants.items():
                state["intermediate_estimates"][name] = estimate_direct_intermediate_bytes(
                    q,
                    grid_size,
                    kwargs.get("query_outer_chunk", args.query_outer_chunk),
                    kwargs,
                )

        should_record = args.max_calls <= 0 or state["calls_current_prompt"] < args.max_calls
        state["calls_current_prompt"] += 1
        state["calls_total"] += 1

        if should_record and not state["warmed"]:
            _ = run_dense(q, k, v, self)
            for name, kwargs in variants.items():
                try:
                    print(f"warming variant: {name}", flush=True)
                    _ = run_variant(q, k, v, grid_size, kwargs)
                    sync()
                except Exception as exc:
                    state["disabled_variants"].add(name)
                    state["variant_errors"][name] = repr(exc)
                    if is_illegal_memory_error(exc):
                        raise RuntimeError(f"{name} failed with an illegal memory access during warmup.") from exc
            sync()
            state["warmed"] = True

        if should_record:
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)

            baseline = None
            if not args.skip_baseline:
                baseline, seconds = timed(
                    lambda: run_variant(q, k, v, grid_size, variants["hard_c4_baseline"])
                )
                stats["hard_c4_baseline"].add_time(seconds)
                stats["hard_c4_baseline"].add_error(baseline, dense)

            for name, kwargs in variants.items():
                if name == "hard_c4_baseline":
                    continue
                if name in state["disabled_variants"]:
                    continue
                try:
                    print(f"timing variant: {name}", flush=True)
                    out, seconds = timed(lambda kwargs=kwargs: run_variant(q, k, v, grid_size, kwargs))
                    stats[name].add_time(seconds)
                    stats[name].add_error(out, dense, baseline=baseline)
                except Exception as exc:
                    state["disabled_variants"].add(name)
                    state["variant_errors"][name] = repr(exc)
                    if is_illegal_memory_error(exc):
                        raise RuntimeError(f"{name} failed with an illegal memory access during timing.") from exc
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
        description="Benchmark exact non-shared hard C4 backends on real Wan self-attention QKV."
    )
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=30)
    parser.add_argument("--query_outer_chunk", type=int, default=10)
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip the slow legacy hard C4 baseline when only direct-route variants are needed.",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated subset. Empty runs all exact variants.",
    )
    parser.add_argument(
        "--skip_variant_errors_from_json",
        default="",
        help="Optional smoke JSON. Variants with an error in that JSON are skipped.",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, args.use_ema)
    if args.num_prompts < 1:
        raise ValueError("--num_prompts must be at least 1.")
    prompts = load_prompts(args.prompt_path, args.prompt_index + args.num_prompts)[
        args.prompt_index : args.prompt_index + args.num_prompts
    ]
    if len(prompts) != args.num_prompts:
        raise ValueError(
            f"Requested {args.num_prompts} prompts from index {args.prompt_index}, "
            f"but only found {len(prompts)}."
        )

    stats = defaultdict(Stats)
    probe_state = install_probe(stats, args)

    sync()
    start = time.perf_counter()
    for prompt_offset, prompt in enumerate(prompts):
        probe_state["calls_current_prompt"] = 0
        probe_state["warmed"] = False
        run_denoise_only(pipeline, config, prompt, args.seed + prompt_offset, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    dense_seconds = stats["dense"].seconds
    baseline_stat = stats.get("hard_c4_baseline")
    baseline_seconds = baseline_stat.seconds if baseline_stat is not None and baseline_stat.calls else None
    variant_summaries = {}
    for name, value in sorted(stats.items()):
        summary = value.summary(
            dense_seconds=dense_seconds if name != "dense" else None,
            baseline_seconds=baseline_seconds if baseline_seconds and name != "dense" else None,
        )
        estimate = probe_state["intermediate_estimates"].get(name)
        if estimate is not None:
            summary["direct_intermediate_estimate"] = estimate
        variant_summaries[name] = summary

    payload = {
        "device": torch.cuda.get_device_name(0),
        "config": args.config,
        "dtype": args.dtype,
        "prompt_index": args.prompt_index,
        "num_prompts": args.num_prompts,
        "max_calls": args.max_calls,
        "max_calls_per_prompt": args.max_calls,
        "total_self_attention_calls": probe_state["calls_total"],
        "query_outer_chunk": args.query_outer_chunk,
        "wall_seconds": wall_seconds,
        "variants": variant_summaries,
        "skipped_variants_from_smoke": probe_state["skipped_variants_from_smoke"],
        "variant_errors": probe_state["variant_errors"],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
