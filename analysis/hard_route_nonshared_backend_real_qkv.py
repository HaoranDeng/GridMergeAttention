import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from generate import load_generator_checkpoint, load_prompts
from pipeline import TokenMergingWanPipeline
from utils.misc import set_seed


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
        "hard_c4_triton_c4": {
            "hard_route_backend": "triton_c4",
            "hard_route_routing_backend": "triton",
        },
        "hard_c4_triton_c4_j16": {
            "hard_route_backend": "triton_c4_j16",
            "hard_route_routing_backend": "triton",
        },
        "hard_c4_direct_route_j16_w2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w2",
            "hard_route_routing_backend": "triton",
        },
        "hard_c4_direct_route_j16_w1_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w1",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j16_w2_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w2",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton",
        },
        "hard_c4_direct_route_j32_w1_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w1",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w4_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w4",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j64_w1_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w1",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j64_w2_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w2",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j64_w4_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w4",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n128": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n128",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n128_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n128_s1",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n128_w2_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n128_w2_s1",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n128_w2_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n128_w2_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n256_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n256_s1",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n32_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n32_s1",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n64_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n64_s1",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n64_w1_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n64_w2_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n64_w2_s1",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_bf16bias_route_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_bf16bias",
            "hard_route_routing_backend": "triton_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s1_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s1",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16bias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16bias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j16_w1_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w1_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j16_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j16_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w1_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w1_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w4_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w4_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j64_w1_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w1_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j64_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j64_w4_exp2merge_forcekeys_fp16kvbias_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w4_exp2merge_forcekeys_fp16kvbias_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16all_s1_fastdot_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16all_s1_fastdot",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s2_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s2",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s4_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_forcekeys_bf16bias_s4",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_bf16bias_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_bf16bias",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m16_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m16_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m16_n128_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m16_n128_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m32_n32_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m32_n32_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m32_n64_w2_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m32_n64_w2_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m32_n128_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m32_n128_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m32_n128_w2_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m32_n128_w2_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m64_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m64_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_exp2merge_direct_m64_n128_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge",
            "hard_route_routing_backend": "triton_direct_m64_n128_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_bf16bias_direct_m32_n64_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_bf16bias",
            "hard_route_routing_backend": "triton_direct_m32_n64_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_direct_m32_n128_w1_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_direct_m32_n128_w1_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_route_m32_n64_w2_s1_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m32_n64_w2_s1_exp2",
        },
        "hard_c4_direct_route_j32_w2_route_m64_n64_w2_s1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_m64_n64_w2_s1",
        },
        "hard_c4_triton_c4_route_m128_n32": {
            "hard_route_backend": "triton_c4",
            "hard_route_routing_backend": "triton_m128_n32",
        },
        "hard_c4_triton_c4_route_m128_n32_s1": {
            "hard_route_backend": "triton_c4",
            "hard_route_routing_backend": "triton_m128_n32_s1",
        },
        "hard_c4_triton_c4_route_m256_n32": {
            "hard_route_backend": "triton_c4",
            "hard_route_routing_backend": "triton_m256_n32",
        },
    }


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


def install_probe(stats, args):
    import wan.modules.model as wan_model

    all_variants = exact_variants()
    selected = [name.strip() for name in args.variants.split(",") if name.strip()]
    if selected:
        missing = [name for name in selected if name not in all_variants]
        if missing:
            raise ValueError(f"unknown variants: {', '.join(missing)}")
        variants = {name: all_variants[name] for name in selected}
    else:
        variants = all_variants
    if "hard_c4_baseline" not in variants:
        variants = {"hard_c4_baseline": all_variants["hard_c4_baseline"], **variants}

    state = {"calls_total": 0, "calls_current_prompt": 0, "warmed": False}

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
            query_outer_chunk=args.query_outer_chunk,
            layout="hard_route",
            num_merge_tokens=4,
            seed_mode="random_with_ith",
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

        should_record = args.max_calls <= 0 or state["calls_current_prompt"] < args.max_calls
        state["calls_current_prompt"] += 1
        state["calls_total"] += 1

        if should_record and not state["warmed"]:
            _ = run_dense(q, k, v, self)
            for kwargs in variants.values():
                _ = run_variant(q, k, v, grid_size, kwargs)
            sync()
            state["warmed"] = True

        if should_record:
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)

            baseline, seconds = timed(
                lambda: run_variant(q, k, v, grid_size, variants["hard_c4_baseline"])
            )
            stats["hard_c4_baseline"].add_time(seconds)
            stats["hard_c4_baseline"].add_error(baseline, dense)

            for name, kwargs in variants.items():
                if name == "hard_c4_baseline":
                    continue
                out, seconds = timed(lambda kwargs=kwargs: run_variant(q, k, v, grid_size, kwargs))
                stats[name].add_time(seconds)
                stats[name].add_error(out, dense, baseline=baseline)
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
    parser.add_argument("--query_outer_chunk", type=int, default=7)
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated subset. Empty runs all exact variants.",
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
    baseline_seconds = stats["hard_c4_baseline"].seconds
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
        "variants": {
            name: value.summary(
                dense_seconds=dense_seconds if name != "dense" else None,
                baseline_seconds=baseline_seconds if name not in {"dense", "hard_c4_baseline"} else None,
            )
            for name, value in sorted(stats.items())
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
