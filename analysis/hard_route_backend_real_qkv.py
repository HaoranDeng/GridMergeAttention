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


class BackendStats:
    def __init__(self):
        self.calls = 0
        self.seconds = 0.0
        self.sqerr = 0.0
        self.ref_sq = 0.0
        self.numel = 0
        self.max_abs = 0.0
        self.rel_l2_sum = 0.0
        self.cosine_error_sum = 0.0

    def add_time(self, seconds):
        self.calls += 1
        self.seconds += float(seconds)

    def add_error(self, out, ref):
        diff = (out.float() - ref.float()).reshape(-1)
        ref_flat = ref.float().reshape(-1)
        sqerr = torch.dot(diff, diff)
        ref_sq = torch.dot(ref_flat, ref_flat).clamp_min(1e-20)
        out_sq = torch.dot(out.float().reshape(-1), out.float().reshape(-1)).clamp_min(1e-20)
        dot = torch.dot(out.float().reshape(-1), ref_flat)
        rel_l2 = torch.sqrt(sqerr / ref_sq)
        cosine_error = 1.0 - dot / torch.sqrt(out_sq * ref_sq)

        self.sqerr += float(sqerr.item())
        self.ref_sq += float(ref_sq.item())
        self.numel += diff.numel()
        self.max_abs = max(self.max_abs, float(diff.abs().max().item()))
        self.rel_l2_sum += float(rel_l2.item())
        self.cosine_error_sum += float(cosine_error.item())

    def summary(self, dense_seconds=None):
        payload = {
            "calls": self.calls,
            "seconds": self.seconds,
            "mean_ms": 1000.0 * self.seconds / self.calls if self.calls else 0.0,
        }
        if dense_seconds is not None and self.seconds:
            payload["speedup_vs_dense"] = dense_seconds / self.seconds
        if self.numel:
            payload.update(
                {
                    "mse": self.sqerr / self.numel,
                    "rel_l2": (self.sqerr / max(self.ref_sq, 1e-20)) ** 0.5,
                    "mean_call_rel_l2": self.rel_l2_sum / self.calls,
                    "max_abs": self.max_abs,
                    "cosine_error": self.cosine_error_sum / self.calls,
                }
            )
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


def load_pipeline(config_path, checkpoint, dtype, device, use_ema=False):
    config = OmegaConf.load(config_path)
    pipeline = TokenMergingWanPipeline(config, device=device)
    load_generator_checkpoint(pipeline, checkpoint, use_ema)
    pipeline = pipeline.to(dtype=dtype)
    pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.eval()
    return config, pipeline


def backend_variants(variant_set):
    baseline = {
        "best": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1",
        },
        "exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
        },
        "forcekeys": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1",
        },
        "forcekeys_exp2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
        },
        "forcekeys_exp2merge_exp2route": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
        },
    }
    tuning = {
        "full_c4_cached": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "full_c4_preserve": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "full_c4_preserve_cached": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_cached",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "full_c4_n64_cached": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "full_c4_j64_w2_cached": {
            "hard_route_backend": "triton_c4_direct_route_j64_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "full_c4_j64_w4_cached": {
            "hard_route_backend": "triton_c4_direct_route_j64_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "full_c4_maxprob": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_maxprob_cached",
            "hard_route_routing_backend": "triton_grid_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "full_c4_countbias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_countbias_cached",
            "hard_route_routing_backend": "triton_grid_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r2_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r2_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r2_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r2_exact": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r2_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r3_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r3_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_m64_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r3_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_m64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_n32": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_n128": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m128_n128_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_w2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m128_n64_w2_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_w8": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m128_n64_w8_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_s2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m128_n64_w4_s2_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_m256": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m256_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r3_exact_bf16bias_m256_n128": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r3_m256_n128_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_m64_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_j16_w1": {
            "hard_route_backend": "triton_c4_direct_route_j16_w1_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_j16_w2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w2_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_j32_w1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w1_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_bf16bias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_exp2merge_bf16bias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_j64_w2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w2_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_j64_w4": {
            "hard_route_backend": "triton_c4_direct_route_j64_w4_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_n32": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_w2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_w2_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_w8": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_w8_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_preserve_cached_s2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_w4_s2_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_n32": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_n32_s2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n32_w4_s2_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_w2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_w2_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_w8": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_w8_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_s2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_w4_s2_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_bf16bias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_maxprob": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_maxprob_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_countbias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_countbias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_exact_m64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_n64_j64_w2": {
            "hard_route_backend": "triton_c4_direct_route_j64_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_n64_j64_w4": {
            "hard_route_backend": "triton_c4_direct_route_j64_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_n64_maxprob": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_maxprob_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r4_n64_countbias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_countbias_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r5_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_m64_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r5_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_n32": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_n32_s2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n32_w4_s2_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_w2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_w2_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_w2_s2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_w2_s2_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_w8": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_w8_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_s2": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_w4_s2_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_s3": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_w4_s3_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_s4": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_w4_s4_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_m64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_bf16bias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_maxprob": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_maxprob_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_countbias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_countbias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_j32_w1": {
            "hard_route_backend": "triton_c4_direct_route_j32_w1_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_j16_w1": {
            "hard_route_backend": "triton_c4_direct_route_j16_w1_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_j16_w2": {
            "hard_route_backend": "triton_c4_direct_route_j16_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_n128": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m128_n128_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_m256": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m256_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r5_exact_m256_n128": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r5_m256_n128_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r6_m128_n32_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r6_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6_m64_n64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r6_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6_exact": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r6_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6_exact_m64": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r6_m64_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6_exact_bf16bias": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r6_m128_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6_exact_n128": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r6_m128_n128_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "outer_r6_exact_m256": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r6_m256_n64_s1_exp2",
            "seed_mode": "fast_random_with_ith",
        },
        "nobias_full_c4": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_nobias_cached",
            "hard_route_routing_backend": "triton_grid_m128_n32_s1_exp2_nobias",
            "seed_mode": "fast_random_with_ith",
        },
        "nobias_outer_r4": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_nobias_cached",
            "hard_route_routing_backend": "triton_grid_outer_local_r4_m128_n32_s1_exp2_nobias",
            "seed_mode": "fast_random_with_ith",
        },
        "nobias_outer_r4_exact": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_nobias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n64_s1_exp2_nobias",
            "seed_mode": "fast_random_with_ith",
        },
        "nobias_outer_r4_exact_n32": {
            "hard_route_backend": "triton_c4_direct_route_j32_w2_forcekeys_exp2merge_nobias_cached",
            "hard_route_routing_backend": "triton_grid_outer_exact_r4_m128_n32_s1_exp2_nobias",
            "seed_mode": "fast_random_with_ith",
        },
    }
    if variant_set == "baseline":
        return baseline
    if variant_set == "tuning":
        return tuning
    raise ValueError(f"unknown variant_set={variant_set!r}")


def install_backend_probe(stats, max_calls, variant_set, selected_variants=None):
    import wan.modules.model as wan_model

    variants = backend_variants(variant_set)
    if selected_variants:
        missing = [name for name in selected_variants if name not in variants]
        if missing:
            raise ValueError(f"unknown variants for {variant_set}: {', '.join(missing)}")
        variants = {name: variants[name] for name in selected_variants}
    call_index = {"value": 0}
    warmed = {"value": False}

    def run_dense(q, k, v, self):
        return wan_model.flash_attention(q=q, k=k, v=v, window_size=self.window_size)

    def run_variant(q, k, v, grid_size, kwargs):
        f, h, w = grid_size
        return wan_model.token_merging_attn(
            q,
            k,
            v,
            1,
            1,
            1,
            h,
            w,
            q_init="ith",
            layout="hard_route_shared",
            num_merge_tokens=4,
            seed_mode=kwargs.get("seed_mode", "random_with_ith"),
            seed=0,
            hard_route_backend=kwargs["hard_route_backend"],
            hard_route_routing_backend=kwargs["hard_route_routing_backend"],
        )

    def probed_self_attention(self, x, grid_size, rope_cache):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if torch.is_grad_enabled() or x.requires_grad:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
        else:
            q = wan_model.rmsnorm(self.q(x), self.norm_q.weight, eps=self.norm_q.eps).view(b, s, n, d)
            k = wan_model.rmsnorm(self.k(x), self.norm_k.weight, eps=self.norm_k.eps).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q, k = wan_model.apply_rope_video_tokens(q, k, rope_cache)

        should_record = max_calls <= 0 or call_index["value"] < max_calls
        call_index["value"] += 1

        if should_record and not warmed["value"]:
            _ = run_dense(q, k, v, self)
            for kwargs in variants.values():
                _ = run_variant(q, k, v, grid_size, kwargs)
            sync()
            warmed["value"] = True

        if should_record:
            dense, seconds = timed(lambda: run_dense(q, k, v, self))
            stats["dense"].add_time(seconds)
            for name, kwargs in variants.items():
                out, seconds = timed(lambda kwargs=kwargs: run_variant(q, k, v, grid_size, kwargs))
                stats[name].add_time(seconds)
                stats[name].add_error(out, dense)
            x = dense
        else:
            x = run_dense(q, k, v, self)

        return self.o(x.flatten(2))

    wan_model.WanSelfAttention.forward = probed_self_attention


@torch.inference_mode()
def run_denoise_only(pipeline, config, prompt, seed, dtype):
    device = torch.device("cuda")
    set_seed(seed)
    latent_shape = list(config.image_or_video_shape)
    noise = torch.randn(latent_shape, device=device, dtype=dtype)
    text_prompts = [prompt] * latent_shape[0]
    conditional_dict = pipeline.text_encoder(text_prompts=text_prompts)

    noisy_image_or_video = noise
    denoise_steps = int(len(pipeline.denoising_step_list) - 1)
    for index, current_timestep in enumerate(pipeline.denoising_step_list[:-1]):
        _, pred_image_or_video = pipeline.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=torch.ones(noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep,
        )
        next_timestep = pipeline.denoising_step_list[index + 1] * torch.ones(
            noise.shape[:2],
            dtype=torch.long,
            device=noise.device,
        )
        noisy_image_or_video = pipeline.scheduler.add_noise(
            pred_image_or_video.flatten(0, 1),
            torch.randn_like(pred_image_or_video.flatten(0, 1)),
            next_timestep.flatten(0, 1),
        ).unflatten(0, noise.shape[:2])

    return denoise_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark hard-route backends on real Wan self-attention QKV.")
    parser.add_argument("--config", default="configs/dense_wan_fewstep.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_path", default="prompts/MovieGenVideoBench_extended.txt")
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_calls", type=int, default=0)
    parser.add_argument("--variant_set", choices=["baseline", "tuning"], default="baseline")
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated subset of variants from --variant_set. Empty runs the full set.",
    )
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda")
    config, pipeline = load_pipeline(args.config, args.checkpoint, dtype, device, use_ema=args.use_ema)
    prompts = load_prompts(args.prompt_path, args.prompt_index + 1)
    prompt = prompts[args.prompt_index]

    stats = defaultdict(BackendStats)
    selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    install_backend_probe(stats, args.max_calls, args.variant_set, selected_variants or None)

    sync()
    start = time.perf_counter()
    denoise_steps = run_denoise_only(pipeline, config, prompt, args.seed, dtype)
    sync()
    wall_seconds = time.perf_counter() - start

    dense_seconds = stats["dense"].seconds
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "dtype": args.dtype,
        "denoise_steps": denoise_steps,
        "wall_seconds": wall_seconds,
        "max_calls": args.max_calls,
        "variant_set": args.variant_set,
        "selected_variants": selected_variants,
        "variants": {
            name: value.summary(dense_seconds=dense_seconds if name != "dense" else None)
            for name, value in sorted(stats.items())
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
