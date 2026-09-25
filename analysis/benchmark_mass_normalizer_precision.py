"""Opt-in native/TF32 dot parity and timing on the same 40-head QKV.

The mathematical selector and all probe locations remain fixed. CUDA timings
exclude compilation and include complete GCA operator work, but not the model's
QKV projections/RoPE. SDPA is forced to the Flash backend and returns the same
contiguous [B,N,H,D] layout as GCA.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.entropy_mass_seed_triton import (
    entropy_mass_seed_rows_triton,
    global_probe_log_norm_triton,
)
from attention.token_merging import _hard_c4_attention
from evaluate_entropy_seed_oracle import _load_grid


RF10 = "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2"
MASS = "entropy_mass_span_c32_p16_g256_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2"
DOT_OPTIONS = {
    "original": {},
    "native": {"native_low_precision_dot": True},
    "tf32": {"tf32_low_precision_dot": True},
}


def _bitwise_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    # torch.equal alone regards +0 and -0 as equal and NaNs as unequal.
    # Compare storage bytes to make this an actual bitwise-parity test.
    return bool(torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    ))


def _agreement(left, right):
    return {
        "bitwise_equal": _bitwise_equal(left, right),
        "slot_agreement": float((left == right).float().mean()),
        "mean_shared_seeds": float(
            (left[..., :, None] == right[..., None, :]).any(-1).float().sum(-1).mean()
        ),
    }


def _difference(left, right):
    delta = left.float() - right.float()
    return {
        "bitwise_equal": _bitwise_equal(left, right),
        "elementwise_equal": bool(torch.equal(left, right)),
        "unequal_elements": int((left != right).sum()),
        "finite": bool(torch.isfinite(left).all()),
        "max_abs_error": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_l2": float(
            delta.square().sum().sqrt() / right.float().square().sum().sqrt().clamp_min(1e-20)
        ),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.warmup) < 1:
        raise ValueError("iterations, repeats and warmup must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    grids = _load_grid(args.qkv, list(range(40)))
    q, k, v = [
        grid.permute(1, 2, 3, 0, 4).reshape(1, 21 * 45 * 80, 40, 128)
        .contiguous().to(device=device, dtype=torch.bfloat16)
        for grid in grids
    ]
    del grids
    if any(tuple(t.shape) != (1, 75600, 40, 128) for t in (q, k, v)):
        raise ValueError("expected full 40-head Wan QKV")
    router = q.reshape(1, 21, 45, 80, 40, 128)[:, 10:11]
    keys = k.reshape(1, 21, 45, 80, 40, 128)
    scale = 128**-0.5
    result = {
        "settings": {**vars(args), "output": str(args.output)},
        "device": torch.cuda.get_device_name(device),
        "shape": list(q.shape), "dtype": str(q.dtype),
        "scope": "complete attention operators incl selector; excludes model projections/RoPE",
        "sdpa_backend": "PyTorch scaled_dot_product_attention, forced FLASH_ATTENTION",
        "output_layout": "contiguous [B,N,H,D] for every method",
        "fixed_algorithm": {
            "router_frame": 10, "candidates": 32, "candidate_geometry": 45,
            "uniform_probes": 16, "global_probes": 256, "block_j": 4,
            "gs_num_warps": 4, "norm_block_q": 32, "norm_num_warps": 8,
            "norm_num_stages": 1,
        },
        "normalizer_dot_modes": DOT_OPTIONS,
        "tf32_parity_caveat": (
            "Normal BF16 and finite FP16 inputs are exactly TF32-representable, "
            "but accumulation/lowering and BF16 subnormals or exceptional values "
            "can prevent bitwise equality. No parity is assumed."
        ),
        "parity": {}, "normalizer_latency": {}, "attention_latency": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def norm(mode):
        return global_probe_log_norm_triton(
            router, keys, scale, block_q=32, num_warps=8,
            **DOT_OPTIONS[mode],
        )

    norm_old = norm("original")
    seed_old = entropy_mass_seed_rows_triton(router, keys, scale, gs_num_warps=4)
    for mode in ("native", "tf32"):
        candidate_norm = norm(mode)
        result["parity"][f"normalizer_{mode}_vs_original"] = _difference(candidate_norm, norm_old)
        candidate_seeds = entropy_mass_seed_rows_triton(
            router, keys, scale, gs_num_warps=4,
            **DOT_OPTIONS[mode],
        )
        result["parity"][f"seeds_{mode}_vs_original"] = _agreement(candidate_seeds, seed_old)
        result["parity"][f"seed_{mode}_gate_passed"] = (
            result["parity"][f"seeds_{mode}_vs_original"]["mean_shared_seeds"] >= 3.98
        )
        del candidate_norm, candidate_seeds
    result["parity"]["seed_gate_passed"] = all(
        result["parity"][f"seed_{mode}_gate_passed"] for mode in ("native", "tf32")
    )
    del norm_old, seed_old
    # The FP32 fallback is a small, isolated one-head parity diagnostic. All
    # headline timing and BF16 parity measurements above/below use 40 heads.
    q32, k32 = router[..., :1, :].contiguous().float(), keys[..., :1, :].contiguous().float()
    fallback_old = global_probe_log_norm_triton(q32, k32, scale)
    for mode in ("native", "tf32"):
        fallback_on = global_probe_log_norm_triton(q32, k32, scale, **DOT_OPTIONS[mode])
        result["parity"][f"fp32_fallback_{mode}_single_head_bitwise_equal"] = _bitwise_equal(
            fallback_old, fallback_on
        )
        del fallback_on
    result["parity"]["fp32_fallback_single_head_bitwise_equal"] = all(
        result["parity"][f"fp32_fallback_{mode}_single_head_bitwise_equal"]
        for mode in ("native", "tf32")
    )
    del q32, k32, fallback_old
    save()
    print("parity", json.dumps(result["parity"]), flush=True)

    def gca(mode):
        return _hard_c4_attention(
            q, k, v, h=45, w=80, h_reduce=1, w_reduce=1,
            sm_scale=scale, query_outer_chunk=21, head_chunk=8,
            hard_route_backend="hard_c4", exact_intra_frame_attention=False,
            exact_neighbor_window_columns=0, exact_query_rows=0,
            first_frame_full_attention=False, full_attention_anchor_time_count=0,
            full_attention_anchor_row_count=0, full_attention_anchor_scale=1.0,
            seed_mode=mode,
        )

    def dense():
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False, scale=scale,
        ).transpose(1, 2).contiguous()

    functions = {
        "rf10": lambda: gca(RF10),
        "mass_original": lambda: gca(MASS),
        "mass_native": lambda: gca(MASS),
        "mass_tf32": lambda: gca(MASS),
        "sdpa": dense,
    }

    def context(name):
        if name.startswith("mass_"):
            return patch(
                "attention.token_merging.entropy_mass_seed_rows_triton",
                partial(
                    entropy_mass_seed_rows_triton,
                    **DOT_OPTIONS[name.removeprefix("mass_")],
                ),
            )
        if name == "sdpa":
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        return nullcontext()

    with context("mass_original"):
        original_output = functions["mass_original"]()
    for mode in ("native", "tf32"):
        with context(f"mass_{mode}"):
            candidate_output = functions[f"mass_{mode}"]()
        result["parity"][f"attention_{mode}_vs_original"] = _difference(candidate_output, original_output)
        result["parity"][f"all_{mode}_bitwise_equal"] = all(
            result["parity"][f"{stage}_{mode}_vs_original"]["bitwise_equal"]
            for stage in ("normalizer", "seeds", "attention")
        )
        del candidate_output
    del original_output
    save()

    def benchmark(name_to_function, section, context_factory):
        names = list(name_to_function)
        for name in names:
            with context_factory(name):
                for _ in range(args.warmup):
                    output = name_to_function[name]()
            del output
        torch.cuda.synchronize(device)
        samples = {name: [] for name in names}
        for repeat in range(args.repeats):
            for name in names if repeat % 2 == 0 else names[::-1]:
                # Install temporary dispatch/backend choices before recording
                # events so Python context setup is excluded from timing.
                with context_factory(name):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        output = name_to_function[name]()
                    end.record()
                    end.synchronize()
                samples[name].append(start.elapsed_time(end) / args.iterations)
                del output
                result[section][name] = {
                    "mean_ms": float(np.mean(samples[name])),
                    "std_ms": float(np.std(samples[name])),
                    "samples_ms": samples[name],
                }
                print(section, repeat, name, samples[name][-1], flush=True)
                save()

    benchmark({name: partial(norm, name) for name in DOT_OPTIONS},
              "normalizer_latency", lambda _: nullcontext())
    benchmark(functions, "attention_latency", context)
    rf10_ms = result["attention_latency"]["rf10"]["mean_ms"]
    sdpa_ms = result["attention_latency"]["sdpa"]["mean_ms"]
    for value in result["attention_latency"].values():
        value["delta_vs_rf10_ms"] = value["mean_ms"] - rf10_ms
        value["ratio_vs_rf10"] = value["mean_ms"] / rf10_ms
        value["speedup_vs_sdpa"] = sdpa_ms / value["mean_ms"]
    save()
    print(json.dumps(result, indent=2), flush=True)
    if not result["parity"]["seed_gate_passed"]:
        raise RuntimeError("native/TF32-dot seed agreement below 3.98/4; inspect per-mode parity")
    if not result["parity"]["fp32_fallback_single_head_bitwise_equal"]:
        raise RuntimeError("FP32 fallback changed")
    for mode in ("native", "tf32"):
        if not result["parity"][f"attention_{mode}_vs_original"]["finite"]:
            raise RuntimeError(f"{mode}-dot attention output is not finite")


if __name__ == "__main__":
    main()
