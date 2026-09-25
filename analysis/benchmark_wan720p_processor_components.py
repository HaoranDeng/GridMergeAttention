"""Benchmark Wan2.1-14B 720p QK normalization and rotary embedding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def timed(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def error(actual, expected):
    difference = actual.float() - expected.float()
    expected_f32 = expected.float()
    return {
        "rel_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected_f32)
        ),
        "max_abs": float(difference.abs().max()),
    }


class EventTimer:
    def __init__(self):
        self.events = []

    def measure(self, function):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        self.events.append((start, end))
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=75600)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    svg_root = Path(args.svg_root).resolve()
    sys.path.insert(0, str(svg_root))
    sys.path.insert(0, str(svg_root / "svg" / "kernels" / "build"))

    from diffusers.models.normalization import RMSNorm
    from diffusers.models.normalization import FP32LayerNorm
    from svg.kernels.triton.layernorm import triton_layernorm_forward
    from svg.kernels.triton.modulate import (
        triton_modulate_gate_residual_forward,
        triton_modulate_shift_forward,
    )
    from svg.kernels.triton.rmsnorm import triton_rmsnorm_forward
    import _kernels

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_dim = args.heads * args.head_dim

    x = torch.randn((1, args.tokens, hidden_dim), device=device, dtype=dtype)
    norm = RMSNorm(hidden_dim, eps=1e-6).to(device=device, dtype=dtype)
    torch_output = norm(x)
    triton_output = triton_rmsnorm_forward(x, norm.weight, norm.eps)
    rmsnorm = {
        "torch_ms": timed(lambda: norm(x), args.warmup, args.repeats),
        "triton_ms": timed(
            lambda: triton_rmsnorm_forward(x, norm.weight, norm.eps),
            args.warmup,
            args.repeats,
        ),
        **error(triton_output, torch_output),
    }
    rmsnorm["speedup"] = rmsnorm["torch_ms"] / rmsnorm["triton_ms"]
    del x, torch_output, triton_output, norm
    torch.cuda.empty_cache()

    q = torch.randn(
        (1, args.heads, args.tokens, args.head_dim), device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    angles = torch.randn(
        (args.tokens, args.head_dim // 2), device=device, dtype=torch.float32
    )
    freq_real, freq_imag = torch.cos(angles), torch.sin(angles)
    complex_freqs = torch.complex(freq_real, freq_imag)[None, None]

    def legacy_rope(query, key):
        def rotate(hidden_states):
            rotated = torch.view_as_complex(
                hidden_states.to(torch.float64).unflatten(3, (-1, 2))
            )
            return torch.view_as_real(rotated * complex_freqs).flatten(3, 4).to(dtype)

        return rotate(query), rotate(key)

    def fast_rope(query, key):
        _kernels.apply_qk_rope_inplace_cossin_complex(
            query, key, freq_real, freq_imag, 0
        )
        return query, key

    legacy_q, legacy_k = legacy_rope(q, k)
    fast_q, fast_k = fast_rope(q.clone(), k.clone())
    rope_error = error(fast_q, legacy_q)
    key_error = error(fast_k, legacy_k)
    rope = {
        "legacy_ms": timed(lambda: legacy_rope(q, k), args.warmup, args.repeats),
        "svg_ms": timed(lambda: fast_rope(q, k), args.warmup, args.repeats),
        "query_rel_l2": rope_error["rel_l2"],
        "query_max_abs": rope_error["max_abs"],
        "key_rel_l2": key_error["rel_l2"],
        "key_max_abs": key_error["max_abs"],
    }
    rope["speedup"] = rope["legacy_ms"] / rope["svg_ms"]
    del q, k, legacy_q, legacy_k, fast_q, fast_k, angles
    del freq_real, freq_imag, complex_freqs
    torch.cuda.empty_cache()

    hidden = torch.randn(
        (1, args.tokens, hidden_dim), device=device, dtype=dtype
    ).contiguous()
    layernorm = FP32LayerNorm(
        hidden_dim, eps=1e-6, elementwise_affine=False
    ).to(device)

    def torch_layernorm():
        return layernorm(hidden.float())

    def svg_layernorm():
        return triton_layernorm_forward(
            hidden,
            layernorm.weight,
            layernorm.bias,
            layernorm.eps,
            layernorm.elementwise_affine,
        )

    torch_norm_output = torch_layernorm()
    svg_norm_output = svg_layernorm()
    transformer_layernorm = {
        "torch_ms": timed(torch_layernorm, args.warmup, args.repeats),
        "triton_ms": timed(svg_layernorm, args.warmup, args.repeats),
        **error(svg_norm_output, torch_norm_output),
    }
    transformer_layernorm["speedup"] = (
        transformer_layernorm["torch_ms"]
        / transformer_layernorm["triton_ms"]
    )

    scale = torch.randn((1, 1, hidden_dim), device=device, dtype=torch.float32)
    shift = torch.randn_like(scale)

    def torch_modulate():
        return (svg_norm_output * (1 + scale) + shift).to(dtype)

    def svg_modulate():
        return triton_modulate_shift_forward(
            svg_norm_output, scale, shift, output_dtype=dtype
        )

    torch_modulate_output = torch_modulate()
    svg_modulate_output = svg_modulate()
    modulate_shift = {
        "torch_ms": timed(torch_modulate, args.warmup, args.repeats),
        "triton_ms": timed(svg_modulate, args.warmup, args.repeats),
        **error(svg_modulate_output, torch_modulate_output),
    }
    modulate_shift["speedup"] = (
        modulate_shift["torch_ms"] / modulate_shift["triton_ms"]
    )

    attention_output = torch.randn_like(hidden)
    gate = torch.randn_like(scale)

    def torch_gate_residual():
        return (hidden.float() + attention_output * gate).to(dtype)

    def svg_gate_residual():
        return triton_modulate_gate_residual_forward(
            hidden, attention_output, gate, output_dtype=dtype
        )

    torch_residual_output = torch_gate_residual()
    svg_residual_output = svg_gate_residual()
    gate_residual = {
        "torch_ms": timed(torch_gate_residual, args.warmup, args.repeats),
        "triton_ms": timed(svg_gate_residual, args.warmup, args.repeats),
        **error(svg_residual_output, torch_residual_output),
    }
    gate_residual["speedup"] = (
        gate_residual["torch_ms"] / gate_residual["triton_ms"]
    )

    timer_input = torch.randn((1024,), device=device)

    def timer_kernel():
        return timer_input + 1.0

    def time_timer_depth(depth, calls=4000):
        timers = [EventTimer() for _ in range(depth)]

        def invoke(level=0):
            if level == depth:
                return timer_kernel()
            return timers[level].measure(lambda: invoke(level + 1))

        for _ in range(20):
            invoke()
        torch.cuda.synchronize()
        for timer in timers:
            timer.events.clear()
        started = time.perf_counter()
        for _ in range(calls):
            invoke()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {"total_ms": elapsed_ms, "ms_per_call": elapsed_ms / calls}

    timer_overhead = {
        "calls": 4000,
        "no_timer": time_timer_depth(0),
        "one_timer": time_timer_depth(1),
        "two_nested_timers": time_timer_depth(2),
    }
    timer_overhead["second_timer_increment_ms_per_call"] = (
        timer_overhead["two_nested_timers"]["ms_per_call"]
        - timer_overhead["one_timer"]["ms_per_call"]
    )

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": {
            "flat_qk": [1, args.tokens, hidden_dim],
            "head_major_qk": [1, args.heads, args.tokens, args.head_dim],
        },
        "dtype": str(dtype),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rmsnorm": rmsnorm,
        "rope": rope,
        "transformer_layernorm": transformer_layernorm,
        "modulate_shift": modulate_shift,
        "gate_residual": gate_residual,
        "timer_overhead": timer_overhead,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
