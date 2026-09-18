"""Validate and benchmark exact GMA attention-periphery optimizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


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


def difference(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "equal": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max()),
        "rel_l2": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=75600)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--head_chunk", type=int, default=8)
    parser.add_argument("--head_chunks", default="4,8,10,20,40")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    svg_root = Path(args.svg_root).resolve()
    sys.path.insert(0, str(svg_root))
    sys.path.insert(0, str(svg_root / "svg" / "kernels" / "build"))
    from analysis.benchmark_wan14b_720p_methods import install_rocm_import_stubs

    install_rocm_import_stubs()

    torch_compile = torch.compile
    torch.compile = lambda function, *args, **kwargs: function
    try:
        from attention.hard_c4_wan21_14b_720p_triton import (
            hard_c4_720p_attention_triton_head_major,
            hard_c4_720p_attention_triton_head_major_to_token,
        )
        from attention.wan_qkv_preprocess_triton import preprocess_qkv_head_major
        from svg.kernels.triton.rmsnorm import triton_rmsnorm_forward
        from svg.models.wan.attention import apply_rotary_emb
    finally:
        torch.compile = torch_compile

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    width = args.heads * args.head_dim
    shape = (1, args.tokens, width)
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    query_weight = torch.randn(width, device=device, dtype=dtype)
    key_weight = torch.randn_like(query_weight)
    angles = torch.randn(
        (args.tokens, args.head_dim // 2), device=device, dtype=torch.float32
    )
    rotary_emb = (torch.cos(angles), torch.sin(angles))
    eps = 1e-6

    projection_weights = [
        torch.randn((width, width), device=device, dtype=dtype) for _ in range(3)
    ]
    fused_projection_weight = torch.cat(projection_weights, dim=0)

    def separate_projection():
        return tuple(F.linear(query, weight) for weight in projection_weights)

    def fused_projection():
        return F.linear(query, fused_projection_weight).split(width, dim=-1)

    separate_qkv = separate_projection()
    fused_qkv = fused_projection()
    projection = {
        "separate_ms": timed(separate_projection, args.warmup, args.repeats),
        "fused_ms": timed(fused_projection, args.warmup, args.repeats),
        "query": difference(fused_qkv[0], separate_qkv[0]),
        "key": difference(fused_qkv[1], separate_qkv[1]),
        "value": difference(fused_qkv[2], separate_qkv[2]),
    }
    projection["speedup"] = projection["separate_ms"] / projection["fused_ms"]
    del projection_weights, fused_projection_weight, separate_qkv, fused_qkv
    torch.cuda.empty_cache()

    def reference_preprocess():
        q = triton_rmsnorm_forward(query, query_weight, eps)
        k = triton_rmsnorm_forward(key, key_weight, eps)
        q = q.unflatten(2, (args.heads, -1)).transpose(1, 2).contiguous()
        k = k.unflatten(2, (args.heads, -1)).transpose(1, 2).contiguous()
        v = value.unflatten(2, (args.heads, -1)).transpose(1, 2).contiguous()
        q, k = apply_rotary_emb(q, k, rotary_emb)
        return q, k, v

    def fused_preprocess():
        q, k, v = preprocess_qkv_head_major(
            query,
            key,
            value,
            query_weight,
            key_weight,
            eps,
            eps,
            args.heads,
        )
        q, k = apply_rotary_emb(q, k, rotary_emb)
        return q, k, v

    q_reference, k_reference, v_reference = reference_preprocess()
    q_fused, k_fused, v_fused = fused_preprocess()
    packed_qkv = torch.cat((query, key, value), dim=-1)
    packed_query, packed_key, packed_value = packed_qkv.split(width, dim=-1)
    packed_q, packed_k, packed_v = preprocess_qkv_head_major(
        packed_query,
        packed_key,
        packed_value,
        query_weight,
        key_weight,
        eps,
        eps,
        args.heads,
    )
    packed_q, packed_k = apply_rotary_emb(packed_q, packed_k, rotary_emb)
    packed_outputs = (packed_q, packed_k, packed_v)
    preprocess = {
        "reference_ms": timed(reference_preprocess, args.warmup, args.repeats),
        "fused_ms": timed(fused_preprocess, args.warmup, args.repeats),
        "query": difference(q_fused, q_reference),
        "key": difference(k_fused, k_reference),
        "value": difference(v_fused, v_reference),
        "packed_query": difference(packed_outputs[0], q_reference),
        "packed_key": difference(packed_outputs[1], k_reference),
        "packed_value": difference(packed_outputs[2], v_reference),
    }
    preprocess["speedup"] = preprocess["reference_ms"] / preprocess["fused_ms"]
    del packed_qkv, packed_query, packed_key, packed_value, packed_outputs

    def run_head_major(head_chunk):
        output = torch.empty_like(q_reference)
        for head_start in range(0, args.heads, head_chunk):
            hard_c4_720p_attention_triton_head_major(
                q_reference,
                k_reference,
                v_reference,
                output=output,
                head_start=head_start,
                head_count=min(head_chunk, args.heads - head_start),
            )
        return output

    def run_head_major_with_layout(head_chunk):
        output = run_head_major(head_chunk)
        return output.transpose(1, 2).flatten(2, 3).contiguous()

    def run_token_output(head_chunk):
        output = torch.empty((1, args.tokens, width), device=device, dtype=dtype)
        for head_start in range(0, args.heads, head_chunk):
            hard_c4_720p_attention_triton_head_major_to_token(
                q_reference,
                k_reference,
                v_reference,
                output=output,
                head_start=head_start,
                head_count=min(head_chunk, args.heads - head_start),
            )
        return output

    head_major = run_head_major_with_layout(args.head_chunk)
    token_output = run_token_output(args.head_chunk)
    attention = {
        "head_major_core_ms": timed(
            lambda: run_head_major(args.head_chunk), 1, args.repeats
        ),
        "head_major_core_plus_layout_ms": timed(
            lambda: run_head_major_with_layout(args.head_chunk), 1, args.repeats
        ),
        "token_output_ms": timed(
            lambda: run_token_output(args.head_chunk), 1, args.repeats
        ),
        "output": difference(token_output, head_major),
    }
    attention["speedup_vs_core_plus_layout"] = (
        attention["head_major_core_plus_layout_ms"] / attention["token_output_ms"]
    )
    attention["head_chunk_sweep"] = {}
    for head_chunk in [int(value) for value in args.head_chunks.split(",")]:
        head_major_ms = timed(
            lambda head_chunk=head_chunk: run_head_major_with_layout(head_chunk),
            1,
            args.repeats,
        )
        token_ms = timed(
            lambda head_chunk=head_chunk: run_token_output(head_chunk),
            1,
            args.repeats,
        )
        attention["head_chunk_sweep"][str(head_chunk)] = {
            "head_major_core_plus_layout_ms": head_major_ms,
            "token_output_ms": token_ms,
            "token_speedup": head_major_ms / token_ms,
        }

    payload = {
        "device": torch.cuda.get_device_name(),
        "shape": list(shape),
        "dtype": str(dtype),
        "head_chunk": args.head_chunk,
        "projection": projection,
        "preprocess": preprocess,
        "attention": attention,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
