import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attention.flash_attn_triton import flash_attention_triton


def sdpa_reference(q, k, v, scale):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.transpose(1, 2).contiguous()


def time_call(fn, warmup, repeats):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(repeats, 1)), out


def error_summary(out, ref):
    diff = out.float() - ref.float()
    return {
        "mse": float((diff * diff).mean().item()),
        "rel_l2": float(diff.norm().item() / ref.float().norm().clamp_min(1e-20).item()),
        "max_abs": float(diff.abs().max().item()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark dense FlashAttention-style Triton kernel against PyTorch SDPA.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q_len", type=int, default=1560)
    parser.add_argument("--k_len", type=int, default=32760)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is required.")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    scale = args.dim ** -0.5
    q = torch.randn(args.batch, args.q_len, args.heads, args.dim, device="cuda", dtype=dtype)
    k = torch.randn(args.batch, args.k_len, args.heads, args.dim, device="cuda", dtype=dtype)
    v = torch.randn(args.batch, args.k_len, args.heads, args.dim, device="cuda", dtype=dtype)

    sdpa_ms, ref = time_call(lambda: sdpa_reference(q, k, v, scale), args.warmup, args.repeats)
    triton_ms, out = time_call(lambda: flash_attention_triton(q, k, v, scale), args.warmup, args.repeats)

    payload = {
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "shape_q": list(q.shape),
        "shape_k": list(k.shape),
        "shape_v": list(v.shape),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sdpa": {"mean_ms": sdpa_ms},
        "flash_attn_triton": {
            "mean_ms": triton_ms,
            "speedup_vs_sdpa": sdpa_ms / triton_ms if triton_ms else None,
            **error_summary(out, ref),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
