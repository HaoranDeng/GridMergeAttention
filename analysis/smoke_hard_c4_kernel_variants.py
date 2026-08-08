import argparse
import json
from pathlib import Path

import torch

from attention.hard_route import hard_route_attention
from analysis.hard_route_nonshared_backend_real_qkv import exact_variants


def rel_l2(out, ref):
    diff = (out.float() - ref.float()).reshape(-1)
    denom = ref.float().reshape(-1).norm().clamp_min(1e-20)
    return float(diff.norm().item() / denom.item())


def run_variant(q, k, v, variant):
    return hard_route_attention(
        q,
        k,
        v,
        scale=q.size(-1) ** -0.5,
        num_merge_tokens=4,
        seed_mode=variant.get("seed_mode", "local_with_ith"),
        seed=0,
        query_outer_offset=0,
        hard_route_backend=variant["hard_route_backend"],
        hard_route_routing_backend=variant["hard_route_routing_backend"],
    )


def time_variant(q, k, v, variant, warmup, repeats):
    """Return average runtime for one attention call after Triton compilation."""
    for _ in range(warmup):
        run_variant(q, k, v, variant)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        run_variant(q, k, v, variant)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(repeats, 1))


def write_payload(output_json, payload):
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test hard C4 Triton variants on random tensors.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--query_outer", type=int, default=2)
    parser.add_argument("--key_outer", type=int, default=2)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--columns", type=int, default=52)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is required.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape_q = (args.batch, args.query_outer, args.rows, args.columns, args.heads, args.dim)
    shape_kv = (args.batch, args.key_outer, args.rows, args.columns, args.heads, args.dim)
    q = torch.randn(shape_q, device=device, dtype=dtype)
    k = torch.randn(shape_kv, device=device, dtype=dtype)
    v = torch.randn(shape_kv, device=device, dtype=dtype)

    variants = exact_variants()
    names = [
        "hard_c4_bestlogit_tokenmajor",
        "hard_c4_bestlogit_packkv_tokenmajor",
        "hard_c4_bestlogit_packkvbias_tokenmajor",
        "hard_c4_bestlogit_packkvbiascompact_tokenmajor",
        "hard_c4_bestlogit_recompute_tokenmajor",
        "hard_c4_bestlogit_recompute_packkv_tokenmajor",
        "hard_c4_bestlogit_recompute_packkvbias_tokenmajor",
        "hard_c4_bestlogit_recompute_packkvbiascompact_tokenmajor",
        # Keep non-power-of-two tile experiments last: an illegal memory access
        # here should not prevent safer variant results from being saved.
        "hard_c4_bestlogit_j52_tokenmajor",
        "hard_c4_bestlogit_packkvbiascompact_j52_tokenmajor",
        "hard_c4_bestlogit_l52_tokenmajor",
        "hard_c4_bestlogit_packkvbiascompact_l52_tokenmajor",
    ]

    results = {}
    payload = {
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "shape_q": list(shape_q),
        "shape_kv": list(shape_kv),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
        "complete": False,
    }

    ref_name = "hard_c4_bestlogit_tokenmajor"
    torch.cuda.synchronize()
    ref = run_variant(q, k, v, variants[ref_name])
    torch.cuda.synchronize()
    ref_time = time_variant(q, k, v, variants[ref_name], warmup=args.warmup, repeats=args.repeats)
    results[ref_name] = {
        "shape": list(ref.shape),
        "max_abs_vs_bestlogit": 0.0,
        "rel_l2_vs_bestlogit": 0.0,
        "mean_ms": ref_time,
    }
    write_payload(args.output_json, payload)

    for name in names:
        if name == ref_name:
            continue
        torch.cuda.synchronize()
        try:
            out = run_variant(q, k, v, variants[name])
            torch.cuda.synchronize()
            diff = (out.float() - ref.float()).abs()
            mean_ms = time_variant(
                q,
                k,
                v,
                variants[name],
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results[name] = {
                "shape": list(out.shape),
                "max_abs_vs_bestlogit": float(diff.max().item()),
                "rel_l2_vs_bestlogit": rel_l2(out, ref),
                "mean_ms": mean_ms,
            }
        except Exception as exc:
            results[name] = {"error": repr(exc)}
        write_payload(args.output_json, payload)

    payload["complete"] = True
    write_payload(args.output_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
