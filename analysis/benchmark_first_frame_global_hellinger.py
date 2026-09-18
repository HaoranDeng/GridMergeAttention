"""Benchmark first-frame global Hellinger routing at Wan 720p shape."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import torch

from attention.adaptive_seed_pytorch import first_frame_global_hellinger_seed_rows


def _synchronize():
    torch.cuda.synchronize()


def _measure(q, k, *, head_chunk, column_chunk, profile_dtype, repeats):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    output = first_frame_global_hellinger_seed_rows(
        q,
        k,
        alpha=0.2,
        candidate_count=32,
        candidate_geometry_count=45,
        num_seeds=4,
        head_chunk=head_chunk,
        column_chunk=column_chunk,
        profile_dtype=profile_dtype,
    )
    _synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = first_frame_global_hellinger_seed_rows(
            q,
            k,
            alpha=0.2,
            candidate_count=32,
            candidate_geometry_count=45,
            num_seeds=4,
            head_chunk=head_chunk,
            column_chunk=column_chunk,
            profile_dtype=profile_dtype,
        )
        _synchronize()
        samples.append(1000.0 * (time.perf_counter() - start))
    return {
        "head_chunk": head_chunk,
        "column_chunk": column_chunk,
        "profile_dtype": str(profile_dtype),
        "mean_ms": sum(samples) / len(samples),
        "samples_ms": samples,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--tiles",
        default="8x4,8x8,8x16,16x8,16x16,20x16,20x20,40x8",
    )
    parser.add_argument(
        "--profile-dtypes", default="float32,bfloat16"
    )
    parser.add_argument(
        "--float32-matmul-precision", default="highest"
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.float32_matmul_precision)

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        1, 1, 45, 80, 40, 4, device=device, dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn(
        1, 21, 45, 80, 40, 4, device=device, dtype=torch.bfloat16, generator=generator
    )
    baseline = None
    for dtype_name in args.profile_dtypes.split(","):
        profile_dtype = getattr(torch, dtype_name)
        for item in args.tiles.split(","):
            head_chunk, column_chunk = (int(value) for value in item.split("x"))
            result = _measure(
                q,
                k,
                head_chunk=head_chunk,
                column_chunk=column_chunk,
                profile_dtype=profile_dtype,
                repeats=args.repeats,
            )
            output = result.pop("output")
            if baseline is None:
                baseline = output
            result["seed_agreement"] = float((output == baseline).float().mean())
            result["seed_sha256"] = hashlib.sha256(
                output.cpu().numpy().tobytes()
            ).hexdigest()
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
