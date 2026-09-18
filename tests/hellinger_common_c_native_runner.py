import json
import time

import torch

from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
from attention.hard_c5_c6_wan21_14b_720p_triton import (
    hard_c_720p_native_temporal_seed_rows_attention_triton_blocks,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm or CUDA GPU is required")

    torch.manual_seed(0)
    device = torch.device("cuda")
    shape = (1, 21, 45, 80, 1, 128)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)
    scale = shape[-1] ** -0.5
    results = []

    for c in (4, 6, 8):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        seed_rows = hellinger_span_seed_rows_triton(
            q[:, :1],
            k,
            scale,
            alpha=0.2,
            candidate_count=32,
            candidate_geometry_count=45,
            top_probes=8,
            uniform_probes=8,
            probe_anchor_count=2,
            block_j=4,
            output_dtype=torch.uint8,
            num_seeds=c,
        )
        output = hard_c_720p_native_temporal_seed_rows_attention_triton_blocks(
            q,
            k,
            v,
            seed_rows,
            c,
            scale,
            head_start=0,
            head_count=1,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        unique_count = torch.sort(seed_rows.to(torch.int16), dim=-1).values
        unique_count = 1 + (unique_count[..., 1:] != unique_count[..., :-1]).sum(-1)
        if int(unique_count.min().item()) != c:
            raise AssertionError(f"C={c} selector returned duplicate seed rows")
        if int(seed_rows.max().item()) >= shape[2]:
            raise AssertionError(f"C={c} selector returned an out-of-range row")
        if not bool(torch.isfinite(output).all().item()):
            raise AssertionError(f"C={c} attention returned a non-finite output")

        results.append(
            {
                "c": c,
                "elapsed_seconds_including_compile": elapsed,
                "seed_min": int(seed_rows.min().item()),
                "seed_max": int(seed_rows.max().item()),
                "minimum_unique_seeds": int(unique_count.min().item()),
                "output_mean": float(output.float().mean().item()),
                "output_std": float(output.float().std().item()),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )
        del seed_rows, output, unique_count
        torch.cuda.empty_cache()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
