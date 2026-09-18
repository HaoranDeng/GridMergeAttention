"""Validate and time conditioned GridMerge representatives on saved Wan QKV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from attention.conditioned_gridmerge_tail_cuda import (
    conditioned_gridmerge_representatives,
)


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def load_qkv(path: Path, heads: list[int], device: torch.device):
    selected = torch.tensor(heads, device=device)
    with np.load(path, allow_pickle=False) as capture:
        return tuple(
            torch.from_numpy(np.asarray(capture[name]))
            .to(device=device, dtype=torch.float16)
            .index_select(1, selected)
            for name in ("q", "k", "v")
        )


def reference_cluster(seed, keys, values):
    score = torch.mv(keys.float(), seed.float()) * seed.size(-1) ** -0.5
    weight = torch.softmax(score, dim=0)
    merged_key = torch.mv(keys.float().T, weight)
    merged_value = torch.mv(values.float().T, weight)
    entropy = -(weight * weight.clamp_min(1e-30).log()).sum()
    return merged_key, merged_value, entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=parse_ints, default=[0, 10, 20, 30])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--svg-root", default="/home/denghaoran/workspace/Sparse-VideoGen"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["SPARSE_VIDEO_GEN_ROOT"] = args.svg_root
    device = torch.device("cuda")
    query, key, value = load_qkv(args.qkv, args.heads, device)

    from attention.svg_family_wan import _configure

    modules = _configure("ear", (21, 45, 80), query)
    processor = modules["ear"](layer_idx=0)
    (
        _,
        key_permuted,
        value_permuted,
        _,
        _,
        _,
        key_sizes,
        _,
        query_centroids,
        _,
        _,
    ) = processor.semantic_aware_permutation(query, key, value)

    for _ in range(args.warmup):
        representatives = conditioned_gridmerge_representatives(
            query_centroids, key_permuted, value_permuted, key_sizes
        )
        torch.cuda.synchronize()
        del representatives

    elapsed_ms = []
    representatives = None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        representatives = conditioned_gridmerge_representatives(
            query_centroids, key_permuted, value_permuted, key_sizes
        )
        end.record()
        torch.cuda.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    merged_keys, merged_values, entropy = representatives
    cumulative = torch.cumsum(key_sizes[0, 0].long(), dim=0)
    starts = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    query_clusters = [0, query_centroids.size(2) // 2, query_centroids.size(2) - 1]
    key_clusters = [0, key_sizes.size(2) // 2, key_sizes.size(2) - 1]
    parity = []
    for query_cluster in query_clusters:
        for key_cluster in key_clusters:
            token_start = int(starts[key_cluster].item())
            token_end = int(cumulative[key_cluster].item())
            reference_key, reference_value, reference_entropy = reference_cluster(
                query_centroids[0, 0, query_cluster],
                key_permuted[0, 0, token_start:token_end],
                value_permuted[0, 0, token_start:token_end],
            )
            parity.append(
                {
                    "query_cluster": query_cluster,
                    "key_cluster": key_cluster,
                    "cluster_size": token_end - token_start,
                    "key_max_abs": float(
                        (
                            merged_keys[0, 0, query_cluster, key_cluster].float()
                            - reference_key
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "value_max_abs": float(
                        (
                            merged_values[0, 0, query_cluster, key_cluster].float()
                            - reference_value
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "entropy_abs": float(
                        (
                            entropy[0, 0, query_cluster, key_cluster]
                            - reference_entropy
                        )
                        .abs()
                        .item()
                    ),
                }
            )

    result = {
        "qkv": str(args.qkv),
        "heads": args.heads,
        "query_clusters": query_centroids.size(2),
        "key_clusters": key_sizes.size(2),
        "sequence": key_permuted.size(2),
        "max_key_cluster_size": int(key_sizes.max().item()),
        "elapsed_ms": elapsed_ms,
        "mean_elapsed_ms": sum(elapsed_ms) / len(elapsed_ms),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "parity": parity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
