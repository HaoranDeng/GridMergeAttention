"""Screen temporal entropy-barycenter routers on real Wan2.1 QKV.

For per-frame probe distributions ``p_a = softmax(q_a K^T)``, the minimizer
of ``mean_a KL(p || p_a)`` is their normalized geometric mean.  Its logits
are produced by the mean query, so it can replace the single RF10 router in
Hellinger-Span without increasing the number of probe query-key products.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from attention.adaptive_seed_triton import hellinger_span_seed_rows_triton
from evaluate_entropy_seed_oracle import (
    COLS,
    C4,
    FRAMES,
    ROWS,
    _load_grid,
    _parse_indices,
)
from sweep_simple_weighted_span_qkv import _dense_frame
from sweep_simple_weighted_span_selectors import evaluate_rows


def _matched_rms(source: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    target = source.float().square().sum(-1).sqrt().mean(1, keepdim=True)
    actual = mean.float().square().sum(-1).sqrt().clamp_min(1.0e-6)
    return (mean.float() * (target / actual).unsqueeze(-1)).to(source.dtype)


def _temporal_medoid(source: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    distance = (source.float() - mean.float()).square().sum(-1)
    index = distance.argmin(1, keepdim=True)
    return source.gather(
        1,
        index[..., None].expand(-1, -1, -1, -1, source.size(-1)),
    )


def router_variants(q: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return [heads,1,row,column,dimension] synthetic router queries."""

    middle = q[:, FRAMES // 2 : FRAMES // 2 + 1]
    mean21 = q.float().mean(1, keepdim=True).to(q.dtype)
    endpoints3 = q[:, (0, FRAMES // 2, FRAMES - 1)].float()
    simpson3 = (
        endpoints3[:, 0:1]
        + 4.0 * endpoints3[:, 1:2]
        + endpoints3[:, 2:3]
    ) / 6.0
    variants = {
        "rf10": middle,
        "mean3": q[:, 9:12].float().mean(1, keepdim=True).to(q.dtype),
        "mean5": q[:, 8:13].float().mean(1, keepdim=True).to(q.dtype),
        "mean21": mean21,
        "mean21_rms": _matched_rms(q, mean21),
        "medoid21": _temporal_medoid(q, mean21),
        "simpson3": simpson3.to(q.dtype),
    }
    for label, mix in (("mix25", 0.25), ("mix50", 0.5), ("mix75", 0.75)):
        variants[label] = torch.lerp(middle.float(), mean21.float(), mix).to(q.dtype)
    for label, mix in (("simp25", 0.25), ("simp50", 0.5)):
        variants[label] = torch.lerp(middle.float(), simpson3, mix).to(q.dtype)
    return variants


def _native_router(router: torch.Tensor) -> torch.Tensor:
    return router.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()


def _rows(table: torch.Tensor, head_slot: int) -> torch.Tensor:
    return (
        table[0, 0, ..., head_slot, :]
        .reshape(FRAMES * ROWS, COLS, C4)
        .permute(1, 0, 2)
        .to(torch.long)
    )


def _overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left[..., :, None] == right[..., None, :])
        .any(-1)
        .float()
        .sum(-1)
        .mean()
    )


def _new_total() -> dict[str, float]:
    return {
        "squared_error": 0.0,
        "reference_squared": 0.0,
        "numel": 0.0,
        "points": 0.0,
        "overlap": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate_count", type=int, default=32)
    parser.add_argument("--candidate_geometry_count", type=int, default=45)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--uniform_probes", type=int, default=16)
    parser.add_argument("--block_j", type=int, default=4)
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    device = torch.device(args.device)
    totals: dict[str, dict[str, float]] = defaultdict(_new_total)
    selector_ms: dict[str, list[float]] = defaultdict(list)
    records: dict[str, object] = {}

    for capture in args.qkv:
        q, k, v = _load_grid(capture, heads)
        q, k, v = q.to(device), k.to(device), v.to(device)
        native_k = k.permute(1, 2, 3, 0, 4).unsqueeze(0).contiguous()
        variants = router_variants(q)

        # Compile once so reported selector times exclude JIT startup.
        hellinger_span_seed_rows_triton(
            _native_router(variants["rf10"]),
            native_k,
            alpha=args.alpha,
            candidate_count=args.candidate_count,
            candidate_geometry_count=args.candidate_geometry_count,
            top_probes=0,
            uniform_probes=args.uniform_probes,
            probe_anchor_count=2,
            block_j=args.block_j,
            group_tile=32,
            output_dtype=torch.uint8,
            gs_num_warps=8,
            selection_mode="gs",
        )
        tables = {}
        for name, router in variants.items():
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            tables[name] = hellinger_span_seed_rows_triton(
                _native_router(router),
                native_k,
                alpha=args.alpha,
                candidate_count=args.candidate_count,
                candidate_geometry_count=args.candidate_geometry_count,
                top_probes=0,
                uniform_probes=args.uniform_probes,
                probe_anchor_count=2,
                block_j=args.block_j,
                group_tile=32,
                output_dtype=torch.uint8,
                gs_num_warps=8,
                selection_mode="gs",
            )
            torch.cuda.synchronize(device)
            selector_ms[name].append(1000.0 * (time.perf_counter() - started))

        capture_record = {}
        for head_slot, head in enumerate(heads):
            key_blocks = k[head_slot].reshape(FRAMES * ROWS, COLS, -1)
            value_blocks = v[head_slot].reshape(FRAMES * ROWS, COLS, -1)
            rows = {name: _rows(table, head_slot) for name, table in tables.items()}
            reference_rows = rows["rf10"]
            for frame in frames:
                query = q[head_slot, frame].permute(1, 0, 2).contiguous()
                dense = _dense_frame(q[head_slot], k[head_slot], v[head_slot], frame)
                point_name = f"{Path(capture).parent.name}:h{head}:f{frame}"
                point = {}
                for name, seed_rows in rows.items():
                    metrics = evaluate_rows(
                        query, key_blocks, value_blocks, dense, seed_rows
                    )
                    metrics["overlap_vs_rf10"] = _overlap(
                        seed_rows, reference_rows
                    )
                    point[name] = metrics
                    total = totals[name]
                    difference = metrics["squared_error"]
                    total["squared_error"] += difference
                    total["reference_squared"] += metrics["reference_squared"]
                    total["numel"] += metrics["numel"]
                    total["points"] += 1
                    total["overlap"] += metrics["overlap_vs_rf10"]
                capture_record[point_name] = point
                order = sorted(point, key=lambda name: point[name]["mse_vs_dense"])
                print(
                    point_name
                    + " "
                    + " ".join(
                        f"{name}={point[name]['mse_vs_dense']:.8f}"
                        for name in order
                    ),
                    flush=True,
                )
                del dense
            del rows, key_blocks, value_blocks
        records[Path(capture).parent.name] = capture_record
        del q, k, v, native_k, variants, tables
        torch.cuda.empty_cache()

    aggregate = {}
    for name, total in totals.items():
        aggregate[name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / total["reference_squared"]
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "overlap_vs_rf10": total["overlap"] / total["points"],
            "selector_ms_all_heads_mean": float(np.mean(selector_ms[name])),
            "selector_ms_all_heads_std": float(np.std(selector_ms[name])),
            "points": int(total["points"]),
        }
    baseline = aggregate["rf10"]
    for value in aggregate.values():
        value["mse_change_vs_rf10"] = 1.0 - (
            value["mse_vs_dense"] / baseline["mse_vs_dense"]
        )
    ranking = sorted(aggregate, key=lambda name: aggregate[name]["mse_vs_dense"])
    result = {
        "settings": {
            **vars(args),
            "qkv": args.qkv,
            "heads": heads,
            "frames": frames,
            "device_name": torch.cuda.get_device_name(device),
            "selector_probe_qk_count_matches_rf10": True,
            "all_output_seeds_are_real_current_frame_queries": True,
        },
        "aggregate": aggregate,
        "ranking": ranking,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("\naggregate", flush=True)
    for name in ranking:
        value = aggregate[name]
        print(
            f"{name:>12s} mse={value['mse_vs_dense']:.10f} "
            f"change={value['mse_change_vs_rf10']:+.2%} "
            f"rel_l2={value['rel_l2_vs_dense']:.7f} "
            f"overlap={value['overlap_vs_rf10']:.3f}/4 "
            f"selector={value['selector_ms_all_heads_mean']:.3f}ms",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
