"""Same-process complete-attention timing for temporal seed-query routers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from attention.token_merging import _hard_c4_attention
from evaluate_entropy_seed_oracle import _load_grid


MODES = {
    "rf10": "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_u16_cg45_sq1_qm_fp8score_skv_nl_ml2",
    "ts3": "hellinger_span_alpha0p2_c32_p16_a2_j4_t1_u16_ts3_cg45_sq1_qm_fp8score_skv_nl_ml2",
}


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seed_mode: str):
    return _hard_c4_attention(
        q,
        k,
        v,
        h=45,
        w=80,
        h_reduce=1,
        w_reduce=1,
        sm_scale=128**-0.5,
        query_outer_chunk=21,
        head_chunk=8,
        hard_route_backend="hard_c4",
        exact_intra_frame_attention=False,
        exact_neighbor_window_columns=0,
        exact_query_rows=0,
        first_frame_full_attention=False,
        full_attention_anchor_time_count=0,
        full_attention_anchor_row_count=0,
        full_attention_anchor_scale=1.0,
        seed_mode=seed_mode,
    )


def timed(q, k, v, mode, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = run(q, k, v, MODES[mode])
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    heads = list(range(40))
    q_grid, k_grid, v_grid = _load_grid(args.qkv, heads)

    def video(grid):
        return (
            grid.permute(1, 2, 3, 0, 4)
            .reshape(1, 21 * 45 * 80, 40, 128)
            .contiguous()
            .to(device=device, dtype=torch.bfloat16)
        )

    q, k, v = video(q_grid), video(k_grid), video(v_grid)
    del q_grid, k_grid, v_grid

    # Compile and populate allocator pools before measurement.
    warm_outputs = {name: run(q, k, v, mode) for name, mode in MODES.items()}
    torch.cuda.synchronize()
    del warm_outputs

    samples = defaultdict(list)
    last_outputs = {}
    forward = list(MODES)
    for repeat in range(args.repeats):
        order = forward if repeat % 2 == 0 else list(reversed(forward))
        for name in order:
            value, output = timed(q, k, v, name, args.iterations)
            samples[name].append(value)
            last_outputs[name] = output

    baseline = last_outputs["rf10"].float()
    output_comparison = {}
    for name, output in last_outputs.items():
        difference = output.float() - baseline
        output_comparison[name] = {
            "rmse_vs_rf10": float(difference.square().mean().sqrt()),
            "relative_l2_vs_rf10": float(
                difference.square().sum().sqrt() / baseline.square().sum().sqrt()
            ),
        }

    summary = {}
    baseline_mean = float(np.mean(samples["rf10"]))
    for name, values in samples.items():
        mean = float(np.mean(values))
        summary[name] = {
            "mean_ms": mean,
            "std_ms": float(np.std(values)),
            "samples_ms": values,
            "delta_vs_rf10_ms": mean - baseline_mean,
            "ratio_vs_rf10": mean / baseline_mean,
        }
    result = {
        "device": torch.cuda.get_device_name(device),
        "capture": args.qkv,
        "iterations_per_sample": args.iterations,
        "repeats": args.repeats,
        "summary": summary,
        "output_comparison": output_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
