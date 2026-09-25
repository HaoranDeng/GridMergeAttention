"""Probe Q-only local basis refinement of a shared weighted-span table."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attention.adaptive_seed_triton import span_seed_rows_triton_split
from sweep_simple_weighted_span_qkv import COLS, FRAMES, ROWS, _dense_frame, _parse_indices
from sweep_simple_weighted_span_selectors import evaluate_rows


def _load_grid(path: str, head: int) -> tuple[torch.Tensor, ...]:
    with np.load(path) as payload:
        return tuple(
            torch.from_numpy(payload[name][0, head].copy()).reshape(
                FRAMES, ROWS, COLS, -1
            )
            for name in ("q", "k", "v")
        )


def _refine_rows(
    query: torch.Tensor,
    base_rows: torch.Tensor,
    radius: int,
    union_candidates: bool,
) -> torch.Tensor:
    """Refine dynamic seed rows using only target-frame query geometry."""

    columns, blocks, _ = base_rows.shape
    offsets = torch.arange(-radius, radius + 1, device=query.device)
    selected_rows = [base_rows[..., 0]]
    first = query[
        torch.arange(columns, device=query.device)[:, None], selected_rows[0]
    ].float()
    basis = [first * torch.rsqrt(first.square().sum(-1, keepdim=True) + 1e-6)]
    output = base_rows.clone()

    for slot in range(1, 4):
        centers = base_rows[..., 1:4] if union_candidates else base_rows[..., slot : slot + 1]
        candidate_rows = (centers[..., None] + offsets).clamp(0, ROWS - 1)
        candidate_rows = candidate_rows.flatten(2)
        column_ids = torch.arange(columns, device=query.device)[:, None, None]
        candidates = query[column_ids, candidate_rows].float()
        residual = candidates
        for vector in basis:
            projection = (residual * vector[:, :, None]).sum(-1, keepdim=True)
            residual = residual - projection * vector[:, :, None]
        score = residual.square().sum(-1)
        for used in selected_rows:
            score = score.masked_fill(candidate_rows == used[..., None], -1.0)
        pick = score.argmax(-1)
        row = candidate_rows.gather(-1, pick[..., None]).squeeze(-1)
        selected_rows.append(row)
        output[..., slot] = row
        picked = residual.gather(
            2,
            pick[..., None, None].expand(-1, -1, 1, residual.size(-1)),
        ).squeeze(2)
        basis.append(picked * torch.rsqrt(picked.square().sum(-1, keepdim=True) + 1e-6))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--frames", default="0,5,10,15,20")
    parser.add_argument("--router_frame", type=int, default=10)
    parser.add_argument("--radii", default="1,2,3,4")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frames = _parse_indices(args.frames)
    radii = _parse_indices(args.radii)
    names = ["base"] + [
        f"{'union' if union else 'slot'}_r{radius}"
        for radius in radii
        for union in (False, True)
    ]
    totals = {
        name: {
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "numel": 0,
            "points": 0,
            "refine_ms": [],
        }
        for name in names
    }

    for capture in args.qkv:
        q_grid, k_grid, v_grid = _load_grid(capture, args.head)
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        v_grid = v_grid.to(args.device)
        production_q = (
            q_grid[args.router_frame : args.router_frame + 1]
            .unsqueeze(0)
            .unsqueeze(4)
            .contiguous()
        )
        production_k = k_grid.unsqueeze(0).unsqueeze(4).contiguous()
        production_v = v_grid.unsqueeze(0).unsqueeze(4).contiguous()
        table = span_seed_rows_triton_split(
            production_q,
            production_k,
            production_v,
            alpha=0.3,
            candidate_count=45,
            top_probes=8,
            uniform_probes=8,
            value_dims=32,
            block_j=4,
        )
        base_rows = (
            table[0, 0, ..., 0, :]
            .reshape(FRAMES * ROWS, COLS, 4)
            .permute(1, 0, 2)
            .contiguous()
        )
        key = k_grid.reshape(FRAMES * ROWS, COLS, -1)
        value = v_grid.reshape(FRAMES * ROWS, COLS, -1)

        for frame in frames:
            query = q_grid[frame].permute(1, 0, 2).contiguous()
            dense = _dense_frame(q_grid, k_grid, v_grid, frame)
            variants = {"base": base_rows}
            for radius in radii:
                for union in (False, True):
                    name = f"{'union' if union else 'slot'}_r{radius}"
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    variants[name] = _refine_rows(query, base_rows, radius, union)
                    torch.cuda.synchronize()
                    totals[name]["refine_ms"].append(
                        1000.0 * (time.perf_counter() - started)
                    )
            for name, rows in variants.items():
                metrics = evaluate_rows(query, key, value, dense, rows)
                totals[name]["squared_error"] += metrics["squared_error"]
                totals[name]["reference_squared"] += metrics["reference_squared"]
                totals[name]["numel"] += metrics["numel"]
                totals[name]["points"] += 1

    result = {"settings": vars(args), "variants": {}}
    for name, total in totals.items():
        timings = total.pop("refine_ms")
        result["variants"][name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "points": total["points"],
            "refine_ms_per_frame_head": (
                sum(timings) / len(timings) if timings else 0.0
            ),
        }
    result["ranking"] = sorted(
        result["variants"],
        key=lambda name: result["variants"][name]["rel_l2_vs_dense"],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["variants"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
