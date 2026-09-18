"""Evaluate interpolation between two weighted-span router tables."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
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


def _aligned_second(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    permutations = torch.tensor(
        list(itertools.permutations(range(3))),
        device=first.device,
        dtype=torch.long,
    )
    candidates = second[..., permutations]
    cost = (first[..., None, :] - candidates).abs().sum(-1)
    best = cost.argmin(-1)
    return candidates.gather(
        2,
        best[..., None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)


def _interpolate_rows(
    tables: torch.Tensor,
    frame: int,
    routers: tuple[int, int],
    align_slots: bool,
) -> torch.Tensor:
    left, right = routers
    if frame <= left:
        return tables[0]
    if frame >= right:
        return tables[1]
    fraction = (frame - left) / (right - left)
    first = tables[0].float()
    second = tables[1].float()
    if align_slots:
        second = torch.cat(
            (second[..., :1], _aligned_second(first[..., 1:4], second[..., 1:4])),
            dim=-1,
        )
    return ((1.0 - fraction) * first + fraction * second).round().long()


def _query_router_choice(
    q_grid: torch.Tensor,
    frame: int,
    routers: tuple[int, int],
    per_column: bool,
) -> torch.Tensor:
    target = q_grid[frame].permute(1, 0, 2).float()
    target = target * torch.rsqrt(target.square().sum(-1, keepdim=True) + 1e-6)
    references = q_grid[list(routers)].permute(0, 2, 1, 3).float()
    references = references * torch.rsqrt(
        references.square().sum(-1, keepdim=True) + 1e-6
    )
    similarity = (references * target[None]).sum(-1).mean(-1)
    if per_column:
        return similarity.argmax(0)
    return similarity.mean(-1).argmax().view(1).expand(COLS)


def _union_query_span(
    q_grid: torch.Tensor,
    tables: torch.Tensor,
    routers: tuple[int, int],
) -> torch.Tensor:
    candidates = torch.cat((tables[0, ..., 1:4], tables[1, ..., 1:4]), dim=-1)
    columns, blocks, candidate_count = candidates.shape
    column_ids = torch.arange(columns, device=q_grid.device)[:, None, None]
    profiles = []
    for router in routers:
        query = q_grid[router].permute(1, 0, 2)
        profiles.append(query[column_ids, candidates].float())
    residual = torch.cat(profiles, dim=-1)
    residual = residual - residual.mean(-1, keepdim=True)
    anchor_row = tables[0, ..., 0]
    anchor_profiles = []
    for router in routers:
        query = q_grid[router].permute(1, 0, 2)
        anchor_profiles.append(
            query[
                torch.arange(columns, device=q_grid.device)[:, None],
                anchor_row,
            ].float()
        )
    basis_src = torch.cat(anchor_profiles, dim=-1)
    basis_src = basis_src - basis_src.mean(-1, keepdim=True)
    used = candidates == anchor_row[..., None]
    selected = [anchor_row]
    for _ in range(3):
        basis = basis_src * torch.rsqrt(
            basis_src.square().sum(-1, keepdim=True) + 1e-6
        )
        projection = (residual * basis[..., None, :]).sum(-1, keepdim=True)
        residual = residual - projection * basis[..., None, :]
        score = residual.square().sum(-1).masked_fill(used, -1.0)
        pick = score.argmax(-1)
        row = candidates.gather(-1, pick[..., None]).squeeze(-1)
        selected.append(row)
        used = used | (candidates == row[..., None])
        basis_src = residual.gather(
            2,
            pick[..., None, None].expand(-1, -1, 1, residual.size(-1)),
        ).squeeze(2)
    return torch.stack(selected, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--frames", default="0,2,5,8,10,12,15,18,20")
    parser.add_argument("--routers", default="5,15")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    frames = _parse_indices(args.frames)
    routers_list = _parse_indices(args.routers)
    if len(routers_list) != 2 or routers_list[0] >= routers_list[1]:
        raise ValueError("routers must be two increasing frame indices")
    routers = (routers_list[0], routers_list[1])
    names = (
        "nearest",
        "linear",
        "linear_aligned",
        "query_global",
        "query_column",
        "union_query_span",
    )
    totals = {
        name: {"squared_error": 0.0, "reference_squared": 0.0, "numel": 0, "points": 0}
        for name in names
    }

    for capture in args.qkv:
        q_grid, k_grid, v_grid = _load_grid(capture, args.head)
        q_grid = q_grid.to(args.device)
        k_grid = k_grid.to(args.device)
        v_grid = v_grid.to(args.device)
        production_q = q_grid[list(routers)].unsqueeze(0).unsqueeze(4).contiguous()
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
        tables = (
            table[0, ..., 0, :]
            .reshape(2, FRAMES * ROWS, COLS, 4)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        key = k_grid.reshape(FRAMES * ROWS, COLS, -1)
        value = v_grid.reshape(FRAMES * ROWS, COLS, -1)
        midpoint = (routers[0] + routers[1]) / 2
        union_query_span = _union_query_span(q_grid, tables, routers)

        for frame in frames:
            query = q_grid[frame].permute(1, 0, 2).contiguous()
            dense = _dense_frame(q_grid, k_grid, v_grid, frame)
            global_choice = _query_router_choice(q_grid, frame, routers, False)
            column_choice = _query_router_choice(q_grid, frame, routers, True)
            columns = torch.arange(COLS, device=q_grid.device)
            variants = {
                "nearest": tables[int(frame > midpoint)],
                "linear": _interpolate_rows(tables, frame, routers, False),
                "linear_aligned": _interpolate_rows(tables, frame, routers, True),
                "query_global": tables[global_choice, columns],
                "query_column": tables[column_choice, columns],
                "union_query_span": union_query_span,
            }
            for name, rows in variants.items():
                metrics = evaluate_rows(query, key, value, dense, rows)
                totals[name]["squared_error"] += metrics["squared_error"]
                totals[name]["reference_squared"] += metrics["reference_squared"]
                totals[name]["numel"] += metrics["numel"]
                totals[name]["points"] += 1

    result = {"settings": vars(args), "variants": {}}
    for name, total in totals.items():
        result["variants"][name] = {
            "rel_l2_vs_dense": math.sqrt(
                total["squared_error"] / max(total["reference_squared"], 1e-20)
            ),
            "mse_vs_dense": total["squared_error"] / total["numel"],
            "points": total["points"],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["variants"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
