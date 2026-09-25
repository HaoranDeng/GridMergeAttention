"""Evaluate strictly query-only Hard-C4 seed selectors with attention KL.

Selectors receive only a query frame and structural grid dimensions. Keys are
used after selection solely to evaluate the induced original-key attention
distribution. All selected seeds are real query rows.
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
import torch.nn.functional as F


FRAMES = 21
ROWS = 45
COLS = 80
C4 = 4


def _parse_indices(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def _load_qk(path: Path, heads: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as payload:
        q = torch.from_numpy(payload["q"][0, heads].copy())
        k = torch.from_numpy(payload["k"][0, heads].copy())
    return q, k


def _production_wide_rows(device: torch.device) -> torch.Tensor:
    key_row = torch.arange(ROWS, device=device)
    start = (key_row - 1).clamp(0, ROWS - C4)
    rows = start[:, None] + torch.arange(C4, device=device)
    offsets = torch.tensor((-2, 0, 1, 3), device=device)
    wide = key_row[:, None] + offsets
    interior = (key_row >= 2) & (key_row <= ROWS - 4)
    return torch.where(interior[:, None], wide, rows)


def _candidate_rows(device: torch.device, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not C4 <= count <= ROWS:
        raise ValueError(f"candidate_count must be in [{C4}, {ROWS}]")
    key_row = torch.arange(ROWS, device=device)
    start = (key_row - count // 2).clamp(0, ROWS - count)
    rows = start[:, None] + torch.arange(count, device=device)
    return rows, key_row - start


def _gather_query_rows(query: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Gather query[J,R,D] with rows[J,B,C] or rows[B,C]."""

    if rows.ndim == 2:
        rows = rows.unsqueeze(0).expand(query.size(0), -1, -1)
    columns = torch.arange(query.size(0), device=query.device)[:, None, None]
    return query[columns, rows]


def _pairwise_squared(points: torch.Tensor) -> torch.Tensor:
    points = points.float()
    norm = points.square().sum(dim=-1)
    gram = torch.einsum("...pd,...qd->...pq", points, points)
    return (norm.unsqueeze(-1) + norm.unsqueeze(-2) - 2.0 * gram).clamp_min_(0.0)


def _distance_column(distance: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    count = distance.size(-1)
    gather_index = index[..., None, None].expand(*index.shape, count, 1)
    return distance.gather(-1, gather_index).squeeze(-1)


def _anchored_farthest(
    distance: torch.Tensor,
    anchor: torch.Tensor,
    importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy metric k-center with a mandatory first center."""

    prefix = distance.shape[:-2]
    count = distance.size(-1)
    anchor = anchor.expand(prefix)
    selected = [anchor]
    used = torch.zeros(*prefix, count, device=distance.device, dtype=torch.bool)
    used.scatter_(-1, anchor.unsqueeze(-1), True)
    minimum = _distance_column(distance, anchor)
    for _ in range(1, C4):
        score = minimum if importance is None else minimum * importance
        index = score.masked_fill(used, -1.0).argmax(dim=-1)
        selected.append(index)
        used.scatter_(-1, index.unsqueeze(-1), True)
        minimum = torch.minimum(minimum, _distance_column(distance, index))
    return torch.stack(selected, dim=-1)


def _anchored_facility(
    distance: torch.Tensor,
    anchor: torch.Tensor,
    demand_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedy reduction of weighted k-medoids coverage cost."""

    prefix = distance.shape[:-2]
    count = distance.size(-1)
    anchor = anchor.expand(prefix)
    selected = [anchor]
    used = torch.zeros(*prefix, count, device=distance.device, dtype=torch.bool)
    used.scatter_(-1, anchor.unsqueeze(-1), True)
    minimum = _distance_column(distance, anchor)
    weight = torch.ones_like(minimum) if demand_weight is None else demand_weight
    for _ in range(1, C4):
        improved = torch.minimum(minimum.unsqueeze(-1), distance)
        gain = ((minimum.unsqueeze(-1) - improved) * weight.unsqueeze(-1)).sum(-2)
        index = gain.masked_fill(used, -1.0).argmax(dim=-1)
        selected.append(index)
        used.scatter_(-1, index.unsqueeze(-1), True)
        minimum = torch.minimum(minimum, _distance_column(distance, index))
    return torch.stack(selected, dim=-1)


def _anchored_qr(points: torch.Tensor, anchor: torch.Tensor, normalize: bool) -> torch.Tensor:
    """Select real rows by pivoted residual span after a fixed anchor."""

    source = points.float()
    if normalize:
        source = F.normalize(source, dim=-1, eps=1.0e-8)
    prefix = source.shape[:-2]
    count = source.size(-2)
    anchor = anchor.expand(prefix)
    selected = [anchor]
    used = torch.zeros(*prefix, count, device=source.device, dtype=torch.bool)
    used.scatter_(-1, anchor.unsqueeze(-1), True)
    residual = source.clone()
    for step in range(C4):
        index = selected[-1]
        basis = source.gather(
            -2,
            index[..., None, None].expand(*index.shape, 1, source.size(-1)),
        ).squeeze(-2)
        basis = F.normalize(basis, dim=-1, eps=1.0e-8)
        projection = (residual * basis.unsqueeze(-2)).sum(-1, keepdim=True)
        residual = residual - projection * basis.unsqueeze(-2)
        if step + 1 == C4:
            break
        score = residual.square().sum(-1).masked_fill(used, -1.0)
        next_index = score.argmax(dim=-1)
        selected.append(next_index)
        used.scatter_(-1, next_index.unsqueeze(-1), True)
    return torch.stack(selected, dim=-1)


def _global_farthest_order(points: torch.Tensor, count: int) -> torch.Tensor:
    """Build one query-only farthest-point traversal per query column.

    The row closest to the query centroid is the deterministic first center.
    Subsequent rows maximize distance to the nearest selected center.  The
    result has shape ``[J, count]`` and is independent of the target key row.
    """

    if points.ndim != 3:
        raise ValueError("points must have shape [column, row, dimension]")
    if not 1 <= count <= points.size(-2):
        raise ValueError("count must be between one and the number of rows")
    source = points.float()
    distance = _pairwise_squared(source)
    centroid = source.mean(dim=-2, keepdim=True)
    first = (source - centroid).square().sum(dim=-1).argmin(dim=-1)
    selected = [first]
    used = torch.zeros(
        source.size(0), source.size(1), device=source.device, dtype=torch.bool
    )
    used.scatter_(-1, first.unsqueeze(-1), True)
    minimum = _distance_column(distance, first)
    for _ in range(1, count):
        index = minimum.masked_fill(used, -1.0).argmax(dim=-1)
        selected.append(index)
        used.scatter_(-1, index.unsqueeze(-1), True)
        minimum = torch.minimum(minimum, _distance_column(distance, index))
    return torch.stack(selected, dim=-1)


def _global_c4_rows(points: torch.Tensor) -> torch.Tensor:
    """Variant 1: share four global query centers across all target rows."""

    order = _global_farthest_order(points, C4)
    return order.unsqueeze(1).expand(-1, points.size(-2), -1)


def _anchor_plus_global_rows(points: torch.Tensor) -> torch.Tensor:
    """Variant 2: matched-row anchor plus three shared global centers."""

    rows = points.size(-2)
    # Four global entries guarantee three remain after excluding an anchor
    # that happens to coincide with one of the shared centers.
    order = _global_farthest_order(points, C4)
    anchor = torch.arange(rows, device=points.device).view(1, rows)
    anchor = anchor.expand(points.size(0), -1)
    candidates = order.unsqueeze(1).expand(-1, rows, -1)
    priority = torch.arange(C4, device=points.device).view(1, 1, C4)
    priority = priority.expand_as(candidates)
    priority = priority.masked_fill(candidates == anchor.unsqueeze(-1), C4)
    positions = priority.argsort(dim=-1)[..., : C4 - 1]
    global_rows = candidates.gather(-1, positions)
    return torch.cat((anchor.unsqueeze(-1), global_rows), dim=-1)


def _anchor_refine_global_pool_rows(
    points: torch.Tensor, pool_size: int
) -> torch.Tensor:
    """Variant 3: global coreset followed by cheap anchor-conditioned FPS."""

    rows = points.size(-2)
    pool_rows = _global_farthest_order(points, pool_size)
    columns = torch.arange(points.size(0), device=points.device)[:, None]
    pool = points[columns, pool_rows].float()
    pool_distance = _pairwise_squared(pool)
    anchor = torch.arange(rows, device=points.device).view(1, rows)
    anchor = anchor.expand(points.size(0), -1)
    minimum = (points.float().unsqueeze(-2) - pool.unsqueeze(1)).square().sum(-1)
    used = pool_rows.unsqueeze(1) == anchor.unsqueeze(-1)
    selected_rows = [anchor]
    expanded_pool_distance = pool_distance.unsqueeze(1).expand(
        -1, rows, -1, -1
    )
    for _ in range(1, C4):
        position = minimum.masked_fill(used, -1.0).argmax(dim=-1)
        selected_rows.append(
            pool_rows.unsqueeze(1).expand(-1, rows, -1).gather(
                -1, position.unsqueeze(-1)
            ).squeeze(-1)
        )
        used.scatter_(-1, position.unsqueeze(-1), True)
        minimum = torch.minimum(
            minimum, _distance_column(expanded_pool_distance, position)
        )
    return torch.stack(selected_rows, dim=-1)


@torch.inference_mode()
def select_query_only_rows(
    query: torch.Tensor,
    candidate_count: int = 15,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Return rows[J, FRAMES*ROWS, C] without accepting a key tensor."""

    candidate_rows, anchor = _candidate_rows(query.device, candidate_count)
    points = _gather_query_rows(query, candidate_rows).float()
    anchor = anchor.unsqueeze(0).expand(query.size(0), -1)
    norm_importance = points.square().mean(-1)
    norm_importance = torch.sqrt(
        norm_importance / norm_importance.mean(-1, keepdim=True).clamp_min(1.0e-12)
    )

    selected: dict[str, torch.Tensor] = {}
    elapsed: dict[str, float] = {}

    def synchronize() -> None:
        if query.is_cuda:
            torch.cuda.synchronize(query.device)

    def record(name: str, function) -> None:
        synchronize()
        started = time.perf_counter()
        position = function()
        synchronize()
        elapsed[name] = time.perf_counter() - started
        selected[name] = candidate_rows.unsqueeze(0).expand(query.size(0), -1, -1).gather(
            -1, position
        )

    def record_rows(name: str, function) -> None:
        synchronize()
        started = time.perf_counter()
        rows = function()
        synchronize()
        elapsed[name] = time.perf_counter() - started
        selected[name] = rows

    fixed = _production_wide_rows(query.device)
    selected["fixed_wide"] = fixed.unsqueeze(0).expand(query.size(0), -1, -1)
    elapsed["fixed_wide"] = 0.0

    position = torch.arange(candidate_count, device=query.device, dtype=torch.float32)
    index_distance = (position[:, None] - position[None, :]).square()
    index_distance = index_distance.view(1, 1, candidate_count, candidate_count).expand(
        query.size(0), ROWS, -1, -1
    )
    record("spatial_kcenter", lambda: _anchored_farthest(index_distance, anchor))

    l2_distance = _pairwise_squared(points)
    record("query_l2_kcenter", lambda: _anchored_farthest(l2_distance, anchor))
    for sketch_dimensions in (16, 32, 64):
        sketch_distance = _pairwise_squared(points[..., :sketch_dimensions])
        record(
            f"query_l2_kcenter_d{sketch_dimensions}",
            lambda distance=sketch_distance: _anchored_farthest(
                distance, anchor
            ),
        )
    record(
        "query_l2_norm_kcenter",
        lambda: _anchored_farthest(l2_distance, anchor, norm_importance),
    )

    centered = points - points.mean(dim=-2, keepdim=True)
    response = torch.einsum("...pd,...rd->...pr", points, centered)
    self_fisher_distance = _pairwise_squared(response)
    record(
        "query_self_fisher_kcenter",
        lambda: _anchored_farthest(self_fisher_distance, anchor, norm_importance),
    )

    increments = (points[..., 1:, :] - points[..., :-1, :]).norm(dim=-1)
    arc = F.pad(increments.cumsum(dim=-1), (1, 0))
    arc_distance = (arc.unsqueeze(-1) - arc.unsqueeze(-2)).square()
    record("query_arc_kcenter", lambda: _anchored_farthest(arc_distance, anchor))

    pseudo_query = 0.5 * (points[..., 1:, :] + points[..., :-1, :])
    pseudo_query = pseudo_query - pseudo_query.mean(dim=-2, keepdim=True)
    surrogate_logits = torch.einsum(
        "...pd,...rd->...pr", points, pseudo_query
    ) / math.sqrt(points.size(-1))
    surrogate_probability = torch.softmax(surrogate_logits, dim=-1)
    hellinger_embedding = torch.sqrt(surrogate_probability.clamp_min(1.0e-12))
    hellinger_distance = _pairwise_squared(hellinger_embedding)
    record(
        "query_surrogate_hellinger_medoid",
        lambda: _anchored_facility(
            hellinger_distance, anchor, norm_importance
        ),
    )

    row_proxy = query.permute(1, 0, 2).float()
    row_proxy_logits = torch.einsum(
        "jbpd,bld->jbpl", points, row_proxy
    ) / math.sqrt(points.size(-1))
    row_proxy_log_probability = torch.log_softmax(row_proxy_logits, dim=-1)
    row_proxy_probability = row_proxy_log_probability.exp()
    row_proxy_hellinger = torch.sqrt(row_proxy_probability.clamp_min(1.0e-12))
    row_proxy_hellinger_distance = _pairwise_squared(row_proxy_hellinger)
    record(
        "query_row_proxy_hellinger_kcenter",
        lambda: _anchored_farthest(
            row_proxy_hellinger_distance, anchor, norm_importance
        ),
    )
    record(
        "query_row_proxy_hellinger_medoid",
        lambda: _anchored_facility(
            row_proxy_hellinger_distance, anchor, norm_importance
        ),
    )

    row_proxy_cross_entropy = -torch.einsum(
        "...il,...jl->...ij", row_proxy_probability, row_proxy_log_probability
    )
    row_proxy_entropy = -(
        row_proxy_probability * row_proxy_log_probability
    ).sum(-1)
    row_proxy_kl = row_proxy_cross_entropy - row_proxy_entropy.unsqueeze(-1)
    row_proxy_symmetric_kl = 0.5 * (
        row_proxy_kl + row_proxy_kl.transpose(-1, -2)
    )
    record(
        "query_row_proxy_symmetric_kl_medoid",
        lambda: _anchored_facility(
            row_proxy_symmetric_kl, anchor, norm_importance
        ),
    )

    record("query_qr_raw", lambda: _anchored_qr(points, anchor, normalize=False))
    record("query_qr_cosine", lambda: _anchored_qr(points, anchor, normalize=True))

    global_points = query[..., : min(32, query.size(-1))]
    record_rows(
        "query_global_anchor_plus3_d32",
        lambda: _anchor_plus_global_rows(global_points),
    )
    record_rows(
        "query_global_c4_d32",
        lambda: _global_c4_rows(global_points),
    )
    for pool_size in (8, 12):
        record_rows(
            f"query_global_pool{pool_size}_local_c4_d32",
            lambda size=pool_size: _anchor_refine_global_pool_rows(
                global_points, size
            ),
        )

    block_key_row = torch.arange(FRAMES * ROWS, device=query.device) % ROWS
    expanded = {name: rows[:, block_key_row] for name, rows in selected.items()}
    return expanded, elapsed


def _gather_seeds(query: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    columns = torch.arange(query.size(0), device=query.device)[:, None, None]
    return query[columns, rows]


@torch.inference_mode()
def evaluate_attention_kl(
    query: torch.Tensor,
    key: torch.Tensor,
    rows: torch.Tensor,
    column_chunk: int = 2,
) -> dict[str, float | list[float]]:
    """Evaluate dense-to-induced-GMA KL over original key tokens."""

    scale = query.size(-1) ** -0.5
    all_full = []
    all_cluster = []
    active_sum = 0.0
    active_count = 0

    for start in range(0, query.size(0), column_chunk):
        end = min(start + column_chunk, query.size(0))
        query_chunk = query[start:end].float()
        rows_chunk = rows[start:end]
        seeds = _gather_seeds(query_chunk, rows_chunk).float()
        seed_logits = torch.einsum("jbcd,bld->jbcl", seeds, key.float()) * scale
        owner = torch.log_softmax(seed_logits, dim=-1).argmax(dim=-2)
        cluster_id = torch.arange(C4, device=query.device).view(1, 1, C4, 1)
        assignment = owner.unsqueeze(-2) == cluster_id
        active = assignment.any(dim=-1)
        active_sum += float(active.float().sum())
        active_count += active.numel()

        masked_seed_logits = seed_logits.masked_fill(~assignment, -float("inf"))
        safe_seed_logits = torch.where(
            active.unsqueeze(-1), masked_seed_logits, torch.zeros_like(masked_seed_logits)
        )
        seed_log_partition = torch.logsumexp(masked_seed_logits, dim=-1)
        seed_log_probability = torch.log_softmax(safe_seed_logits, dim=-1)
        seed_probability = torch.where(
            assignment, seed_log_probability.exp(), torch.zeros_like(seed_log_probability)
        )
        merged_key = torch.einsum("jbcl,bld->jbcd", seed_probability, key.float())
        tangent_bias = (seeds * merged_key).sum(-1) * scale - seed_log_partition

        approximate_cluster_logits = (
            torch.einsum("jrd,jbcd->jrbc", query_chunk, merged_key) * scale
            - tangent_bias.unsqueeze(1)
        )
        approximate_cluster_logits = approximate_cluster_logits.masked_fill(
            ~active.unsqueeze(1), -float("inf")
        )
        approximate_cluster_log_probability = torch.log_softmax(
            approximate_cluster_logits.flatten(-2), dim=-1
        ).reshape_as(approximate_cluster_logits)

        dense_logits = torch.einsum("jrd,bld->jrbl", query_chunk, key.float()) * scale
        dense_log_partition = torch.logsumexp(dense_logits.flatten(-2), dim=-1)
        dense_log_probability = dense_logits - dense_log_partition[..., None, None]
        dense_probability = dense_log_probability.exp()

        owner_for_queries = owner.unsqueeze(1).expand(
            -1, query_chunk.size(1), -1, -1
        )
        cluster_log_probability_for_key = (
            approximate_cluster_log_probability.gather(
                -1, owner_for_queries
            )
        )
        within_log_probability_for_key = (
            seed_log_probability.permute(0, 1, 3, 2)
            .gather(-1, owner.unsqueeze(-1))
            .squeeze(-1)
            .unsqueeze(1)
        )
        induced_log_probability = (
            cluster_log_probability_for_key + within_log_probability_for_key
        )
        full_kl = (
            dense_probability * (dense_log_probability - induced_log_probability)
        ).sum(dim=(-2, -1)).clamp_min(0.0)

        dense_cluster_probability = torch.zeros_like(
            approximate_cluster_log_probability
        )
        dense_cluster_probability.scatter_add_(
            -1, owner_for_queries, dense_probability
        )
        dense_cluster_log_probability = torch.where(
            dense_cluster_probability > 0.0,
            dense_cluster_probability.clamp_min(1.0e-30).log(),
            torch.zeros_like(dense_cluster_probability),
        )
        cluster_support = dense_cluster_probability > 0.0
        cluster_kl_term = torch.where(
            cluster_support,
            dense_cluster_probability
            * (
                dense_cluster_log_probability
                - approximate_cluster_log_probability
            ),
            torch.zeros_like(dense_cluster_probability),
        )
        cluster_kl = cluster_kl_term.sum(dim=(-2, -1)).clamp_min(0.0)

        all_full.append(full_kl.cpu())
        all_cluster.append(cluster_kl.cpu())

    full = torch.cat(all_full).flatten()
    cluster = torch.cat(all_cluster).flatten()
    conditional = (full - cluster).clamp_min(0.0)
    return {
        "mean_key_kl": float(full.mean()),
        "median_key_kl": float(full.median()),
        "p95_key_kl": float(torch.quantile(full, 0.95)),
        "max_key_kl": float(full.max()),
        "mean_cluster_kl": float(cluster.mean()),
        "mean_conditional_kl": float(conditional.mean()),
        "active_cluster_fraction": active_sum / active_count,
        "queries": full.numel(),
    }


def _aggregate(records: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    output = {}
    for name, values in records.items():
        queries = sum(int(value["queries"]) for value in values)
        output[name] = {
            key: sum(float(value[key]) * int(value["queries"]) for value in values) / queries
            for key in (
                "mean_key_kl",
                "mean_cluster_kl",
                "mean_conditional_kl",
                "active_cluster_fraction",
            )
        }
        output[name]["cases"] = len(values)
        output[name]["queries"] = queries
    baseline = output["fixed_wide"]["mean_key_kl"]
    for value in output.values():
        value["mean_key_kl_ratio_vs_fixed"] = value["mean_key_kl"] / baseline
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", default="0,10,20,30")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate_count", type=int, default=15)
    parser.add_argument("--column_chunk", type=int, default=2)
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated selector names to evaluate.",
    )
    parser.add_argument(
        "--router_frame",
        type=int,
        default=-1,
        help="Use one query frame's seeds for every evaluated frame; -1 selects per frame.",
    )
    args = parser.parse_args()

    heads = _parse_indices(args.heads)
    frames = _parse_indices(args.frames)
    requested_methods = {
        item.strip() for item in args.methods.split(",") if item.strip()
    }
    if args.router_frame < -1 or args.router_frame >= FRAMES:
        raise ValueError(f"router_frame must be -1 or in [0, {FRAMES - 1}]")
    device = torch.device(args.device)
    totals: dict[str, list[dict[str, float]]] = defaultdict(list)
    cases = {}
    selector_seconds: dict[str, list[float]] = defaultdict(list)

    for path in args.qkv:
        q_heads, k_heads = _load_qk(path, heads)
        capture = {}
        for head_slot, head in enumerate(heads):
            q_grid = q_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(device)
            k_grid = k_heads[head_slot].reshape(FRAMES, ROWS, COLS, -1).to(device)
            key_blocks = k_grid.reshape(FRAMES * ROWS, COLS, -1)
            shared_rows = None
            shared_elapsed = None
            if args.router_frame >= 0:
                router_query = (
                    q_grid[args.router_frame].permute(1, 0, 2).contiguous()
                )
                shared_rows, shared_elapsed = select_query_only_rows(
                    router_query, args.candidate_count
                )
            for frame in frames:
                query = q_grid[frame].permute(1, 0, 2).contiguous()
                if shared_rows is None:
                    rows, elapsed = select_query_only_rows(
                        query, args.candidate_count
                    )
                else:
                    rows = shared_rows
                    elapsed = {
                        name: seconds / len(frames)
                        for name, seconds in shared_elapsed.items()
                    }
                if requested_methods:
                    missing = requested_methods.difference(rows)
                    if missing:
                        raise ValueError(
                            f"unknown selector methods: {sorted(missing)}"
                        )
                    rows = {
                        name: table
                        for name, table in rows.items()
                        if name in requested_methods
                    }
                    elapsed = {
                        name: seconds
                        for name, seconds in elapsed.items()
                        if name in requested_methods
                    }
                case_name = f"{path.stem}:h{head}:f{frame}"
                case = {}
                for name, seed_rows in rows.items():
                    metrics = evaluate_attention_kl(
                        query, key_blocks, seed_rows, args.column_chunk
                    )
                    case[name] = metrics
                    totals[name].append(metrics)
                    selector_seconds[name].append(elapsed[name])
                capture[case_name] = case
                ordered = sorted(case, key=lambda name: case[name]["mean_key_kl"])
                print(
                    f"{case_name} "
                    + " ".join(
                        f"{name}={case[name]['mean_key_kl']:.6f}"
                        for name in ordered[:4]
                    ),
                    flush=True,
                )
                del rows
                torch.cuda.empty_cache()
            del q_grid, k_grid, key_blocks
            torch.cuda.empty_cache()
        cases[path.stem] = capture
        del q_heads, k_heads

    aggregate = _aggregate(totals)
    for name, values in selector_seconds.items():
        aggregate[name]["selector_python_ms_per_case"] = 1000.0 * sum(values) / len(values)
    ranking = sorted(aggregate, key=lambda name: aggregate[name]["mean_key_kl"])
    result = {
        "settings": {
            **vars(args),
            "qkv": [str(path) for path in args.qkv],
            "heads": heads,
            "frames": frames,
            "device_name": torch.cuda.get_device_name(device),
            "selector_reads_key": False,
            "all_seeds_are_real_queries": True,
            "attention_kl_direction": "KL(dense || induced_gma)",
        },
        "aggregate": aggregate,
        "ranking": ranking,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("\naggregate")
    for name in ranking:
        value = aggregate[name]
        print(
            f"{name:>34s} KL={value['mean_key_kl']:.8f} "
            f"ratio={value['mean_key_kl_ratio_vs_fixed']:.4f} "
            f"cluster={value['mean_cluster_kl']:.8f} "
            f"conditional={value['mean_conditional_kl']:.8f}",
            flush=True,
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
